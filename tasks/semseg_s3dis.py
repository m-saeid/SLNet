# tasks/main.py

from tasks.imports_s3dis import *

def _update_confmat(confmat: torch.Tensor, preds: torch.Tensor,
                    targets: torch.Tensor, num_classes: int) -> None:
    """Vectorized confusion matrix update, ignores label=-100."""
    valid = (targets >= 0) & (targets < num_classes)
    if not valid.any():
        return
    p = preds[valid]
    t = targets[valid]
    idx = (num_classes * t + p).to(torch.int64)
    confmat += torch.bincount(idx, minlength=num_classes ** 2).view(num_classes, num_classes)

def compute_metrics(confmat: np.ndarray):
    tp = np.diag(confmat)
    fp = confmat.sum(0) - tp
    fn = confmat.sum(1) - tp
    denom_iou = tp + fp + fn
    denom_acc = confmat.sum(1)  # per-class total GT points

    with np.errstate(divide='ignore', invalid='ignore'):
        iou  = np.where(denom_iou > 0, tp / denom_iou, np.nan)
        cacc = np.where(denom_acc > 0, tp / denom_acc, np.nan)  # per-class acc

    mean_iou = float(np.nanmean(iou))
    mean_acc = float(np.nanmean(cacc))   # mAcc
    overall_acc = float(tp.sum() / confmat.sum()) if confmat.sum() > 0 else 0.0
    return iou, cacc, mean_iou, mean_acc, overall_acc


# ─────────────────────────── Train Epoch ────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scheduler,                          # per-batch or per-epoch, detected automatically
    scaler: Optional[GradScaler],
    device: torch.device,
    num_classes: int = 13,
    accum_steps: int = 1,
    grad_clip: float = 1.0,
    use_amp: bool = True,
    log_interval: int = 50,
    logger=None,
    ema_model=None,
) -> dict:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    # GPU accumulators
    run_loss = torch.tensor(0.0, device=device)
    run_correct = torch.tensor(0, device=device, dtype=torch.int64)
    run_tokens = 0
    confmat = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)

    step_per_batch = getattr(scheduler, 'step_per_batch', False)

    for batch_idx, batch in enumerate(loader):
        x = batch['x'].to(device, non_blocking=True)          # (B, N, F)
        y = batch['y'].to(device, non_blocking=True)           # (B, N)
        feats = x.permute(0, 2, 1).contiguous()               # (B, F, N)

        with autocast("cuda", enabled=use_amp):
            logits = model(feats)                              # (B, C, N) or (B, N, C)
            # Normalize to (B, N, C)
            if logits.size(1) == num_classes and logits.size(2) != num_classes:
                logits = logits.permute(0, 2, 1).contiguous()
            B, N, C = logits.shape
            loss = criterion(logits.reshape(-1, C), y.reshape(-1))
            loss_scaled = loss / accum_steps

        if use_amp:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        is_step = (batch_idx + 1) % accum_steps == 0
        is_last = (batch_idx + 1) == len(loader)

        if is_step or is_last:
            if use_amp:
                scaler.unscale_(optimizer)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            if step_per_batch:
                scheduler.step()

            if ema_model is not None:
                ema_model.update_parameters(model)

        # ── Metrics (no-grad, GPU) ─────────────────────────────────────────
        with torch.no_grad():
            tokens = B * N
            run_loss += loss.detach() * tokens
            run_tokens += tokens
            preds = logits.detach().argmax(dim=-1)             # (B, N)
            run_correct += (preds == y).sum()
            _update_confmat(confmat, preds.reshape(-1), y.reshape(-1), num_classes)

        if logger and (batch_idx + 1) % log_interval == 0:
            avg_l = run_loss.item() / max(run_tokens, 1)
            avg_a = run_correct.item() / max(run_tokens, 1)
            iou, cacc, mean_iou, mean_acc, overall_acc = compute_metrics(confmat.cpu().numpy())
            # logger.info(f"  [train] step {batch_idx+1}/{len(loader)} "
            #            f"loss={avg_l:.4f} acc={avg_a:.4f} mIoU={mean_iou:.4f}")

    # ── Epoch-level scheduler step ─────────────────────────────────────────
    if not step_per_batch and scheduler is not None:
        scheduler.step()

    # ── Final metrics ──────────────────────────────────────────────────────
    confmat_np = confmat.cpu().numpy()
    iou, cacc, mean_iou, mean_acc, overall_acc = compute_metrics(confmat_np)
    avg_loss = run_loss.item() / max(run_tokens, 1)

    return {
        'loss': avg_loss,
        'oa': overall_acc,
        'miou': mean_iou,
        'macc': mean_acc,
        'per_class_iou': iou,
        'per_class_acc': cacc,
    }



# ─────────────────────────── Val Epoch ────────────────────────────────────────────


CLASS_NAMES = ['ceiling','floor','wall','beam','column','window',
               'door','chair','table','bookcase','sofa','board','clutter']

def val_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 13,
    use_amp: bool = True,
    logger=None,
) -> dict:
    """
    Validation loop with sliding-window voting.

    Expected batch dict keys:
        x          : (B, N, F)   float32   features
        y          : (B, N)      int64     labels (per-point original room labels for blocks)
        room_id    : (B,)        int64     which room each crop belongs to
        point_idx  : (B, N)      int64     original point indices in room (for aggregation)

    Returns dict with keys: loss, oa, miou, per_class_iou
    """
    model.eval()

    # Room accumulators — lazy init on first batch seeing a room_id
    # room_id -> (N_room, num_classes) logit sum  [GPU]
    logit_acc: Dict[int, torch.Tensor] = {}
    count_acc: Dict[int, torch.Tensor] = {}
    # room_id -> (N_room,) ground truth labels     [CPU numpy]
    gt_labels: Dict[int, np.ndarray] = {}

    run_loss = 0.0
    run_tokens = 0

    # Cache of ones tensors to avoid repeated allocation in index_add_
    ones_cache = {}  # size → tensor, avoids repeated torch.ones() calls

    with torch.inference_mode():
        for batch in loader:
            x = batch['x'].to(device, non_blocking=True)          # (B, N, F)
            y = batch['y'].to(device, non_blocking=True)           # (B, N)
            room_ids = batch['room_id']                            # (B,) — keep CPU for dict lookup
            point_idx = batch['point_idx'].to(device, non_blocking=True)  # (B, N)

            feats = x.permute(0, 2, 1).contiguous()               # (B, F, N)

            with autocast("cuda", enabled=use_amp):
                logits = model(feats)                              # (B, C, N) or (B, N, C)
                if logits.size(1) == num_classes and logits.size(2) != num_classes:
                    logits = logits.permute(0, 2, 1).contiguous()  # (B, N, C)
                B, N, C = logits.shape
                loss = criterion(logits.reshape(-1, C), y.reshape(-1))

            run_loss += loss.item() * B * N
            run_tokens += B * N

            # ── Per-room vectorized accumulation ──────────────────────────
            # logits: (B, N, C) float  — may be fp16 under AMP, cast once
            logits_f = logits.float()   # (B, N, C)

            # Gather ground-truth for each room (from dataset, done once per room_id)
            dataset = loader.dataset
            for bi in range(B):
                rid = int(room_ids[bi])
                if rid not in logit_acc:
                    n_room = int(dataset.room_xyz[rid].shape[0])  # ← use full room size, not just max point_idx
                    logit_acc[rid] = torch.zeros(n_room, C, device=device)
                    count_acc[rid] = torch.zeros(n_room, device=device)
                    # gt_labels[rid] = dataset.cache.get(rid)[:, 6].view(np.int64).copy()   ##################################################
                    gt_labels[rid] = dataset.cache.get(rid)[:, 6].astype(np.int64).copy()

                idxs = point_idx[bi]                               # (N,) long
                # vectorized: scatter logits into room accumulator
                # index_add_(dim, index, src) — adds src[i] into self[index[i]]
                logit_acc[rid].index_add_(0, idxs, logits_f[bi])  # (N_room, C) += (N, C)

                if N not in ones_cache:
                    ones_cache[N] = torch.ones(N, device=device)
                count_acc[rid].index_add_(0, idxs, ones_cache[N])

                # count_acc[rid].index_add_(0, idxs,
                #     torch.ones(N, device=device))                  # (N_room,) += 1

    # ── Aggregate predictions and compute metrics ─────────────────────────
    confmat = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)

    for rid in logit_acc:
        counts = count_acc[rid]                                    # (N_room,)
        valid_pts = counts > 0
        if not valid_pts.any():
            continue

        avg_logits = logit_acc[rid].clone()
        avg_logits[valid_pts] /= counts[valid_pts].unsqueeze(1)
        preds = avg_logits.argmax(dim=1)                           # (N_room,)

        gt = torch.from_numpy(gt_labels[rid].astype(np.int64)).to(device)
        # Only evaluate points that were actually covered
        mask = valid_pts & (gt >= 0) & (gt < num_classes)
        if mask.any():
            _update_confmat(confmat, preds[mask], gt[mask], num_classes)

    confmat_np = confmat.cpu().numpy()
    iou, cacc, mean_iou, mean_acc, oa = compute_metrics(confmat_np)
    avg_loss = run_loss / max(run_tokens, 1)

    if logger:
        logger.info(f"  [val] loss={avg_loss:.4f} OA={oa:.4f} mIoU={mean_iou:.4f} mAcc={mean_acc:.4f}")
        for i, (name, v, a) in enumerate(zip(CLASS_NAMES, iou, cacc)):
            logger.info(f"    {name:<12} IoU={v:.4f}  Acc={a:.4f}")

    return {
        'loss': avg_loss,
        'oa': oa,
        'miou': mean_iou,
        'macc': mean_acc,
        'per_class_iou': iou,
        'per_class_acc': cacc,
         }



# ─────────────────────────── Main ────────────────────────────────────────────

def main(cfg: Config, no_compile: bool = False):
    # ── torch.compile control ─────────────────────────────────────────────
    # Pass --no-compile to disable Dynamo tracing entirely.
    # Useful for debugging: removes graph breaks from pybind11 ops (FPS, kNN)
    # and makes stack traces readable.  Re-enable once correctness confirmed.

    inspect(cfg, title="Experiment Configuration", methods=False)

    if no_compile:
        torch._dynamo.config.suppress_errors = True
        torch.compile = lambda model, *a, **k: model   # no-op compile

    # ── Experiment dir ────────────────────────────────────────────────────
    # exp_name = cfg.exp_name or f"s3dis_area{cfg.test_area}_{datetime.now():%Y%m%d-%H%M%S}"

    # Replace the exp_name assignment:
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    exp_name = cfg.exp_name or f"SLNetT_area{cfg.test_area}_{cfg.encoder_type}_{cfg.loss_type}_{ts}"

    ckpt_dir = Path(cfg.log_dir) / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(str(ckpt_dir))
    logger.info(f"Experiment: {ckpt_dir}")
    logger.info(json.dumps(vars(cfg), indent=2, default=str))
    if no_compile:
        logger.info("torch.compile DISABLED (--no-compile)")

    # ── WandB Init ────────────────────────────────────────────────────────────
    run = None
    if cfg.use_wandb:
        system_info = get_system_info()
        wandb_config = {**vars(cfg), **system_info, 'exp_dir': str(ckpt_dir)}
        run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity or None,
            name=exp_name,
            config=wandb_config,
            dir=str(ckpt_dir),
            tags=[
                f'area{cfg.test_area}',
                cfg.encoder_type,
                cfg.loss_type,
                f'bs{cfg.batch_size}',
                'SLNet-T',
            ],
        )
        logger.info(f"WandB run: {run.url}")

    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark    = True
    torch.backends.cuda.matmul.allow_tf32  = True
    torch.backends.cudnn.allow_tf32   = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────
    
    train_loader = build_train_loader(
        data_root=cfg.data_dir,
        test_area=cfg.test_area,
        num_points=cfg.num_points,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        samples_per_room=cfg.samples_per_room,
        crop_radius=cfg.crop_radius,
    )
    

    val_loader = build_test_loader(
        data_root=cfg.data_dir,
        test_area=cfg.test_area,
        num_points=cfg.num_points,
        num_workers=max(2, cfg.num_workers // 2),
        test_stride=cfg.test_stride,
        batch_size_test=cfg.batch_size_test,
        crop_radius=cfg.crop_radius,
    )
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = SegModel(cfg).to(device)
    logger.info(f"Parameters: {param_count(model)}")

    if not no_compile and torch.__version__ >= "2.0" and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead')
            logger.info("torch.compile() enabled")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")

    if cfg.use_ema:
        ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(cfg.ema_decay))
        logger.info(f"EMA enabled (decay={cfg.ema_decay})")
    else:
        ema_model = None

    profile_stats = profile_model(model, cfg, device, logger)
    if run:
        wandb.log({'model/' + k: v for k, v in profile_stats.items() if v is not None}, step=0)

    # ── Loss ──────────────────────────────────────────────────────────────
    class_weights = getattr(train_loader.dataset, 'class_weights', None)
    criterion = build_criterion(cfg, class_weights=class_weights, device=device)

    # ── Optimizer + Scheduler + Scaler ────────────────────────────────────
    optimizer = build_optimizer(cfg, model)
    scheduler, step_per_batch = build_scheduler(cfg, optimizer, len(train_loader))
    scheduler.step_per_batch = step_per_batch
    scaler = GradScaler("cuda") if cfg.use_amp else None

    # ── Training loop ─────────────────────────────────────────────────────
    tb = SummaryWriter(log_dir=str(ckpt_dir / 'tb'))
    best_miou = -1.0
    best_epoch = -1
    patience_counter = 0
    t_start = time.time()

    # ── Before training loop: compute rare rooms ONCE ─────────────────
    rare_rooms = precompute_rare_rooms(train_loader.dataset)
    logger.info(f"[sampler] {len(rare_rooms)}/{len(train_loader.dataset.room_files)} "
                f"rooms have rare classes (precomputed)")

    for epoch in range(cfg.epochs):
        t_ep = time.time()
        train_loader.dataset.on_epoch_end()

        # Rebuild sampler — NO file I/O (uses precomputed rare_rooms)
        new_sampler = make_sampler_from_rare_rooms(train_loader.dataset, cfg.rare_room_weight, rare_rooms)
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=cfg.batch_size,
            sampler=new_sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=2 if cfg.num_workers > 0 else None,
            drop_last=True,
        )

        train_metrics = train_epoch(
            model=model, loader=train_loader, optimizer=optimizer,
            criterion=criterion, scheduler=scheduler, scaler=scaler,
            device=device, num_classes=cfg.num_classes,
            accum_steps=cfg.accum_steps, grad_clip=cfg.grad_clip,
            use_amp=cfg.use_amp, log_interval=cfg.log_interval,
            logger=logger, ema_model=ema_model,
        )

        if run:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_metrics['loss'],
                'train/oa':   train_metrics['oa'],
                'train/miou': train_metrics['miou'],
                'train/macc': train_metrics['mean_acc'] if 'mean_acc' in train_metrics else 0.0,   # after Step 1
                'lr': optimizer.param_groups[0]['lr'],
                'gpu/memory_used_mb': torch.cuda.memory_allocated(device) / 1e6,
                'gpu/memory_reserved_mb': torch.cuda.memory_reserved(device) / 1e6,
            }, step=epoch + 1)

        tb.add_scalar('train/loss', train_metrics['loss'], epoch)
        tb.add_scalar('train/oa',   train_metrics['oa'],   epoch)
        tb.add_scalar('train/miou', train_metrics['miou'], epoch)
        tb.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        do_val = ((epoch + 1) % cfg.validate_every == 0) or (epoch == cfg.epochs - 1)
        val_metrics = None
        if do_val:
            val_model = ema_model if (ema_model is not None) else model
            val_metrics = val_epoch(
                model=val_model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                num_classes=cfg.num_classes,
                use_amp=cfg.use_amp,
                logger=logger,
            )

            if run and val_metrics:
                val_log = {
                    'val/loss': val_metrics['loss'],
                    'val/oa':   val_metrics['oa'],
                    'val/miou': val_metrics['miou'],
                    'val/macc': val_metrics['macc'],
                }
                # Per-class IoU
                for i, (name, iou_val) in enumerate(zip(CLASS_NAMES, val_metrics['per_class_iou'])):
                    val_log[f'val/iou_{name}'] = float(iou_val) if not np.isnan(iou_val) else 0.0
                for i, (name, acc_val) in enumerate(zip(CLASS_NAMES, val_metrics['per_class_acc'])):
                    val_log[f'val/acc_{name}'] = float(acc_val) if not np.isnan(acc_val) else 0.0
                wandb.log(val_log, step=epoch + 1)


            tb.add_scalar('val/loss', val_metrics['loss'], epoch)
            tb.add_scalar('val/oa',   val_metrics['oa'],   epoch)
            tb.add_scalar('val/miou', val_metrics['miou'], epoch)

            if val_metrics['miou'] > best_miou:
                best_miou  = val_metrics['miou']
                best_epoch = epoch + 1
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'ema_model': ema_model.state_dict() if ema_model else None,
                    'ema_weights': ema_model.module.state_dict() if ema_model else None,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_miou': best_miou,
                    'cfg': vars(cfg),
                }, ckpt_dir / 'best.pth')

                logger.info(f"  ★★★ New best mIoU={best_miou:.4f} (epoch {best_epoch}) ★★★")

                if run:
                    wandb.run.summary['best_miou'] = best_miou
                    wandb.run.summary['best_epoch'] = best_epoch
                    wandb.run.summary['best_oa']    = val_metrics['oa']
                    wandb.run.summary['best_macc']  = val_metrics['macc']

            else:
                patience_counter += cfg.validate_every

            if patience_counter >= cfg.early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch+1} (no improvement for {patience_counter} epochs)")
                break

        ep_time = time.time() - t_ep
        lr = optimizer.param_groups[0]['lr']
        val_str = (f" | val_L={val_metrics['loss']:.4f} "
                   f"val_miou={val_metrics['miou']:.4f}") if val_metrics else ""

        logger.info(
            f"[bold]Epoch {epoch+1:03d}/{cfg.epochs}[/bold] "
            f"Loss=[red]{train_metrics['loss']:.4f}[/] "
            f"OA=[green]{train_metrics['oa']:.4f}[/] "
            f"mIoU=[bold cyan]{train_metrics['miou']:.4f}[/] "
            f"{val_str} "
            f"lr=[dim]{lr:.2e}[/] t=[clock]{ep_time:.0f}s[/]"
        )


    total = (time.time() - t_start) / 3600
    # logger.info(f"\n{75*'*'}\n*** Done. Best mIoU={best_miou:.4f} @E{best_epoch}. Total: {total:.2f}h ***\n{75*'*'}")
    logger.info(
        f"\n[bold green]{'='*30} TRAINING COMPLETE {'='*30}[/]\n"
        f"[bold white]Best mIoU:[/] [bold yellow]{best_miou:.4f}[/] at Epoch {best_epoch}\n"
        f"[bold white]Total Time:[/] {total:.2f}h\n"
        f"[bold green]{'='*79}[/]"
    )

    tb.close()

    if run:
        artifact = wandb.Artifact(f'model-{exp_name}', type='model')
        artifact.add_file(str(ckpt_dir / 'best.pth'))
        run.log_artifact(artifact)
        run.finish()

    return model


if __name__ == '__main__':
    import traceback
    _no_compile = '--no-compile' in sys.argv
    if _no_compile:
        sys.argv.remove('--no-compile')
    cfg = parse_config()
    try:
        main(cfg, no_compile=_no_compile)
    except Exception as e:
        # Write crash to a crash.log in the experiment directory
        ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        crash_path = Path(cfg.log_dir) / f"crash_{ts}.log"
        with open(crash_path, 'w') as f:
            traceback.print_exc(file=f)
        raise