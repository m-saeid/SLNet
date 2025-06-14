import re
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = re.findall('(.*)/tasks', BASE_DIR)[0]
sys.path.append(BASE_DIR)

from utils.cls_fewshot_util import *

def main():
    args = parse_args()

    args.model = 'cls_fewshot'
    args.embed = [args.initial_dim, args.embed_dim, args.embed, args.alpha_beta, args.sigma]
    args.dim_ratio = list(map(lambda x: int(x), args.dim_ratio.split('-')))

    args.num_blocks1 = list(map(lambda x: int(x), args.num_blocks1.split('-')))
    args.transfer_mode = list(map(lambda x: str(x), args.transfer_mode.split('-')))
    args.block1_mode = list(map(lambda x: str(x), args.block1_mode.split('-')))

    args.num_blocks2 = list(map(lambda x: int(x), args.num_blocks2.split('-')))
    args.block2_mode = list(map(lambda x: str(x), args.block2_mode.split('-')))

    args.k_neighbors = list(map(lambda x: int(x), args.k_neighbors.split('-')))
    args.sampling_mode = list(map(lambda x: str(x), args.sampling_mode.split('-')))
    args.sampling_ratio = list(map(lambda x: int(x), args.sampling_ratio.split('-')))

    if args.seed is None:
        args.seed = np.random.randint(1, 10000)
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.set_printoptions(10)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(args.seed)


    OA_fold = []
    mAcc_fold = []

    for fold in tqdm(range(10)):


        time_str = datetime.datetime.now().strftime('-%Y%m%d%H%M%S')
        message = time_str if args.msg is None else "-" + args.msg
        args.checkpoint = f'checkpoints/{args.model}/n{args.n_way}_k{args.k_shot}/fold{fold}/{message}-{str(args.seed)}'
        
        code_dir = f'{args.checkpoint}/code/'
        os.makedirs(code_dir, exist_ok=True)
        shutil.copy('tasks/cls_fewshot.py', f'{code_dir}/cls_fewshot.py')
        shutil.copy('utils/provider.py', f'{code_dir}/provider.py')
        shutil.copy('utils/util.py', f'{code_dir}/util.py')
        shutil.copytree('encoder', f'{code_dir}/encoder')
        shutil.copytree('decoder', f'{code_dir}/encode')
        
        args.path = args.checkpoint
        if not os.path.isdir(args.checkpoint):
            mkdir_p(args.checkpoint)

        screen_logger = logging.getLogger("Model")
        screen_logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        file_handler = logging.FileHandler(os.path.join(args.checkpoint, "out.txt"))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        screen_logger.addHandler(file_handler)

        def printf(str):
            screen_logger.info(str)
            print(str)

        printf(f"args: {args}")
        printf('==> Building model..')
        net = Classification(n=args.n,
                            embed=args.embed,
                            res_dim_ratio=args.res_dim_ratio,
                            bias=args.bias,
                            use_xyz=args.use_xyz,           
                            norm_mode=args.norm_mode,
                            std_mode=args.std_mode,
                            dim_ratio=args.dim_ratio,
                            num_blocks1=args.num_blocks1,
                            transfer_mode=args.transfer_mode,
                            block1_mode=args.block1_mode,
                            num_blocks2=args.num_blocks2,
                            block2_mode=args.block2_mode,
                            k_neighbors=args.k_neighbors,
                            sampling_mode=args.sampling_mode,
                            sampling_ratio=args.sampling_ratio,
                            classifier_mode=args.classifier_mode,
                            fps_method=args.fps_method,
                            knn_method=args.knn_method,
                            )
        
        criterion = cal_loss
        net = net.to(device)
        args.param = trainable_params(net)
        printf(f'number of params: {args.param}')
        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True

        # Initialize AveragedModel for EMA if enabled
        ema_net = None
        if args.ema.lower() == "yes":
            ema_net = AveragedModel(net)
            ema_net = ema_net.to(device)
            printf("AveragedModel (SWA/EMA) is enabled.")

        best_test_acc = 0.0
        best_train_acc = 0.0
        best_test_acc_avg = 0.0
        best_train_acc_avg = 0.0
        best_test_loss = float("inf")
        best_train_loss = float("inf")
        start_epoch = 0
        optimizer_dict = None

        if not os.path.isfile(os.path.join(args.checkpoint, "last_checkpoint.pth")):
            save_args(args)
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title="FewShot" + args.model)
            logger.set_names(["Epoch-Num", 'Learning-Rate',
                            'Train-Loss', 'Train-acc-B', 'Train-acc',
                            'Valid-Loss', 'Valid-acc-B', 'Valid-acc'])
        else:
            printf(f"Resuming last checkpoint from {args.checkpoint}")
            checkpoint_path = os.path.join(args.checkpoint, "last_checkpoint.pth")
            checkpoint = torch.load(checkpoint_path)
            net.load_state_dict(checkpoint['net'])
            if ema_net is not None and "ema_net" in checkpoint:
                ema_net.load_state_dict(checkpoint["ema_net"])
            start_epoch = checkpoint['epoch']
            best_test_acc = checkpoint['best_test_acc']
            best_train_acc = checkpoint['best_train_acc']
            best_test_acc_avg = checkpoint['best_test_acc_avg']
            best_train_acc_avg = checkpoint['best_train_acc_avg']
            best_test_loss = checkpoint['best_test_loss']
            best_train_loss = checkpoint['best_train_loss']
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title="FewShot" + args.model, resume=True)
            optimizer_dict = checkpoint['optimizer']

        printf('==> Preparing data..')
        train_loader = DataLoader(ModelNet40FewShot(partition='train', num_points=args.n, n_way=args.n_way, k_shot=args.k_shot, in_d=3, out_d=args.embed_dim),
                                  num_workers=args.workers, batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(ModelNet40FewShot(partition='test', num_points=args.n, n_way=args.n_way, k_shot=args.k_shot, in_d=3, out_d=args.embed_dim),
                                 num_workers=args.workers, batch_size=args.batch_size // 2, shuffle=False, drop_last=False)

        optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
        if optimizer_dict is not None:
            optimizer.load_state_dict(optimizer_dict)
        scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=args.min_lr, last_epoch=start_epoch - 1)


        train_loader.dataset.set_fold(fold)
        test_loader.dataset.set_fold(fold)
        for epoch in range(start_epoch, args.epoch):
            printf('Epoch(%d/%s) Learning Rate %s:' % (epoch + 1, args.epoch, optimizer.param_groups[0]['lr']))
            
            train_out = train(net, train_loader, optimizer, criterion, device, ema_net)
            test_out = validate(net, test_loader, criterion, device)
            scheduler.step()

            if test_out["acc"] > best_test_acc:
                best_test_acc = test_out["acc"]
                is_best = True
            else:
                is_best = False

            best_test_acc = test_out["acc"] if (test_out["acc"] > best_test_acc) else best_test_acc
            best_train_acc = train_out["acc"] if (train_out["acc"] > best_train_acc) else best_train_acc
            best_test_acc_avg = test_out["acc_avg"] if (test_out["acc_avg"] > best_test_acc_avg) else best_test_acc_avg
            best_train_acc_avg = train_out["acc_avg"] if (train_out["acc_avg"] > best_train_acc_avg) else best_train_acc_avg
            best_test_loss = test_out["loss"] if (test_out["loss"] < best_test_loss) else best_test_loss
            best_train_loss = train_out["loss"] if (train_out["loss"] < best_train_loss) else best_train_loss

            # Save checkpoint including ema_net state
            save_model(
                net, epoch, path=args.checkpoint, acc=test_out["acc"], is_best=is_best, fold=fold,
                best_test_acc=best_test_acc,
                best_train_acc=best_train_acc,
                best_test_acc_avg=best_test_acc_avg,
                best_train_acc_avg=best_train_acc_avg,
                best_test_loss=best_test_loss,
                best_train_loss=best_train_loss,
                optimizer=optimizer.state_dict(),
                ema_net=ema_net.state_dict() if ema_net is not None else None
            )
            # logger.append([epoch, optimizer.param_groups[0]['lr'],
            #            train_out["loss"], train_out["acc_avg"], train_out["acc"],
            #            test_out["loss"], test_out["acc_avg"], test_out["acc"]])
            printf(
                f"Training loss:{train_out['loss']} acc_avg:{train_out['acc_avg']}% acc:{train_out['acc']}% time:{train_out['time']}s")
            printf(
                f"Testing loss:{test_out['loss']} acc_avg:{test_out['acc_avg']}% "
                f"acc:{test_out['acc']}% time:{test_out['time']}s [best test acc: {best_test_acc}%] \n\n")
        logger.close()

        printf("++++++++" * 2 + "Final results" + "++++++++" * 2)
        printf(f"++  Last Train time: {train_out['time']} | Last Test time: {test_out['time']}  ++")
        printf(f"++  Best Train loss: {best_train_loss} | Best Test loss: {best_test_loss}  ++")
        printf(f"++  Best Train acc_B: {best_train_acc_avg} | Best Test acc_B: {best_test_acc_avg}  ++")
        printf(f"++  Best Train acc: {best_train_acc} | Best Test acc: {best_test_acc}  ++")
        printf("++++++++" * 5)

        OA_fold.append(best_test_acc)
        mAcc_fold.append(best_test_acc_avg)

    results = {
        "task": "cls_fewshot",
        "OA_test": best_test_acc,
        "mAcc_test": best_test_acc_avg,
        "OA_train": best_train_acc,
        "mAcc_train": best_train_acc_avg,
        "n_way":args.n_way,
        "k_shot":args.k_shot,
        "fold_OA_mean":statistics.mean(OA_fold),
        "fold_OA_std":statistics.stdev(OA_fold),
        "fold_mAcc_mean":statistics.mean(mAcc_fold),
        "fold_mAcc_std":statistics.stdev(mAcc_fold),
        "OA_fold":OA_fold,
        "mAcc_fold":mAcc_fold,

        "OA1:":OA_fold[0],
        "OA2:":OA_fold[1],
        "OA3:":OA_fold[2],
        "OA4:":OA_fold[3],
        "OA5:":OA_fold[4],
        "OA6:":OA_fold[5],
        "OA7:":OA_fold[6],
        "OA8:":OA_fold[7],
        "OA9:":OA_fold[8],
        "OA10:":OA_fold[9],

        "mAcc1:":mAcc_fold[0],
        "mAcc2:":mAcc_fold[1],
        "mAcc3:":mAcc_fold[2],
        "mAcc4:":mAcc_fold[3],
        "mAcc5:":mAcc_fold[4],
        "mAcc6:":mAcc_fold[5],
        "mAcc7:":mAcc_fold[6],
        "mAcc8:":mAcc_fold[7],
        "mAcc9:":mAcc_fold[8],
        "mAcc10:":mAcc_fold[9],

        "num_param": args.param,
        "path": args.path,
        "n": args.n,
        "embed": args.embed,
        "initial_dim": args.initial_dim,
        "embed_dim": args.embed_dim,
        "res_dim_ratio": args.res_dim_ratio,
        "bias": args.bias,
        "use_xyz": args.use_xyz,
        "norm_mode": args.norm_mode,
        "std_mode": args.std_mode,
        "dim_ratio": args.dim_ratio,
        "num_blocks1": args.num_blocks1,
        "transfer_mode": args.transfer_mode,
        "block1_mode": args.block1_mode,
        "num_blocks2": args.num_blocks2,
        "block2_mode": args.block2_mode,
        "k_neighbors": args.k_neighbors,
        "sampling_mode": args.sampling_mode,
        "sampling_ratio": args.sampling_ratio,
        "classifier_mode": args.classifier_mode,
        "checkpoint": args.checkpoint,
        "msg": args.msg,
        "batch_size": args.batch_size,
        "epoch": f"{args.epoch}_ema:{args.ema}" if args.ema=='yes' else args.epoch,
        "learning_rate": args.learning_rate,
        "min_lr": args.min_lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "workers": args.workers,
    }
    log_experiment(results, excel_path=f'fewshot_n{args.n_way}_k{args.k_shot}.xlsx')


if __name__ == '__main__':
    main()


if 0:#__name__ == '__main__':
    def all_params(model):
        return sum(p.numel() for p in model.parameters())
    def trainable_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    data = torch.rand(2, 3, 1024)
    feature = torch.rand(2, 16, 1024)
    model = Classification()
    print(f'number of params: {trainable_params(model)}') 
    out = model(xyz=data, feature=feature)
    print(out.shape)