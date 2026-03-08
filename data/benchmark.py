# benchmark.py

import time
import torch
from torch.utils.data import DataLoader

from s3dis_dataset import S3DISDataset

def benchmark_loader___(loader: DataLoader, n_batches: int = 100):
    """Measure throughput in batches/sec and points/sec."""
    times = []
    total_points = 0

    it = iter(loader)
    # Warm up
    for _ in range(3):
        batch = next(it)

    for _ in range(n_batches):
        t0 = time.perf_counter()
        batch = next(it)
        # Force GPU transfer
        _ = batch['pos'].cuda(non_blocking=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        total_points += batch['pos'].shape[0] * batch['pos'].shape[1]

    avg_t = sum(times) / len(times)
    print(f"Avg batch time:  {avg_t*1000:.1f} ms")
    print(f"Throughput:      {total_points / sum(times):.0f} pts/sec")
    print(f"Batches/sec:     {1/avg_t:.1f}")


def measure_memory_usage(dataset: S3DISDataset):
    """Estimate RAM used by cache."""
    import sys
    if dataset.cache.mode == 'ram':
        total = sum(v.nbytes for v in dataset.cache._cache.values())
        print(f"RAM cache: {total / 1e9:.2f} GB")
    total_kdtrees = sum(sys.getsizeof(t) for t in dataset.kdtrees)
    print(f"KD-trees: ~{total_kdtrees / 1e6:.1f} MB")