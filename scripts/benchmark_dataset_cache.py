"""
Dataset Caching Benchmark
=========================
Evaluates whether caching the training dataset in GPU VRAM or CPU RAM
provides a meaningful speedup over the current H5-from-disk loading.

Usage (single GPU, from project root):
    CUDA_VISIBLE_DEVICES=5,6 python scripts/benchmark_dataset_cache.py --config training/train_config_46M.yaml --measure_compute

What this script does:
  1. Instantiates the model + runs a few forward/backward passes to measure
     peak GPU memory consumed by the model.
  2. Scans the training dataset to estimate total tensor size.
  3. Measures wall-clock throughput (samples/sec) for three strategies:
       A. Current:  H5 on-disk with DataLoader workers (your existing setup)
       B. CPU-RAM:  Preload all tensors into pinned CPU RAM, no disk I/O during training
       C. GPU-VRAM: Preload all tensors into GPU memory (if enough free VRAM exists)
  4. Prints a summary table and a recommendation.
"""

import os
import sys
import time
import argparse
import psutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Make sure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import NeuralRadiosityConfig
from model.global_illumination_model import GlobalIlluminationModel
from training.dataset import NPZSceneDataset as SceneDataset, scene_collate_fn
from training.ray_generator import RayGenerator

# ─── Helpers ────────────────────────────────────────────────────────────────

def bytes_to_gib(b):
    return b / (1024 ** 3)

def get_free_gpu_gib(device):
    """Return (free, total) VRAM in GiB for device."""
    props = torch.cuda.get_device_properties(device)
    total = bytes_to_gib(props.total_memory)
    reserved = bytes_to_gib(torch.cuda.memory_reserved(device))
    free = total - reserved
    return free, total

def get_free_ram_gib():
    """Return (free, total) system RAM in GiB."""
    vm = psutil.virtual_memory()
    return bytes_to_gib(vm.available), bytes_to_gib(vm.total)

def measure_model_peak_vram(model, ray_generator, dataset, batch_size, image_res, device, n_iters=3, measure_compute=False):
    """
    Run n_iters forward+backward passes to get a stable peak VRAM measurement.
    Returns (peak_vram_gib, avg_compute_ms).
    """
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=scene_collate_fn, num_workers=4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.L1Loss()
    
    compute_times = []

    for i, batch in enumerate(loader):
        if i >= n_iters:
            break
        c2w = batch['c2w'].to(device)
        fov_rad = (batch['fov_deg'].to(device) * (torch.pi / 180.0)).unsqueeze(-1)
        rays_o, rays_d = ray_generator(c2w, fov_rad, image_res)
        w2c = torch.inverse(c2w)
        kwargs = dict(
            rays_o=rays_o, rays_d=rays_d,
            obj_positions=batch['obj_positions'].to(device),
            obj_properties=batch['obj_properties'].to(device),
            obj_normals=batch['obj_normals'].to(device),
            obj_mask=batch['obj_mask'].to(device) if 'obj_mask' in batch else None,
            w2c=w2c,
        )
        target = batch['target_image'].to(device)
        
        if measure_compute:
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            pred = model(**kwargs)
        loss = loss_fn(pred.float(), target.float())
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if measure_compute:
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            if i > 0: # skip first iter warmup
                compute_times.append((t1 - t0) * 1000.0)

    peak_bytes = torch.cuda.max_memory_allocated(device)
    avg_compute_ms = sum(compute_times) / len(compute_times) if compute_times else 0.0
    return bytes_to_gib(peak_bytes), avg_compute_ms

def estimate_dataset_gib(dataset, n_probe=20):
    """
    Sample a few items from the dataset and extrapolate the total tensor size in GiB.
    """
    total_bytes = 0
    for i in range(min(n_probe, len(dataset))):
        item = dataset[i]
        for v in item.values():
            total_bytes += v.element_size() * v.numel()
    bytes_per_sample = total_bytes / n_probe
    return bytes_to_gib(bytes_per_sample * len(dataset)), bytes_to_gib(bytes_per_sample)

def benchmark_loader(loader, device, n_batches=20, label=""):
    """
    Measure throughput of a DataLoader: samples/sec over n_batches.
    Returns (samples_per_sec, avg_batch_time_ms).
    """
    total_samples = 0
    times = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        t0 = time.perf_counter()
        # Simulate the .to(device) transfer that happens in the train loop
        _ = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)
        total_samples += batch['target_image'].shape[0]

    avg_ms = (sum(times) / len(times)) * 1000
    sps = total_samples / sum(times)
    print(f"  [{label}] {n_batches} batches | avg {avg_ms:.1f} ms/batch | {sps:.1f} samples/sec")
    return sps, avg_ms


def preload_to_ram(dataset, label="CPU-RAM"):
    """
    Eagerly loads the full dataset into a list of tensors in CPU pinned memory.
    Returns a list of sample dicts.
    """
    print(f"  Preloading {len(dataset)} samples into {label}...")
    cache = []
    for i in range(len(dataset)):
        cache.append(dataset[i])
    return cache


class CachedDataset(torch.utils.data.Dataset):
    """Wraps a preloaded list of sample dicts for use with DataLoader."""
    def __init__(self, cache):
        self.cache = cache
    def __len__(self):
        return len(self.cache)
    def __getitem__(self, idx):
        return self.cache[idx]


class GpuCachedDataset(torch.utils.data.Dataset):
    """Preloads a fixed-shape tensor cache onto the GPU for zero-copy batching."""
    def __init__(self, cache, device):
        print(f"  Moving {len(cache)} samples to GPU {device}...")
        self.cache = [{k: v.to(device) for k, v in item.items()} for item in cache]
    def __len__(self):
        return len(self.cache)
    def __getitem__(self, idx):
        return self.cache[idx]


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='training/train_config_46M.yaml')
    parser.add_argument('--n_batches', type=int, default=30, help="Batches to time per strategy")
    parser.add_argument('--model_iters', type=int, default=3, help="Forward/backward passes to probe peak VRAM")
    parser.add_argument('--measure_compute', action='store_true', help="Measure fw/bw compute time and compare with loading time")
    args = parser.parse_args()

    device = torch.device('cuda:0')
    config = NeuralRadiosityConfig.from_yaml(args.config)
    tc = config.training
    batch_size = tc.batch_size

    print("\n" + "="*60)
    print("  DATASET CACHE FEASIBILITY BENCHMARK")
    print("="*60)

    # ── 1. Build dataset ──────────────────────────────────────────────────
    print(f"\n[1/4] Building dataset from: {tc.data_dir}")
    dataset = SceneDataset(
        data_dir=tc.data_dir,
        image_res=tc.image_res,
        split='train',
        max_dataset_size=None,
        shuffle=True, shuffle_seed=42
    )
    print(f"      Total train samples: {len(dataset)}")

    # ── 2. Probe model VRAM ───────────────────────────────────────────────
    n_probe_iters = max(args.model_iters, 5) if args.measure_compute else args.model_iters
    print(f"\n[2/4] Measuring model peak VRAM ({n_probe_iters} iters, batch={batch_size})...")
    model = GlobalIlluminationModel(config).to(device)
    ray_gen = RayGenerator().to(device)
    peak_model_gib, avg_compute_ms = measure_model_peak_vram(
        model, ray_gen, dataset, batch_size, tc.image_res, device, n_iters=n_probe_iters, measure_compute=args.measure_compute
    )
    free_vram_gib, total_vram_gib = get_free_gpu_gib(device)
    free_ram_gib, total_ram_gib = get_free_ram_gib()
    print(f"      Peak model VRAM:       {peak_model_gib:.2f} GiB")
    print(f"      GPU free after model:  {free_vram_gib:.2f} / {total_vram_gib:.2f} GiB")
    print(f"      System RAM free:       {free_ram_gib:.2f} / {total_ram_gib:.2f} GiB")
    del model, ray_gen
    torch.cuda.empty_cache()

    # ── 3. Estimate dataset size ──────────────────────────────────────────
    print(f"\n[3/4] Estimating dataset tensor footprint (probing 20 samples)...")
    total_ds_gib, per_sample_gib = estimate_dataset_gib(dataset, n_probe=20)
    print(f"      Per-sample:            {per_sample_gib * 1024:.1f} MiB")
    print(f"      Total dataset (~{len(dataset)} samples): {total_ds_gib:.2f} GiB")

    can_gpu_cache = total_ds_gib < free_vram_gib * 0.8
    can_ram_cache = total_ds_gib < free_ram_gib * 0.8

    print(f"\n      GPU-VRAM cache feasible: {'YES' if can_gpu_cache else 'NO  (not enough free VRAM after model)'}")
    print(f"      CPU-RAM  cache feasible: {'YES' if can_ram_cache else 'NO  (not enough system RAM)'}")

    # ── 4. Benchmark ──────────────────────────────────────────────────────
    print(f"\n[4/4] Benchmarking dataloader throughput ({args.n_batches} batches each)...")
    ram_cache = None

    # Strategy A: Current H5 on-disk loader
    print("\n  Strategy A -- Current (H5 on-disk, multi-worker DataLoader):")
    loader_disk = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=tc.num_workers, pin_memory=True,
        collate_fn=scene_collate_fn, persistent_workers=True
    )
    sps_disk, ms_disk = benchmark_loader(loader_disk, device, args.n_batches, label="Disk")

    # Strategy B: CPU-RAM cache
    if can_ram_cache:
        print(f"\n  Strategy B -- CPU-RAM cache (preload {total_ds_gib:.2f} GiB into RAM):")
        t_load = time.perf_counter()
        ram_cache = preload_to_ram(dataset, label="CPU-RAM")
        print(f"    Preload time: {time.perf_counter() - t_load:.1f}s")
        cached_ds = CachedDataset(ram_cache)
        loader_ram = DataLoader(
            cached_ds, batch_size=batch_size, shuffle=True,
            num_workers=0,  # no workers needed, data already in RAM
            pin_memory=True, collate_fn=scene_collate_fn,
        )
        sps_ram, ms_ram = benchmark_loader(loader_ram, device, args.n_batches, label="CPU-RAM")
    else:
        print(f"\n  Strategy B -- CPU-RAM cache: SKIPPED (dataset {total_ds_gib:.2f} GiB > {free_ram_gib * 0.8:.2f} GiB available)")
        sps_ram, ms_ram = None, None

    # Strategy C: GPU-VRAM cache (only if feasible)
    if can_gpu_cache:
        print(f"\n  Strategy C -- GPU-VRAM cache (preload {total_ds_gib:.2f} GiB into VRAM):")
        t_load = time.perf_counter()
        if ram_cache is None:
            ram_cache = preload_to_ram(dataset, label="GPU-VRAM")
        gpu_ds = GpuCachedDataset(ram_cache, device)
        print(f"    Preload time: {time.perf_counter() - t_load:.1f}s")
        loader_gpu = DataLoader(
            gpu_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=False,  # already on GPU
            collate_fn=scene_collate_fn,
        )
        sps_gpu, ms_gpu = benchmark_loader(loader_gpu, device, args.n_batches, label="GPU-VRAM")
    else:
        print(f"\n  Strategy C -- GPU-VRAM cache: SKIPPED (dataset {total_ds_gib:.2f} GiB > {free_vram_gib * 0.8:.2f} GiB available after model)")
        sps_gpu, ms_gpu = None, None

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  RESULTS SUMMARY")
    print("="*60)
    print(f"  {'Strategy':<30} {'samples/sec':>12}  {'ms/batch':>10}  {'vs Disk':>10}")
    print(f"  {'-'*66}")
    print(f"  {'A: Disk (H5, current)':<30} {sps_disk:>12.1f}  {ms_disk:>10.1f}  {'(baseline)':>10}")
    if sps_ram:
        speedup = f"{sps_ram/sps_disk:.2f}x"
        print(f"  {'B: CPU-RAM cache':<30} {sps_ram:>12.1f}  {ms_ram:>10.1f}  {speedup:>10}")
    else:
        print(f"  {'B: CPU-RAM cache':<30} {'N/A (infeasible)':>24}")
    if sps_gpu:
        speedup = f"{sps_gpu/sps_disk:.2f}x"
        print(f"  {'C: GPU-VRAM cache':<30} {sps_gpu:>12.1f}  {ms_gpu:>10.1f}  {speedup:>10}")
    else:
        print(f"  {'C: GPU-VRAM cache':<30} {'N/A (infeasible)':>24}")

    print(f"\n  Model peak VRAM:       {peak_model_gib:.2f} GiB / {total_vram_gib:.2f} GiB total")
    print(f"  Dataset total size:    {total_ds_gib:.2f} GiB")
    print(f"  GPU free post-model:   {free_vram_gib:.2f} GiB")
    
    if args.measure_compute and avg_compute_ms > 0:
        print("\n  COMPUTE VS DATA LOADING:")
        print(f"  Avg Compute (Forward/Backward): {avg_compute_ms:.1f} ms/batch")
        print(f"  Avg Disk Loading:               {ms_disk:.1f} ms/batch")
        ratio = (ms_disk / avg_compute_ms) * 100
        print(f"  Load-to-Compute Ratio:          {ratio:.1f}% (ideally < 100%)")
        if ms_disk < avg_compute_ms:
            print("  ✅ Data loading is FASTER than compute. The GPU is not starved.")
        else:
            print("  ❌ Data loading is SLOWER than compute. The GPU is idling waiting for data.")

    print("\n  RECOMMENDATION:")
    if not can_gpu_cache and not can_ram_cache:
        print("  Neither cache strategy is feasible. Your current disk setup is optimal.")
        print("  Consider increasing num_workers or using a faster storage backend (e.g. NVMe SSD).")
    elif can_ram_cache and not can_gpu_cache:
        print("  CPU-RAM cache is feasible but GPU-VRAM is not.")
        print("  GPU-VRAM is saturated by your model + activations.")
        if sps_ram and sps_ram > sps_disk * 1.15:
            print(f"  CPU-RAM gives {sps_ram/sps_disk:.2f}x speedup -- worth using if disk I/O is your bottleneck.")
        else:
            print("  CPU-RAM speedup is marginal (<15%). Your H5 DataLoader already keeps up with GPU compute.")
            print("  Your bottleneck is compute, not data loading -- no caching needed.")
    elif can_gpu_cache:
        print("  GPU-VRAM cache is feasible!")
        if sps_gpu and sps_gpu > sps_disk * 1.15:
            print(f"  GPU-VRAM gives {sps_gpu/sps_disk:.2f}x speedup -- strongly recommended.")
        else:
            print("  GPU cache speedup is small. Your H5 DataLoader already keeps up with GPU compute.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
