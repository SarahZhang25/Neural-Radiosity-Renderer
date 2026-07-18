"""
Dataset Caching & Batch Size Benchmark
=======================================
Evaluates whether caching the training dataset in GPU VRAM or CPU RAM
provides a meaningful speedup over the current H5-from-disk loading.
Also 


What this script does:
    - Instantiates the model + runs a few forward/backward passes to measure
     peak GPU memory consumed by the model.
    - Sweeps over candidate batch sizes to find the largest batch size
that fits in GPU memory without OOM.
    - Scans the training dataset to estimate total tensor size.
    - Measures wall-clock throughput (samples/sec) for three strategies:
       A. Current:  H5 on-disk with DataLoader workers (your existing setup)
       B. CPU-RAM:  Preload all tensors into pinned CPU RAM, no disk I/O during training
       C. GPU-VRAM: Preload all tensors into GPU memory (if enough free VRAM exists)
    - Prints a summary table and a recommendation.

Usage (from project root):
    # Run all experiments (default)
    CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_dataset_cache.py \
        --config training/train_config_46M_litePT.yaml

    # Only sweep batch sizes (fast, skips everything else)
    CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_dataset_cache.py \
        --config training/train_config_46M_litePT.yaml \
        --no-vram_probe --no-dataset_estimate --no-cache_benchmark

    # Only run caching strategies benchmark (skip batch sweep)
    CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_dataset_cache.py \
        --config training/train_config_46M_litePT.yaml \
        --no-batch_sweep

    # Measure compute vs data loading bottleneck
    CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_dataset_cache.py \
        --config training/train_config_46M_litePT.yaml \
        --measure_compute

Experiments (all enabled by default, disable individually with --no-<name>):
    --batch_sweep       Sweep candidate batch sizes measuring peak VRAM, compute
                        time, and throughput. OOM is caught gracefully per size.
                        Use --batch_sizes to override candidates (default: 16 32 64 128).
    --vram_probe        Probe peak VRAM at the batch_size set in your config.
                        Reuses the sweep result if the config batch_size was
                        already covered by --batch_sweep.
    --dataset_estimate  Estimate total dataset tensor footprint by probing 20
                        samples and extrapolating. Determines whether GPU-VRAM
                        or CPU-RAM caching is feasible for your dataset size.
    --cache_benchmark   Benchmark three dataloader strategies and report
                        throughput (samples/sec) and ms/batch:
                          A. H5 on-disk  - current setup with DataLoader workers
                          B. CPU-RAM     - preload all data into pinned system RAM
                          C. GPU-VRAM    - preload all data directly into GPU memory
                        Requires --dataset_estimate to assess feasibility.

Other flags:
    --config PATH           Path to YAML training config (default: train_config_46M.yaml)
    --n_batches N           Batches to time per caching strategy (default: 30)
    --model_iters N         Forward/backward passes per batch-size sweep point (default: 3)
    --measure_compute       Also measure fw+bw compute time and compare to data loading time
    --batch_sizes N [N...]  Override candidate batch sizes for the sweep
"""

import os
import sys
import time
import argparse
import psutil
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Make sure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import NeuralRadiosityConfig
from model.global_illumination_model import GlobalIlluminationModel
# from training.dataset import NPZSceneDataset as SceneDataset, scene_collate_fn
from training.dataset import (
    H5SceneDataset as SceneDataset, 
    scene_collate_fn, 
    preload_to_ram, 
    CachedDataset, 
    GpuCachedDataset
)
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


def run_forward_backward(model, ray_generator, dataset, batch_size, image_res, num_workers, device,
                          n_iters=3, measure_compute=False):
    """
    Run n_iters forward+backward passes at a given batch size.
    Returns (peak_vram_gib, avg_compute_ms, samples_per_sec) or raises torch.cuda.OutOfMemoryError.
    """
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    model.train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=scene_collate_fn, num_workers=num_workers)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.L1Loss()

    compute_times = []
    total_samples = 0

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
            elapsed = t1 - t0
            if i > 0:  # skip first iter warmup
                compute_times.append(elapsed * 1000.0)
                total_samples += batch_size

    peak_bytes = torch.cuda.max_memory_allocated(device)
    avg_compute_ms = sum(compute_times) / len(compute_times) if compute_times else 0.0
    sps = (total_samples / (sum(compute_times) / 1000.0)) if compute_times else 0.0
    return bytes_to_gib(peak_bytes), avg_compute_ms, sps


def sweep_batch_sizes(model_config, ray_gen_class, dataset, image_res, num_workers, device,
                       candidate_batch_sizes, n_iters=3):
    """
    Sweep over candidate batch sizes, measuring peak VRAM and throughput.
    OOM is caught gracefully. Returns a list of result dicts.
    """
    results = []
    total_vram_gib = bytes_to_gib(torch.cuda.get_device_properties(device).total_memory)

    for bs in candidate_batch_sizes:
        print(f"    batch_size={bs:>4} ...", end="", flush=True)
        # Rebuild a fresh model for each sweep to avoid gradient state bleeding across runs
        model = GlobalIlluminationModel(model_config).to(device)
        ray_gen = ray_gen_class().to(device)
        try:
            peak_gib, avg_ms, sps = run_forward_backward(
                model, ray_gen, dataset, bs, image_res, num_workers, device,
                n_iters=n_iters, measure_compute=True
            )
            utilization_pct = (peak_gib / total_vram_gib) * 100
            print(f"  peak={peak_gib:.2f} GiB ({utilization_pct:.0f}%),  "
                  f"compute={avg_ms:.0f} ms/batch,  {sps:.1f} samples/sec  ✅")
            results.append({
                'batch_size': bs,
                'peak_gib': peak_gib,
                'utilization_pct': utilization_pct,
                'avg_compute_ms': avg_ms,
                'samples_per_sec': sps,
                'oom': False,
            })
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM  ❌")
            results.append({
                'batch_size': bs,
                'peak_gib': None,
                'utilization_pct': None,
                'avg_compute_ms': None,
                'samples_per_sec': None,
                'oom': True,
            })
        finally:
            del model, ray_gen
            torch.cuda.empty_cache()

    return results


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


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='training/train_config_46M.yaml')
    parser.add_argument('--n_batches', type=int, default=30, help="Batches to time per strategy")
    parser.add_argument('--model_iters', type=int, default=3, help="Forward/backward passes per batch-size sweep point")
    parser.add_argument('--measure_compute', action='store_true', help="Measure fw/bw compute time and compare with loading time")
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=None,
                        help="Candidate batch sizes to sweep. Default: [16, 32, 64, 128]")
    # Per-experiment enable flags — all on by default, pass --no-X to disable
    parser.add_argument('--batch_sweep', default=True, action=argparse.BooleanOptionalAction,
                        help="Sweep candidate batch sizes and measure VRAM/throughput")
    parser.add_argument('--vram_probe', default=True, action=argparse.BooleanOptionalAction,
                        help="Probe peak VRAM at the config batch_size")
    parser.add_argument('--dataset_estimate', default=True, action=argparse.BooleanOptionalAction,
                        help="Estimate total dataset tensor footprint")
    parser.add_argument('--cache_benchmark', default=True, action=argparse.BooleanOptionalAction,
                        help="Benchmark disk / CPU-RAM / GPU-VRAM caching strategies")
    args = parser.parse_args()

    device = torch.device('cuda:0')
    config = NeuralRadiosityConfig.from_yaml(args.config)
    tc = config.training
    batch_size = tc.batch_size

    candidate_batch_sizes = args.batch_sizes or [16, 32, 64, 128]
    total_vram_gib = bytes_to_gib(torch.cuda.get_device_properties(device).total_memory)

    print("\n" + "="*70)
    print("  DATASET CACHE & BATCH SIZE FEASIBILITY BENCHMARK")
    print("="*70)

    # ── 1. Build dataset ──────────────────────────────────────────────────
    print(f"\n[1/5] Building dataset from: {tc.data_dir}")
    dataset = SceneDataset(
        data_dir=tc.data_dir,
        image_res=tc.image_res,
        split='train',
        max_dataset_size=None,
        shuffle=True, shuffle_seed=42
    )
    print(f"      Total train samples: {len(dataset)}")

    sweep_results = []
    best_valid = None

    # ── Batch size sweep ─────────────────────────────────────────────────
    if args.batch_sweep:
        print(f"\n[batch_sweep] Sweeping batch sizes {candidate_batch_sizes} ({args.model_iters} iters each)...")
        print(f"      GPU: {torch.cuda.get_device_name(device)}  |  Total VRAM: {total_vram_gib:.1f} GiB\n")
        sweep_results = sweep_batch_sizes(
            config, RayGenerator, dataset, tc.image_res, tc.num_workers, device,
            candidate_batch_sizes, n_iters=args.model_iters
        )

        print(f"\n  {'Batch':>6}  {'Peak VRAM':>10}  {'VRAM %':>7}  {'ms/batch':>10}  {'samples/sec':>13}  {'Status':>8}")
        print(f"  {'-'*62}")
        for r in sweep_results:
            if r['oom']:
                print(f"  {r['batch_size']:>6}  {'N/A':>10}  {'N/A':>7}  {'N/A':>10}  {'N/A':>13}  {'OOM ❌':>8}")
            else:
                print(f"  {r['batch_size']:>6}  {r['peak_gib']:>9.2f}G  {r['utilization_pct']:>6.0f}%"
                      f"  {r['avg_compute_ms']:>10.0f}  {r['samples_per_sec']:>13.1f}  {'✅':>8}")
                best_valid = r

        if best_valid:
            print(f"\n  Recommended batch_size: {best_valid['batch_size']}  "
                  f"(largest that fits, {best_valid['peak_gib']:.2f} GiB / {total_vram_gib:.1f} GiB = "
                  f"{best_valid['utilization_pct']:.0f}% VRAM)")
            print(f"  Config currently uses:  batch_size={batch_size}")
            if batch_size != best_valid['batch_size']:
                cmp = "above" if batch_size > best_valid['batch_size'] else "below"
                print(f"  ⚠️  Your config batch_size ({batch_size}) is {cmp} the recommended value.")
        else:
            print("\n  ❌ All batch sizes OOM'd. Check model size and GPU memory.")

    # ── VRAM probe at config batch_size ──────────────────────────────────
    peak_model_gib = 0.0
    avg_compute_ms = 0.0
    free_vram_gib, total_vram_gib = get_free_gpu_gib(device)
    free_ram_gib, total_ram_gib = get_free_ram_gib()

    if args.vram_probe:
        config_sweep = next((r for r in sweep_results if r['batch_size'] == batch_size and not r['oom']), None)
        print(f"\n[vram_probe] Model peak VRAM at config batch_size={batch_size}...")
        if config_sweep:
            peak_model_gib = config_sweep['peak_gib']
            avg_compute_ms = config_sweep['avg_compute_ms']
            print(f"      (Reusing result from batch_sweep)")
        else:
            print(f"      Running probe ({args.model_iters} iters)...")
            model = GlobalIlluminationModel(config).to(device)
            ray_gen = RayGenerator().to(device)
            try:
                peak_model_gib, avg_compute_ms, _ = run_forward_backward(
                    model, ray_gen, dataset, batch_size, tc.image_res, tc.num_workers, device,
                    n_iters=args.model_iters, measure_compute=True
                )
            except torch.cuda.OutOfMemoryError:
                print(f"      ❌ OOM at batch_size={batch_size}. Set a valid batch size in config.")
            finally:
                del model, ray_gen
                torch.cuda.empty_cache()

        # Get baseline empty VRAM (after model is deleted from the probe)
        free_empty_vram_gib, total_vram_gib = get_free_gpu_gib(device)
        free_ram_gib, total_ram_gib = get_free_ram_gib()
        print(f"      Peak model VRAM:       {peak_model_gib:.2f} GiB")
        print(f"      Base CUDA overhead:    {total_vram_gib - free_empty_vram_gib:.2f} GiB")
        print(f"      System RAM free:       {free_ram_gib:.2f} / {total_ram_gib:.2f} GiB")

    # ── Dataset size estimate ─────────────────────────────────────────────
    total_ds_gib, per_sample_gib = 0.0, 0.0
    can_gpu_cache = False
    can_ram_cache = False

    if args.dataset_estimate:
        print(f"\n[dataset_estimate] Estimating dataset tensor footprint (probing 20 samples)...")
        total_ds_gib, per_sample_gib = estimate_dataset_gib(dataset, n_probe=20)
        print(f"      Per-sample:            {per_sample_gib * 1024:.1f} MiB")
        print(f"      Total dataset (~{len(dataset)} samples): {total_ds_gib:.2f} GiB")
        
        # To cache in GPU, we need room for the dataset AND the model's peak training spike
        # plus the base CUDA context overhead.
        projected_peak_vram = total_ds_gib + peak_model_gib + (total_vram_gib - free_empty_vram_gib)
        can_gpu_cache = projected_peak_vram < (total_vram_gib * 0.95)
        can_ram_cache = total_ds_gib < free_ram_gib * 0.8
        
        print(f"\n      GPU-VRAM cache feasible: {'YES' if can_gpu_cache else 'NO  (Dataset + Peak Model would exceed VRAM)'}")
        print(f"      CPU-RAM  cache feasible: {'YES' if can_ram_cache else 'NO  (not enough system RAM)'}")

    # ── Caching strategy benchmark ────────────────────────────────────────
    sps_disk, ms_disk = None, None
    sps_ram, ms_ram = None, None
    sps_gpu, ms_gpu = None, None

    if args.cache_benchmark:
        if not args.dataset_estimate:
            print("\n  ⚠️  --cache_benchmark requires --dataset_estimate to assess feasibility. Skipping.")
        else:
            print(f"\n[cache_benchmark] Benchmarking dataloader throughput ({args.n_batches} batches each)...")
            ram_cache = None

            print("\n  Strategy A -- Current (H5 on-disk, multi-worker DataLoader):")
            loader_disk = DataLoader(
                dataset, batch_size=batch_size, shuffle=True,
                num_workers=tc.num_workers, pin_memory=True,
                collate_fn=scene_collate_fn, persistent_workers=True
            )
            sps_disk, ms_disk = benchmark_loader(loader_disk, device, args.n_batches, label="Disk")

            if can_ram_cache:
                print(f"\n  Strategy B -- CPU-RAM cache (preload {total_ds_gib:.2f} GiB into RAM):")
                t_load = time.perf_counter()
                ram_cache = preload_to_ram(dataset, label="CPU-RAM")
                print(f"    Preload time: {time.perf_counter() - t_load:.1f}s")
                loader_ram = DataLoader(
                    CachedDataset(ram_cache), batch_size=batch_size, shuffle=True,
                    num_workers=0, pin_memory=True, collate_fn=scene_collate_fn,
                )
                sps_ram, ms_ram = benchmark_loader(loader_ram, device, args.n_batches, label="CPU-RAM")
            else:
                print(f"\n  Strategy B -- CPU-RAM cache: SKIPPED (dataset {total_ds_gib:.2f} GiB > {free_ram_gib * 0.8:.2f} GiB available)")

            if can_gpu_cache:
                print(f"\n  Strategy C -- GPU-VRAM cache (preload {total_ds_gib:.2f} GiB into VRAM):")
                t_load = time.perf_counter()
                if ram_cache is None:
                    ram_cache = preload_to_ram(dataset, label="GPU-VRAM")
                gpu_ds = GpuCachedDataset(ram_cache, device)
                print(f"    Preload time: {time.perf_counter() - t_load:.1f}s")
                loader_gpu = DataLoader(
                    gpu_ds, batch_size=batch_size, shuffle=True,
                    num_workers=0, pin_memory=False, collate_fn=scene_collate_fn,
                )
                sps_gpu, ms_gpu = benchmark_loader(loader_gpu, device, args.n_batches, label="GPU-VRAM")
            else:
                print(f"\n  Strategy C -- GPU-VRAM cache: SKIPPED (Dataset {total_ds_gib:.2f}G + Model Peak {peak_model_gib:.2f}G > Total VRAM)")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  RESULTS SUMMARY")
    print("="*70)

    if args.batch_sweep and sweep_results:
        print(f"\n  BATCH SIZE SWEEP  (GPU: {torch.cuda.get_device_name(device)}, {total_vram_gib:.1f} GiB total)")
        print(f"  {'Batch':>6}  {'Peak VRAM':>10}  {'VRAM %':>7}  {'ms/batch':>10}  {'samples/sec':>13}  Status")
        print(f"  {'-'*64}")
        for r in sweep_results:
            if r['oom']:
                print(f"  {r['batch_size']:>6}  {'OOM':>10}  {'---':>7}  {'---':>10}  {'---':>13}  ❌")
            else:
                marker = " ◀ recommended" if r == best_valid else ""
                print(f"  {r['batch_size']:>6}  {r['peak_gib']:>9.2f}G  {r['utilization_pct']:>6.0f}%"
                      f"  {r['avg_compute_ms']:>10.0f}  {r['samples_per_sec']:>13.1f}  ✅{marker}")

    if args.cache_benchmark and sps_disk is not None:
        print(f"\n  CACHING STRATEGIES  (batch_size={batch_size})")
        print(f"  {'Strategy':<30} {'samples/sec':>12}  {'ms/batch':>10}  {'vs Disk':>10}")
        print(f"  {'-'*66}")
        print(f"  {'A: Disk (H5, current)':<30} {sps_disk:>12.1f}  {ms_disk:>10.1f}  {'(baseline)':>10}")
        if sps_ram:
            speedup = f"{sps_ram/sps_disk:.2f}x"
            print(f"  {'B: CPU-RAM cache':<30} {sps_ram:>12.1f}  {ms_ram:>10.1f}  {speedup:>10}")
        else:
            print(f"  {'B: CPU-RAM cache':<30} {'N/A (infeasible)':<24}")
        if sps_gpu:
            speedup = f"{sps_gpu/sps_disk:.2f}x"
            print(f"  {'C: GPU-VRAM cache':<30} {sps_gpu:>12.1f}  {ms_gpu:>10.1f}  {speedup:>10}")
        else:
            print(f"  {'C: GPU-VRAM cache':<30} {'N/A (infeasible)':<24}")

    print(f"\n  Model peak VRAM:       {peak_model_gib:.2f} GiB / {total_vram_gib:.2f} GiB total")
    print(f"  Dataset total size:    {total_ds_gib:.2f} GiB")
    
    if args.dataset_estimate and args.vram_probe:
        projected = total_ds_gib + peak_model_gib + (total_vram_gib - free_empty_vram_gib)
        print(f"  Projected Peak VRAM:   {projected:.2f} GiB (if GPU-cached)")

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
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
