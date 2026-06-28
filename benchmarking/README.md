# Efficiency Benchmarking

This directory contains tools to benchmark the inference time and peak VRAM usage of Renderformer versus the Neural-Radiosity-Renderer model across varying scene complexities. 

The benchmark uses dynamically generated dummy tensors to purely measure computational execution cost, isolating the metrics from disk I/O.

## Scripts

`run_efficiency_benchmark.py`: Simulates scene complexities from 1 up to 500 objects, dynamically generating corresponding dense tensor inputs. Measures inference time and tracks memory overhead. Results are exported to a CSV, and visual comparison plots (`inference_time_plot.png` and `memory_cost_plot.png`) are automatically generated using matplotlib.

## Usage

Run the benchmark (creates `efficiency_benchmark_results.csv` and auto-generates the plots):

```bash
python benchmarking/run_efficiency_benchmark.py
```

Notes: 
Existing plots and results produced with the following trained models:
```
    --nmr_pkg_path ./training/logs/rf_ds2_chairs1-3_cbox0-3/20260625-155637_full_ds_params46M_res128x128/checkpoints/model_package_epoch_20000.pt \
    --renderformer_checkpoint_path ./renderformer/training/logs/ds2_chairs1-3_cbox0-3/20260623-154203_full_ds_params46M_res128x128/checkpoints/model_epoch_20000.pt \ 
```