# Efficiency Benchmarking

This directory contains tools to benchmark the inference time and peak VRAM usage of Renderformer versus the Neural-Radiosity-Renderer model across varying scene complexities. 

The benchmark uses dynamically generated dummy tensors to purely measure computational execution cost, isolating the metrics from disk I/O.

## Scripts

1. `run_efficiency_benchmark.py`: Simulates scene complexities from 1 up to 500 objects, dynamically generating corresponding dense tensor inputs. Measures inference time and tracks memory overhead. Results are exported to a CSV.
2. `plot_benchmark_results.py`: Generates visual comparisons of computational cost (`time (ms)`) and spatial complexity (`Peak VRAM (MB)`) across both models using matplotlib.

## Usage

Run the benchmark (creates `efficiency_benchmark_results.csv`):

```bash
python benchmarking/run_efficiency_benchmark.py
```

Plot the results (creates `inference_time_plot.png` and `memory_cost_plot.png`):

```bash
python benchmarking/plot_benchmark_results.py --csv_file benchmarking/efficiency_benchmark_results.csv --out_dir benchmarking/
```
