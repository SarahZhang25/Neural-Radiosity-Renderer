# Efficiency Benchmarking

This directory contains tools to benchmark the inference time and peak VRAM usage of Renderformer versus the Neural-Radiosity-Renderer model across varying scene complexities. 

The benchmark uses dynamically generated dummy tensors to purely measure computational execution cost, isolating the metrics from disk I/O.

## Scripts

1. `run_efficiency_benchmark.py`: Simulates scene complexities from 1 up to 500 objects, dynamically generating corresponding dense tensor inputs. Measures inference time and tracks memory overhead. Results are exported to a CSV, and visual comparison plots (`inference_time_plot.png` and `memory_cost_plot.png`) are automatically generated using matplotlib.

## Usage

Run the benchmark (creates `efficiency_benchmark_results.csv` and auto-generates the plots):

```bash
python benchmarking/run_efficiency_benchmark.py
```
