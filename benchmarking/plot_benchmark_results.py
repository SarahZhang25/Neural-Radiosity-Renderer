import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot benchmarking results")
    parser.add_argument("--csv_file", type=str, default="efficiency_benchmark_results.csv", help="Path to the benchmarking CSV file")
    parser.add_argument("--out_dir", type=str, default=".", help="Directory to save the output plots")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"Error: Could not find CSV file at {args.csv_file}")
        return
        
    df = pd.read_csv(args.csv_file)
    
    # Replace 'OOM' with NaN for plotting purposes
    df.replace('OOM', pd.NA, inplace=True)
    
    # Convert numerical columns
    cols_to_convert = ['my_model_time_ms', 'my_model_mem_mb', 'renderformer_time_ms', 'renderformer_mem_mb']
    for col in cols_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Plot 1: Inference Time
    plt.figure(figsize=(10, 6))
    if not df['my_model_time_ms'].isna().all():
        plt.plot(df['num_objects'], df['my_model_time_ms'], marker='o', label='My Model', color='blue')
    if not df['renderformer_time_ms'].isna().all():
        plt.plot(df['num_objects'], df['renderformer_time_ms'], marker='s', label='RenderFormer', color='red')
        
    plt.title('Inference Time vs Scene Complexity')
    plt.xlabel('Number of Objects')
    plt.ylabel('Time (ms)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    time_plot_path = os.path.join(args.out_dir, "inference_time_plot.png")
    plt.savefig(time_plot_path)
    print(f"Saved {time_plot_path}")
    
    # Plot 2: Memory Cost
    plt.figure(figsize=(10, 6))
    if not df['my_model_mem_mb'].isna().all():
        plt.plot(df['num_objects'], df['my_model_mem_mb'], marker='o', label='My Model', color='blue')
    if not df['renderformer_mem_mb'].isna().all():
        plt.plot(df['num_objects'], df['renderformer_mem_mb'], marker='s', label='RenderFormer', color='red')
        
    plt.title('Peak VRAM vs Scene Complexity')
    plt.xlabel('Number of Objects')
    plt.ylabel('Peak VRAM (MB)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    mem_plot_path = os.path.join(args.out_dir, "memory_cost_plot.png")
    plt.savefig(mem_plot_path)
    print(f"Saved {mem_plot_path}")

if __name__ == "__main__":
    main()
