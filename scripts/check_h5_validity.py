"""
Description:
* Scans a specified directory for all dataset_chunk_*.h5 files based on the format you provide (e.g., --formats nmr).
* Opens each chunk and reads the hdr_target_image and c2w matrix for every scene.
* Validates that the HDR images have the correct dimensions (e.g. exactly 4 views) and that they do not contain any 
    corrupted NaN or Inf values.
* Explicitly prints out any invalid scenes and the chunk they belong to so you can delete them.
* Analyzes the full range of contiguous valid indices and outputs exactly which indices are missing.

Usage Example:
python scripts/check_h5_validity.py --dir /storage/sazhang/datasets/dataset_uniform --formats nmr --num_views 4
"""
import os
import glob
import h5py
import numpy as np
import argparse
import re
from pathlib import Path
from tqdm import tqdm
import concurrent.futures

def check_chunk(chunk_path, num_views, exhaustive):
    scene_indices = []
    invalid_scenes = []
    chunk_name = chunk_path.name
    
    try:
        with h5py.File(chunk_path, "r") as handle:
            chunk_scene_indices = []
            for scene_name in handle.keys():
                match = re.search(r"scene_(\d+)$", scene_name)
                if match is None:
                    continue
                scene_idx = int(match.group(1))
                chunk_scene_indices.append(scene_idx)
                
                # Validate contents
                is_valid = True
                issues = []
                if 'hdr_target_image' not in handle[scene_name]:
                    is_valid = False
                    issues.append("Missing hdr_target_image")
                else:
                    dataset = handle[scene_name]['hdr_target_image']
                    if dataset.shape[0] != num_views:
                        is_valid = False
                        issues.append(f"Expected {num_views} views, got {dataset.shape[0]}")
                    
                    if exhaustive:
                        # Only load the array into RAM and decompress if exhaustive checking is enabled
                        hdr_data = dataset[:]
                        if np.isnan(hdr_data).any():
                            is_valid = False
                            issues.append("Contains NaN values")
                            
                        if np.isinf(hdr_data).any():
                            is_valid = False
                            issues.append("Contains Inf values")
                        
                if 'c2w' not in handle[scene_name]:
                    is_valid = False
                    issues.append("Missing c2w")
                    
                if is_valid:
                    scene_indices.append(scene_idx)
                else:
                    invalid_scenes.append((scene_name, chunk_name, issues))
                    
            if chunk_scene_indices:
                chunk_scene_indices = sorted(chunk_scene_indices)
                msg = f"[*] {chunk_name}: {len(chunk_scene_indices)} scenes (range: {chunk_scene_indices[0]} to {chunk_scene_indices[-1]})"
            else:
                msg = f"[*] {chunk_name}: 0 scenes"
                
            return (scene_indices, invalid_scenes, msg)
            
    except Exception as e:
        msg = f"[*] Error reading {chunk_path}: {e}"
        return (scene_indices, invalid_scenes, msg)

def check_h5_validity(output_dir, formats, num_views, exhaustive):
    print(f"[*] Checking directory: {output_dir}")
    print(f"[*] Formats to check: {formats}")
    print(f"[*] Expected number of views: {num_views}")
    print(f"[*] Exhaustive pixel checking: {exhaustive}")

    parent_dir = Path(output_dir)
    
    for fmt in formats:
        print(f"\n{'='*50}\n[*] Checking format: {fmt}\n{'='*50}")
        chunk_glob = f"{fmt}_dataset_chunk_*.h5"
        chunk_paths = sorted(parent_dir.glob(chunk_glob))
        
        if not chunk_paths:
            print(f"No chunk files matching {chunk_glob} found.")
            continue
            
        scene_indices = []
        invalid_scenes = []
        
        print(f"[*] Found {len(chunk_paths)} chunk files. Scanning contents...")
        
        workers = 1 #min(16, os.cpu_count() or 4)
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(check_chunk, path, num_views, exhaustive) for path in chunk_paths]
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Checking chunks"):
                chunk_scenes, chunk_invalid, msg = future.result()
                scene_indices.extend(chunk_scenes)
                invalid_scenes.extend(chunk_invalid)
                tqdm.write(msg)
                
        if invalid_scenes:
            print(f"\n[!] Found {len(invalid_scenes)} invalid scenes:")
            for scene_name, chunk_name, issues in invalid_scenes:
                print(f"  - {scene_name} (in {chunk_name}): {', '.join(issues)}")
        else:
            print("\n[*] All scenes contain valid HDR images and C2W matrices.")

        if not scene_indices:
            print(f"[*] No valid scenes found for format {fmt}.")
            continue
            
        from collections import Counter
        counts = Counter(scene_indices)
        duplicates = {idx: count for idx, count in counts.items() if count > 1}
        
        if duplicates:
            print(f"\n[!] Found {len(duplicates)} duplicated scene indices! (They appear in multiple chunks)")
            dup_list = list(duplicates.items())
            if len(dup_list) > 20:
                print(f"[*] Showing first 20 duplicates: {dup_list[:20]}")
            else:
                print(f"[*] Duplicates (index, count): {dup_list}")
        else:
            print("\n[*] No duplicate scene indices found.")
            
        scene_indices = sorted(set(scene_indices))
        min_scene = scene_indices[0]
        max_scene = scene_indices[-1]
        missing = [scene_idx for scene_idx in range(min_scene, max_scene + 1) if scene_idx not in scene_indices]
        
        print(f"\n[*] Scene index range: {min_scene} to {max_scene}")
        if missing:
            print(f"[*] Missing valid scenes ({len(missing)}): {missing}")
        else:
            print("[*] Missing valid scenes in range: none")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check validity of generated H5 scenes")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing H5 chunks")
    parser.add_argument("--formats", nargs='+', choices=['nmr', 'rf'], default=['nmr', 'rf'], help="Formats to check")
    parser.add_argument("--num_views", type=int, default=4, help="Expected number of views per scene")
    parser.add_argument("--exhaustive", action="store_true", help="Perform slow pixel-level checking for NaNs/Infs on all HDR images")
    
    args = parser.parse_args()
    check_h5_validity(args.dir, args.formats, args.num_views, args.exhaustive)
