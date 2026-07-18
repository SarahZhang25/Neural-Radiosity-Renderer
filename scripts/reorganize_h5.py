"""
Reorganize dataset_*.h5 files by removing duplicates and slotting backfilled scenes into holes.
Treats the newest chunk as a "donor", pulling scenes out and into the holes in the older chunks.

Usage:
python scripts/reorganize_h5.py --dir /storage/sazhang/datasets/dataset_uniform --format nmr

"""
import os
import glob
import h5py
import argparse
import re
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def reorganize_h5(output_dir, fmt):
    print(f"[*] Reorganizing directory: {output_dir}")
    print(f"[*] Format: {fmt}")

    parent_dir = Path(output_dir)
    chunk_glob = f"{fmt}_dataset_chunk_*.h5"
    chunk_paths = sorted(parent_dir.glob(chunk_glob))
    
    if not chunk_paths:
        print(f"[!] No chunk files matching {chunk_glob} found.")
        return
        
    print(f"[*] Found {len(chunk_paths)} chunk files. Scanning for duplicates...")
    
    # 1. Map scene indices to the chunks they appear in
    scene_to_chunks = defaultdict(list)
    chunk_to_scenes = defaultdict(list)
    
    for chunk_path in tqdm(chunk_paths, desc="Scanning chunks"):
        try:
            with h5py.File(chunk_path, "r") as handle:
                for scene_name in handle.keys():
                    match = re.search(r"scene_(\d+)$", scene_name)
                    if match:
                        scene_idx = int(match.group(1))
                        scene_to_chunks[scene_idx].append(chunk_path)
                        chunk_to_scenes[chunk_path].append(scene_idx)
        except Exception as e:
            print(f"[!] Error reading {chunk_path}: {e}")
            
    # 2. Identify and remove duplicates
    duplicates = {idx: paths for idx, paths in scene_to_chunks.items() if len(paths) > 1}
    
    if duplicates:
        print(f"\n[*] Found {len(duplicates)} duplicated scenes. Deleting redundant copies...")
        for idx, paths in tqdm(duplicates.items(), desc="Removing duplicates"):
            # Keep the first path (earliest chunk), delete from the rest
            keep_path = paths[0]
            delete_paths = paths[1:]
            
            for del_path in delete_paths:
                scene_name = f"scene_{idx:06d}"
                try:
                    with h5py.File(del_path, "a") as handle:
                        if scene_name in handle:
                            del handle[scene_name]
                    # Update our in-memory tracking
                    chunk_to_scenes[del_path].remove(idx)
                except Exception as e:
                    print(f"[!] Failed to delete {scene_name} from {del_path.name}: {e}")
    else:
        print("\n[*] No duplicates found to remove.")

    # 3. Identify holes and backfill from the last chunk
    # Infer target sizes: if a chunk has > 500 scenes (after dup removal), it was meant to be 1000. Else 500.
    # The last chunk is considered the "donor" chunk and doesn't have a target size we need to fill.
    
    donor_chunk = chunk_paths[-1]
    holes = [] # list of tuples: (target_chunk_path, number_of_holes)
    
    for chunk_path in chunk_paths[:-1]:
        current_size = len(chunk_to_scenes[chunk_path])
        if current_size == 0:
            continue
            
        target_size = 1000 if current_size > 500 else 500
        if current_size < target_size:
            holes.append((chunk_path, target_size - current_size))
            
    total_holes = sum(count for _, count in holes)
    print(f"\n[*] Found {total_holes} holes across earlier chunks.")
    
    if total_holes > 0:
        print(f"[*] Moving scenes from donor chunk ({donor_chunk.name}) to fill holes...")
        
        try:
            with h5py.File(donor_chunk, "a") as donor_handle:
                donor_scenes = sorted([s for s in donor_handle.keys() if s.startswith("scene_")])
                
                if len(donor_scenes) < total_holes:
                    print(f"[!] Warning: Donor chunk only has {len(donor_scenes)} scenes, but we need {total_holes} to fill all holes!")
                    print("[!] You may need to run the generation pipeline with --resume first.")
                    
                for target_path, hole_count in holes:
                    if not donor_scenes:
                        break
                        
                    with h5py.File(target_path, "a") as target_handle:
                        for _ in range(hole_count):
                            if not donor_scenes:
                                break
                            
                            scene_to_move = donor_scenes.pop(0)
                            
                            # Copy from donor to target
                            donor_handle.copy(scene_to_move, target_handle, name=scene_to_move)
                            # Delete from donor
                            del donor_handle[scene_to_move]
                            
                            print(f"    Moved {scene_to_move} -> {target_path.name}")
                            
        except Exception as e:
            print(f"[!] Error during scene transfer: {e}")
            
        # Check if donor chunk is now empty and can be deleted
        try:
            with h5py.File(donor_chunk, "r") as donor_handle:
                remaining = len([s for s in donor_handle.keys() if s.startswith("scene_")])
                
            if remaining == 0:
                print(f"[*] Donor chunk {donor_chunk.name} is now empty. Deleting it...")
                os.remove(donor_chunk)
        except Exception as e:
             pass
    else:
        print("\n[*] No holes to fill.")
        
    print("\n[*] Reorganization complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorganize H5 chunks by removing duplicates and slotting backfilled scenes into holes.")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing H5 chunks")
    parser.add_argument("--format", type=str, default="nmr", help="Format to reorganize (e.g., nmr or rf)")
    
    args = parser.parse_args()
    reorganize_h5(args.dir, args.format)
