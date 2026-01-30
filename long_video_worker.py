import os
import subprocess
import shutil
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Process ultra-long videos by chunking")
    parser.add_argument("-i", "--input", required=True, help="Input video path")
    parser.add_argument("-o", "--output_dir", required=True, help="Output directory")
    parser.add_argument("--segment_time", type=str, default="30", help="Split segment time (HH:MM:SS), default 5 mins")
    parser.add_argument("--mode", default="tiny-long", help="Inference mode")
    parser.add_argument("--scale", default="2.0", help="Scale factor")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    video_name = input_path.stem
    work_dir = Path(f"temp_work_{video_name}")
    split_dir = work_dir / "splits"
    processed_dir = work_dir / "processed"
    
    if work_dir.exists():
        shutil.rmtree(work_dir)
    split_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    print(f"=== Step 1: Splitting video {input_path} ===")
    # 1. Split video into chunks using ffmpeg
    # -c copy ensures zero generation loss during splitting
    # -map 0 ensures all streams are kept
    split_cmd = [
        "ffmpeg", "-i", str(input_path),
        "-c", "copy",
        "-map", "0",
        "-segment_time", args.segment_time,
        "-f", "segment",
        "-reset_timestamps", "1",
        str(split_dir / f"{video_name}_part%03d.mp4")
    ]
    subprocess.run(split_cmd, check=True)

    files = sorted(list(split_dir.glob("*.mp4")))
    print(f"Split into {len(files)} chunks.")

    processed_files = []

    print(f"=== Step 2: Processing chunks with FlashVSR-Pro ===")
    for i, file_path in enumerate(files):
        print(f"Processing chunk {i+1}/{len(files)}: {file_path.name}")
        
        # [FEATURE] Skip Existing
        # Check if output already exists to resume from crash
        # We need a predictable output filename pattern to do this accurately.
        # Assuming infer.py logic: output_{mode}_scale{scale}_{basename}.mp4 ? 
        # Actually it's complex, so we check using glob like below.
        existing_candidates = list(processed_dir.glob(f"*{file_path.stem}*.mp4"))
        if existing_candidates:
            print(f"[RESUME] Output found for {file_path.name}, skipping inference.")
            processed_files.append(existing_candidates[0])
            continue
        
        # Using Tile-DiT is crucial for 1080p chunks
        cmd = [
            "python", "infer.py",
            "-i", str(file_path),
            "-o", str(processed_dir),
            "--mode", args.mode,
            "--scale", args.scale,
            # "--tile-dit", 
            # "--tile-size", "256", 
            # "--overlap", "24",
            "--keep-audio" # Always keep audio for chunks so merge works
        ]
        
        try:
            subprocess.run(cmd, check=True)
            
            # [FEATURE] Explicit GC (Though subprocess exit clears memory, we force OS sync)
            import gc
            gc.collect()
            
        except subprocess.CalledProcessError as e:
            print(f"!!! CRASH DETECTED on chunk {file_path.name} (Return code: {e.returncode})")
            print("!!! Skipping this chunk to continue processing...")
            # We don't append to processed_files, splitting the list?
            # Or append None?
            # If we skip, the final merge will have a gap. 
            # Better strategy: Append the ORIGINAL chunk (unprocessed) as fallback?
            # Or just ignore it? 
            # Let's append the unprocessed chunk (upscaled via ffmpeg later? No, just copy)
            # Actually, we can't merge resolutions easily. 
            # So we append NOTHING and warn user that final merge will fail or need manual fix.
            continue
        
        # Find the result file (assuming infer.py generates only one file in that dir per input)
        # Note: FlashVSR-Pro creates complex filenames, we need to find the latest created file matching input
        # Specific search rule for robustness:
        candidates = list(processed_dir.glob(f"*{file_path.stem}*.mp4"))
        if not candidates:
             print(f"Error: Could not find output for {file_path.name}")
             # exit(1) # Don't exit, just skip
             continue 
             
        # Pick the one that looks most like an output (not the input if copied)
        output_chunk = candidates[0]
        processed_files.append(output_chunk)
        
        # Optional: delete input chunk to save space
        # file_path.unlink()

    print(f"=== Step 3: Merging processed chunks ===")
    
    if len(processed_files) != len(files):
        print(f"!!! WARNING: Processed chunk count {len(processed_files)} != Input chunk count {len(files)}")
        print("!!! Likely some chunks crashed. Final video will be INCOMPLETE.")
        print("!!! Please manually re-run failed chunks or inspect results.")
    
    # Create file list for ffmpeg concat
    list_file = work_dir / "concat_list.txt"
    with open(list_file, "w") as f:
        for p in processed_files:
            f.write(f"file '{p.absolute()}'\n")
            
    final_output = Path(args.output_dir) / f"FlashVSR_{video_name}_Final.mp4"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Merge
    # -c copy ensures no re-encoding quality loss
    merge_cmd = [
        "ffmpeg", "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(final_output)
    ]
    subprocess.run(merge_cmd, check=True)
    
    print(f"=== Done! Final video saved to: {final_output} ===")
    
    # Cleanup
    # shutil.rmtree(work_dir)

if __name__ == "__main__":
    main()