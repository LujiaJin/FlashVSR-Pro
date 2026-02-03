import os
import subprocess
import shutil
import argparse
import gc
import re
from pathlib import Path

def detect_scene_changes(video_path, threshold=0.4):
    """
    Detect scene change timestamps using ffmpeg scene detection.
    Returns a list of timestamps in seconds.
    """
    print(f"Detecting scene changes (threshold={threshold})...")
    
    # Use ffmpeg to detect scenes and output to stderr
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-f", "null", "-"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
    
    # Parse timestamps from showinfo output
    timestamps = []
    # Pattern: pts_time:12.345
    pattern = r'pts_time:(\d+\.?\d*)'
    
    for match in re.finditer(pattern, result.stderr):
        try:
            ts = float(match.group(1))
            timestamps.append(ts)
        except ValueError:
            continue
    
    # Always include 0.0 as the first timestamp
    if not timestamps or timestamps[0] != 0.0:
        timestamps.insert(0, 0.0)
    
    return sorted(set(timestamps))

def split_timestamps_with_max_duration(timestamps, max_duration):
    """
    Split timestamps ensuring no segment exceeds max_duration.
    If two consecutive scene changes are >max_duration apart, insert intermediate timestamps.
    """
    result = [timestamps[0]]  # Start with the first timestamp
    
    for i in range(1, len(timestamps)):
        prev_ts = result[-1]
        curr_ts = timestamps[i]
        duration = curr_ts - prev_ts
        
        # If duration exceeds max, insert intermediate timestamps
        if duration > max_duration:
            num_splits = int(duration // max_duration)
            for j in range(1, num_splits + 1):
                intermediate = prev_ts + j * max_duration
                result.append(intermediate)
        
        result.append(curr_ts)
    
    return result

def split_video_at_timestamps(input_path, output_dir, timestamps, video_name):
    """
    Split video at specific timestamps using ffmpeg.
    """
    output_files = []
    
    for i in range(len(timestamps) - 1):
        start_time = timestamps[i]
        end_time = timestamps[i + 1]
        output_file = output_dir / f"{video_name}_part_{i:04d}.mp4"
        
        # Use -ss and -to for precise cutting with -c copy for speed
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-to", str(end_time),
            "-i", str(input_path),
            "-c", "copy",
            "-map", "0",
            "-avoid_negative_ts", "make_zero",
            str(output_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error splitting segment {i}: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
        output_files.append(output_file)
    
    # Handle the last segment (from last timestamp to end)
    if timestamps:
        last_start = timestamps[-1]
        output_file = output_dir / f"{video_name}_part_{len(timestamps)-1:04d}.mp4"
        
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(last_start),
            "-i", str(input_path),
            "-c", "copy",
            "-map", "0",
            str(output_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error splitting last segment: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
        output_files.append(output_file)
    
    return output_files

def main():
    parser = argparse.ArgumentParser(description="Process ultra-long videos by scene-aware chunking with OOM retry")
    parser.add_argument("-i", "--input", required=True, help="Input video path")
    parser.add_argument("-o", "--output_dir", required=True, help="Output directory")
    parser.add_argument("--max_segment_time", type=int, default=30, help="Maximum segment time in seconds")
    parser.add_argument("--mode", default="tiny-long", help="Inference mode")
    parser.add_argument("--scale", default="2.0", help="Scale factor")
    
    args = parser.parse_args()
    
    # Normalize path (handle backslashes on Windows/cross-platform)
    input_path = Path(args.input).resolve()
    
    # Verify input file exists
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return
    
    video_name = input_path.stem
    print(f"Input video: {input_path}")
    print(f"Video name: {video_name}")
    
    work_dir = Path(args.output_dir) / "MyOwnSwordsman"
    split_dir = work_dir / "01_splits_original"
    processed_dir = work_dir / "02_splits_processed"
    failed_log = work_dir / "failed_segments.txt"
    
    # Create directories
    work_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Step 1: Intelligent scene-based splitting (max {args.max_segment_time}s per segment) ===")
    
    # Step 1.1: Detect scene changes
    scene_timestamps = detect_scene_changes(input_path, threshold=0.4)
    print(f"Detected {len(scene_timestamps)} scene changes.")
    
    # Step 1.2: Apply max duration constraint
    split_timestamps = split_timestamps_with_max_duration(scene_timestamps, args.max_segment_time)
    print(f"After applying {args.max_segment_time}s max duration: {len(split_timestamps)} split points.")
    
    # Step 1.3: Split video at computed timestamps
    print(f"Splitting video into segments...")
    files = split_video_at_timestamps(input_path, split_dir, split_timestamps, video_name)
    print(f"Split into {len(files)} segments.")

    processed_files = []
    failed_segments = []

    print(f"=== Step 2: Processing segments with FlashVSR-Pro (with OOM retry) ===")
    for i, file_path in enumerate(files):
        print(f"\n[{i+1}/{len(files)}] Processing: {file_path.name}")
        
        # Check if output already exists (resume capability)
        existing_candidates = list(processed_dir.glob(f"*{file_path.stem}*.mp4"))
        if existing_candidates:
            print(f"[RESUME] Output found for {file_path.name}, skipping.")
            processed_files.append((file_path, existing_candidates[0]))
            continue
        
        # Define retry strategies for OOM handling
        strategies = [
            {"name": "Standard (No Tiling)", "args": []},
            {"name": "Tile DiT Only", "args": ["--tile-dit"]},
            {"name": "Tile DiT + VAE", "args": ["--tile-dit", "--tile-vae"]},
            {"name": "Small Tile (128/24)", "args": ["--tile-dit", "--tile-vae", "--tile-size", "128", "--overlap", "24"]}
        ]
        
        success = False
        output_file = None
        
        for strategy in strategies:
            print(f"  > Attempting: {strategy['name']}...")
            
            cmd = [
                "python", "infer.py",
                "-i", str(file_path),
                "-o", str(processed_dir),
                "--mode", args.mode,
                "--scale", args.scale,
                "--keep-audio"
            ] + strategy["args"]
            
            try:
                subprocess.run(cmd, check=True)
                
                # Force garbage collection
                gc.collect()
                
                # Find output file
                candidates = list(processed_dir.glob(f"*{file_path.stem}*.mp4"))
                if candidates:
                    output_file = candidates[0]
                    processed_files.append((file_path, output_file))
                    success = True
                    print(f"  > [SUCCESS] Processed using {strategy['name']}")
                    break
                else:
                    print(f"  > Warning: Command succeeded but output file not found.")
                    
            except subprocess.CalledProcessError as e:
                print(f"  > Failed: {strategy['name']} (Exit Code: {e.returncode})")
                gc.collect()
                continue
        
        if not success:
            print(f"!!! [FAILED] All retry strategies exhausted for {file_path.name}")
            failed_segments.append(file_path)

    print(f"\n=== Step 3: Checking results ===")
    print(f"Total segments: {len(files)}")
    print(f"Successfully processed: {len(processed_files)}")
    print(f"Failed: {len(failed_segments)}")
    
    if failed_segments:
        print(f"\n!!! FAILED SEGMENTS DETECTED !!!")
        print(f"The following segments failed to process:")
        with open(failed_log, "w", encoding="utf-8") as f:
            for seg in failed_segments:
                print(f"  - {seg}")
                f.write(f"{seg.absolute()}\n")
        
        print(f"\nFailed segment paths saved to: {failed_log}")
        print(f"Please manually process these segments, then re-run this script to merge.")
        print(f"Exiting without merging.")
        return
    
    print(f"\n=== Step 4: Merging all segments ===")
    # Create concat list
    list_file = work_dir / "concat_list.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for orig, processed in processed_files:
            f.write(f"file '{processed.absolute()}'\n")
    
    final_output = work_dir / f"{video_name}_FlashVSR_x{args.scale}_Final.mp4"
    
    # Merge with ffmpeg concat demuxer (preserves audio/video sync perfectly)
    merge_cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(final_output)
    ]
    
    try:
        subprocess.run(merge_cmd, check=True)
        print(f"\n=== SUCCESS! ===")
        print(f"Final video saved to: {final_output}")
        print(f"All audio preserved with seamless transitions.")
    except subprocess.CalledProcessError as e:
        print(f"\n!!! Merge failed with error code {e.returncode}")
        print(f"Check concat list: {list_file}")
        return

if __name__ == "__main__":
    main()