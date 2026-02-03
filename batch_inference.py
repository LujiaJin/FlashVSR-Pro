import os
import subprocess
from pathlib import Path
import sys

def main():
    # 基础配置 / Basic Config
    INPUT_DIR = Path("inputs")
    OUTPUT_DIR = Path("results")
    
    # 文件过滤规则 / Rules
    # 排除列表 (文件名)
    EXCLUDE_FILES = {
        "MyOwnSwordsman_S01E01.mp4",
        "MyOwnSwordsman_S01E01_1080p.mp4"
    }
    
    # 特殊配置列表 (文件名 -> 额外参数)
    SPECIAL_CONFIGS = {
        "example_audio.mp4": ["--keep-audio"]
    }
    
    # 视频后缀支持
    VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
    
    print(f"Starting batch inference...")
    print(f"Input Directory: {INPUT_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    
    if not INPUT_DIR.exists():
        print(f"Error: Input directory '{INPUT_DIR}' not found.")
        sys.exit(1)


    # 1. 常规批量推理
    processed_count = 0
    skipped_count = 0
    
    for root, dirs, files in os.walk(INPUT_DIR):
        for filename in files:
            file_path = Path(root) / filename
            
            # Check extension
            if file_path.suffix.lower() not in VIDEO_EXTS:
                continue
                
            # Check exclusion
            if filename in EXCLUDE_FILES:
                print(f"[SKIP] Excluding specific file: {file_path}")
                skipped_count += 1
                continue
            
            # Calculate output path and maintain directory structure
            try:
                rel_path = file_path.relative_to(INPUT_DIR)
            except ValueError:
                continue
                
            target_dir = OUTPUT_DIR / rel_path.parent
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Construct command
            # Requirements: tiny-long mode, x4 scale
            # Using tiny-long as requested to save memory on long videos
            base_cmd = [
                "python", "infer.py",
                "-i", str(file_path),
                "-o", str(target_dir),
                "--mode", "tiny-long",
                "--scale", "4.0"
            ]
            
            if filename in SPECIAL_CONFIGS:
                extra_args = SPECIAL_CONFIGS[filename]
                base_cmd.extend(extra_args)
                print(f"[CONFIG] Applying specific args {extra_args} for {filename}")
            
            print(f"[{processed_count+1}] Processing: {rel_path} -> {target_dir} (tiny-long x4)")
            
            # Define retry strategies
            # 1. Standard: No tiling (fastest)
            # 2. Tile DiT: Save DiT memory
            # 3. Tile DiT + VAE: Save VAE memory too
            # 4. Small Tile: Minimal memory usage (tile-size 128 is usually a safe multiple of 64/32)
            strategies = [
                {"name": "Standard (No Tiling)", "args": []},
                {"name": "Tile DiT", "args": ["--tile-dit"]},
                {"name": "Tile DiT + VAE", "args": ["--tile-dit", "--tile-vae"]},
                {"name": "Small Tile (Size 128/Overlap 24)", "args": ["--tile-dit", "--tile-vae", "--tile-size", "128", "--overlap", "24"]}
            ]
            
            success = False
            for strategy in strategies:
                print(f"  > Attempting: {strategy['name']}...")
                current_cmd = base_cmd + strategy["args"]
                
                try:
                    subprocess.run(current_cmd, check=True)
                    print(f"  > [SUCCESS] Processed using {strategy['name']}")
                    success = True
                    break
                except subprocess.CalledProcessError as e:
                    print(f"  > Failed: {strategy['name']} (Exit Code: {e.returncode})")
                    # Continue to next strategy
            
            if success:
                processed_count += 1
            else:
                print(f"!!! All attempts failed for {filename}. Skipping.")
                continue

    print(f"\nBatch processing completed. Processed: {processed_count}, Skipped: {skipped_count}")

if __name__ == "__main__":
    main()
