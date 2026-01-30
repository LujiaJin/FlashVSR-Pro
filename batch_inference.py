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
        "MyOwnSwordsman_S01E01.mp4"
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


    # 1. 常规批量推理（原有逻辑）
    # processed_files = []
    # for root, dirs, files in os.walk(INPUT_DIR):
    #     for filename in files:
    #         file_path = Path(root) / filename
    #         if file_path.suffix.lower() not in VIDEO_EXTS:
    #             continue
    #         if filename in EXCLUDE_FILES:
    #             print(f"[SKIP] Excluding specific file: {file_path}")
    #             continue
    #         rel_path = file_path.relative_to(INPUT_DIR)
    #         target_dir = OUTPUT_DIR / rel_path.parent
    #         target_dir.mkdir(parents=True, exist_ok=True)
    #         cmd = [
    #             "python", "infer.py",
    #             "-i", str(file_path),
    #             "-o", str(target_dir),
    #             "--mode", "tiny-long",
    #             "--scale", "4.0"
    #         ]
    #         if filename in SPECIAL_CONFIGS:
    #             extra_args = SPECIAL_CONFIGS[filename]
    #             cmd.extend(extra_args)
    #             print(f"[CONFIG] Applying specific args {extra_args} for {filename}")
    #         print(f"Processing: {rel_path} -> {target_dir}")
    #         try:
    #             subprocess.run(cmd, check=True)
    #             processed_files.append((file_path, rel_path, target_dir))
    #         except subprocess.CalledProcessError:
    #             print(f"!!! Error occurred while processing {filename}")
    #             continue

    # 2. 检查缺失的x4结果（有x2但无x4）并补充tile_dit处理
    print("\n[Tile补充] 检查缺失的x4结果并补充处理...")
    # 获取所有输入视频名
    input_dir = INPUT_DIR / "JIUTIAN-gen"
    result_dir = OUTPUT_DIR / "JIUTIAN-gen"
    input_videos = [f for f in input_dir.iterdir() if f.suffix.lower() in VIDEO_EXTS]
    # 生成x2/x4结果名
    missing_tile_list = []
    for f in input_videos:
        name = f.stem
        x2 = result_dir / f"FlashVSR-Pro_tiny_scale2.0_{name}.mp4"
        x4 = result_dir / f"FlashVSR-Pro_tiny-long_scale4.0_{name}.mp4"
        if x2.exists() and not x4.exists():
            missing_tile_list.append(f)
    if missing_tile_list:
        print("以下视频缺失x4结果，将用tile_dit补充处理：")
        for f in missing_tile_list:
            print(f"  {f}")
        for f in missing_tile_list:
            rel_path = f.relative_to(INPUT_DIR)
            target_dir = OUTPUT_DIR / rel_path.parent
            target_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                "python", "infer.py",
                "-i", str(f),
                "-o", str(target_dir),
                "--mode", "tiny-long",
                "--scale", "4.0",
                "--tile-dit",
                "--tile-size", "192",
                "--overlap", "32"
            ]
            print(f"[Tile补充] Processing: {rel_path} -> {target_dir}")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                print(f"!!! Error occurred while tile-processing {f.name}")
                continue
    else:
        print("所有x4结果均已存在，无需补充tile处理。")

    print("Batch processing completed.")

if __name__ == "__main__":
    main()
