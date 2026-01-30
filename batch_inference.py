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

    # 遍历所有文件 / Walk through all files
    for root, dirs, files in os.walk(INPUT_DIR):
        for filename in files:
            file_path = Path(root) / filename
            
            # 1. 检查是不是视频 / Check extension
            if file_path.suffix.lower() not in VIDEO_EXTS:
                continue
            
            # 2. 检查是否在排除列表 / Check exclusion
            if filename in EXCLUDE_FILES:
                print(f"[SKIP] Excluding specific file: {file_path}")
                continue
                
            # 3. 计算相对路径，用于构建输出目录结构 / Calculate relative path
            # inputs/JIUTIAN-gen/video.mp4 -> JIUTIAN-gen/video.mp4
            rel_path = file_path.relative_to(INPUT_DIR)
            
            # 4. 构建输出目录 / Build output dir
            # results/JIUTIAN-gen/
            target_dir = OUTPUT_DIR / rel_path.parent
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # 5. 准备参数 / Prepare arguments
            # 默认为 tiny 模式, x2 倍率
            cmd = [
                "python", "infer.py",
                "-i", str(file_path),
                "-o", str(target_dir),
                "--mode", "tiny",
                "--scale", "2.0"
            ]
            
            # 6. 检查是否有特殊配置 (如 keep-audio) / Apply special configs
            if filename in SPECIAL_CONFIGS:
                extra_args = SPECIAL_CONFIGS[filename]
                cmd.extend(extra_args)
                print(f"[CONFIG] Applying specific args {extra_args} for {filename}")
            
            # 7. 执行推理 / Run inference
            print(f"Processing: {rel_path} -> {target_dir}")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                print(f"!!! Error occurred while processing {filename}")
                # 可以选择 continue 或 break，这里继续处理下一个
                continue

    print("Batch processing completed.")

if __name__ == "__main__":
    main()
