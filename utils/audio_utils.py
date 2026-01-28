# utils/audio_utils.py
"""
Audio utilities for FlashVSR-Pro
Handles audio extraction and preservation during video processing.
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
import warnings


def has_audio_stream(video_path):
    """Check if video file contains audio stream."""
    try:
        cmd = [
            'ffprobe', '-v', 'error', 
            '-select_streams', 'a', 
            '-show_entries', 'stream=codec_type', 
            '-of', 'csv=p=0', 
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return 'audio' in result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        warnings.warn(f"ffprobe not found or error checking audio in {video_path}")
        return False


def extract_audio(video_path, audio_output_path=None):
    """Extract audio from video file."""
    if audio_output_path is None:
        temp_dir = tempfile.mkdtemp()
        audio_output_path = os.path.join(temp_dir, 'extracted_audio.aac')
    
    try:
        cmd = [
            'ffmpeg', '-i', video_path, 
            '-vn', '-acodec', 'copy', 
            '-y', audio_output_path
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return audio_output_path, True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        warnings.warn(f"Failed to extract audio: {e}")
        return None, False


def merge_audio_video(video_path, audio_path, output_path):
    """Merge video and audio into single file."""
    try:
        cmd = [
            'ffmpeg', 
            '-i', video_path, 
            '-i', audio_path,
            '-c:v', 'copy', 
            '-c:a', 'aac', 
            '-map', '0:v:0', 
            '-map', '1:a:0',
            '-shortest', 
            '-y', output_path
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        warnings.warn(f"Failed to merge audio: {e}")
        shutil.copy2(video_path, output_path)
        return False


def copy_video_with_audio(original_video_path, processed_video_path, output_path):
    """
    Smart audio handling: preserve audio from original video if exists.
    
    Args:
        original_video_path: Original video with audio (optional)
        processed_video_path: Processed video without audio
        output_path: Final output with audio from original (if exists)
    """
    # Check if original has audio
    if not has_audio_stream(original_video_path):
        print("[Audio] No audio in original, saving silent video")
        shutil.copy2(processed_video_path, output_path)
        return True
    
    try:
        # Extract audio
        print("Extracting audio...")
        audio_path, success = extract_audio(original_video_path)
        if not success:
            warnings.warn("Audio extraction failed, outputting silent video")
            shutil.copy2(processed_video_path, output_path)
            return False
        
        # Create temp output
        temp_output = output_path + '.temp.mp4'
        
        # Merge audio with processed video
        print("Merging audio...")
        merge_success = merge_audio_video(processed_video_path, audio_path, temp_output)
        
        if merge_success:
            # Replace final output
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(temp_output, output_path)
            print("Audio merged successfully")
        else:
            print("Audio merge failed, using silent video")
            shutil.copy2(processed_video_path, output_path)
        
        # Clean up
        if os.path.exists(audio_path):
            os.remove(audio_path)
            temp_dir = os.path.dirname(audio_path)
            if temp_dir.startswith(tempfile.gettempdir()) and os.path.isdir(temp_dir):
                try:
                    os.rmdir(temp_dir)
                except OSError:
                    pass
        
        return merge_success
        
    except Exception as e:
        warnings.warn(f"Error in audio processing: {e}")
        shutil.copy2(processed_video_path, output_path)
        return False