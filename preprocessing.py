import os
import subprocess
import cv2
import tempfile
import re
import glob
from .config import StabilizationConfig

class VideoPreprocessor:
    def __init__(self, config: StabilizationConfig):
        self.config = config
        
    def extract_frames(self, video_path: str) -> str:
        """Extract frames from video to temporary directory."""
        # Create temp directory
        if self.config.temp_dir:
            frames_dir = os.path.join(self.config.temp_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
        else:
            self.temp_dir_obj = tempfile.TemporaryDirectory(prefix="deep3d_frames_")
            frames_dir = self.temp_dir_obj.name
        # Get list of frames
        ext = self.config.temp_frame_format
        frames = sorted(glob.glob(os.path.join(frames_dir, f"*.{ext}")))
        
        print(f"=> Extracting frames to {frames_dir}")
        
        # Use ffmpeg to extract
        self._run_ffmpeg_extract(video_path, frames_dir)
        
        # Get metadata
        metadata = self._get_video_metadata(video_path)
        self.config.update_metadata(metadata)
        
        return frames_dir
        
    def _run_ffmpeg_extract(self, video_path, output_dir):
        # Use %05d.png for 5-digit numbering
        ext = self.config.temp_frame_format
        quality_flag = ""
        if ext == "jpg":
            # ffmpeg uses -q:v for jpg quality (1-31, lower is better)
            # We map 0-100 quality to 31-1
            q = int((100 - self.config.temp_frame_quality) * 30 / 100 + 1)
            quality_flag = f"-q:v {q}"
            
        cmd = f'ffmpeg -y -i "{video_path}" {quality_flag} "{output_dir}/%05d.{ext}"'
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
    def _get_video_metadata(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        
        return {
            'fps': fps,
            'width': width,
            'height': height
        }
