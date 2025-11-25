import os
import subprocess
import cv2
import tempfile
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
            
        print(f"=> Extracting frames to {frames_dir}")
        
        # Use ffmpeg to extract
        self._run_ffmpeg_extract(video_path, frames_dir)
        
        # Get metadata
        metadata = self._get_video_metadata(video_path)
        self.config.update_metadata(metadata)
        
        return frames_dir
        
    def _run_ffmpeg_extract(self, video_path, output_dir):
        # Use %05d.png for 5-digit numbering
        output_pattern = os.path.join(output_dir, "%05d.png")
        cmd = [
            'ffmpeg', 
            '-y', # Overwrite output
            '-i', video_path, 
            '-vsync', '0', 
            '-q:v', '1', 
            output_pattern
        ]
        
        # Run quietly
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
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
