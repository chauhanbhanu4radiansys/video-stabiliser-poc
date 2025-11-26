import os
import subprocess
import cv2
from typing import List
import numpy as np
from tqdm import tqdm
from .config import StabilizationConfig

class VideoPostprocessor:
    def __init__(self, config: StabilizationConfig):
        self.config = config
        
    def create_video(
        self, 
        frames: List[np.ndarray], 
        output_path: str
    ):
        """Create video from stabilized frames."""
        if not frames:
            print("Warning: No frames to write!")
            return
            
        h, w = frames[0].shape[:2]
        fps = self.config.fps if self.config.fps else 30.0
        
        print(f"=> Writing {len(frames)} frames to {output_path} ({w}x{h} @ {fps}fps)")
        
        # Create video writer
        # Try mp4v first
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            output_path, 
            fourcc, 
            fps, 
            (w, h)
        )
        
        if not writer.isOpened():
            print("Failed to open video writer with mp4v, trying avc1")
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            writer = cv2.VideoWriter(
                output_path, 
                fourcc, 
                fps, 
                (w, h)
            )
        
        # Write frames
        for frame in tqdm(frames, desc="Writing video"):
            # Check if frame is valid
            if frame is None or frame.size == 0:
                print(f"Warning: Skipping empty frame at index {frames.index(frame) if frame in frames else 'unknown'}")
                continue
                
            # Convert RGB to BGR for OpenCV
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
        writer.release()
        
        # Optionally convert to h264 for better compatibility
        if self.config.optimize_codec:
            self._optimize_codec(output_path)
            
    def _optimize_codec(self, video_path):
        """Re-encode with h264 for better compatibility."""
        print("=> Optimizing codec (h264)...")
        temp_path = video_path + ".temp.mp4"
        
        # Use ffmpeg to re-encode
        cmd = [
            'ffmpeg',
            '-y', # Overwrite output
            '-i', video_path,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-pix_fmt', 'yuv420p', # Ensure compatibility
            temp_path
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.replace(temp_path, video_path)
        except subprocess.CalledProcessError:
            print("Warning: ffmpeg optimization failed. Keeping original file.")
            if os.path.exists(temp_path):
                os.remove(temp_path)
