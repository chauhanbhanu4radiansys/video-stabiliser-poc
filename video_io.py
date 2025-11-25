import cv2
import numpy as np
import os

class VideoReader:
    """Reads frames directly from video file using OpenCV."""
    
    def __init__(self, video_path: str):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
            
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        self.current_index = -1
        
    def __len__(self):
        return self.num_frames
        
    def get_frame(self, index: int) -> np.ndarray:
        """Get frame at specific index (RGB)."""
        if index < 0 or index >= self.num_frames:
            raise IndexError(f"Frame index {index} out of bounds (0-{self.num_frames-1})")
            
        # Optimize sequential access
        if index != self.current_index + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame at index {index}")
            
        self.current_index = index
        
        # Convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
        
    def close(self):
        if self.cap.isOpened():
            self.cap.release()
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
