import numpy as np
import cv2
from tqdm import tqdm
import os

class RobustStabilizer:
    def __init__(self, input_path, output_path, smoothing_radius=30, crop_ratio=0.8):
        self.input_path = input_path
        self.output_path = output_path
        self.smoothing_radius = smoothing_radius
        self.crop_ratio = crop_ratio
        
    def get_correction_transforms(self):
        """Calculate and return the correction transforms for each frame."""
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open video")
            
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 1. Get transformations (frame-to-frame)
        transforms = self._get_transforms(cap, n_frames, w, h)
        cap.release()
        
        # 2. Smooth trajectory
        # trajectory[i] is pose at frame i relative to frame 0
        trajectory = np.cumsum(transforms, axis=0)
        smoothed_trajectory = self._smooth_trajectory(trajectory)
        
        # 3. Calculate difference (Correction)
        # We want to move from trajectory to smoothed_trajectory
        # correction = smoothed - trajectory
        difference = smoothed_trajectory - trajectory
        
        return difference

    def stabilize(self):
        print(f"Stabilizing {self.input_path}...")
        
        cap = cv2.VideoCapture(self.input_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        corrections = self.get_correction_transforms()
        
        # 4. Apply stabilization
        self._apply_stabilization(corrections, w, h, fps)
        
        print(f"Stabilization complete. Saved to {self.output_path}")
        
    def _get_transforms(self, cap, n_frames, w, h):
        transforms = np.zeros((n_frames, 3), np.float32)
        
        _, prev = cap.read()
        if prev is None:
            return transforms
            
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        
        for i in tqdm(range(n_frames-1), desc="Estimating Motion"):
            success, curr = cap.read()
            if not success:
                break
                
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
            
            # Detect features
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
            
            if prev_pts is None:
                # No features found, assume no motion
                transforms[i] = [0, 0, 0]
                prev = curr
                prev_gray = curr_gray
                continue
                
            # Track features
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
            
            # Filter valid points
            idx = np.where(status==1)[0]
            prev_pts = prev_pts[idx]
            curr_pts = curr_pts[idx]
            
            # Estimate affine transform (translation + rotation)
            # We use estimateAffinePartial2D for 4 degrees of freedom (zoom, rotation, translation)
            # But standard stabilization often just uses translation + rotation (Euclidean)
            # Let's use estimateAffinePartial2D
            m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
            
            if m is None:
                transforms[i+1] = [0, 0, 0]
            else:
                # Extract dx, dy, da (angle)
                dx = m[0, 2]
                dy = m[1, 2]
                da = np.arctan2(m[1, 0], m[0, 0])
                transforms[i+1] = [dx, dy, da]
                
            prev = curr
            prev_gray = curr_gray
            
        return transforms
        
    def _smooth_trajectory(self, trajectory):
        smoothed = np.copy(trajectory)
        radius = self.smoothing_radius
        
        for i in range(len(trajectory)):
            start = max(0, i - radius)
            end = min(len(trajectory), i + radius + 1)
            
            smoothed[i] = np.mean(trajectory[start:end], axis=0)
            
        return smoothed
        
    def _apply_stabilization(self, corrections, w, h, fps):
        cap = cv2.VideoCapture(self.input_path)
        
        # Define codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (w, h))
        
        # Calculate crop border
        # We want to zoom in slightly to hide borders
        # crop_ratio 0.8 means we keep 80% of the image
        # So we zoom in by 1/0.8 = 1.25
        
        # Fix border artifacts by zooming
        # Scale factor
        scale = 1.0 / self.crop_ratio
        center = (w/2, h/2)
        
        # Zoom matrix
        M_zoom = cv2.getRotationMatrix2D(center, 0, scale)
        
        for i in tqdm(range(len(corrections)), desc="Stabilizing"):
            success, frame = cap.read()
            if not success:
                break
                
            dx = corrections[i, 0]
            dy = corrections[i, 1]
            da = corrections[i, 2]
            
            # Construct affine matrix
            m = np.zeros((2, 3), np.float32)
            m[0, 0] = np.cos(da)
            m[0, 1] = -np.sin(da)
            m[1, 0] = np.sin(da)
            m[1, 1] = np.cos(da)
            m[0, 2] = dx
            m[1, 2] = dy
            
            # Apply stabilization transform
            frame_stabilized = cv2.warpAffine(frame, m, (w, h))
            
            # Apply zoom to hide borders
            frame_zoomed = cv2.warpAffine(frame_stabilized, M_zoom, (w, h))
            
            out.write(frame_zoomed)
            
        cap.release()
        out.release()
