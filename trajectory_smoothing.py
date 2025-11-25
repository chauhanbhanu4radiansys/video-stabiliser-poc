import numpy as np
import scipy.signal
from scipy.spatial.transform import Rotation as R
from .config import StabilizationConfig

class TrajectorySmoother:
    def __init__(self, config: StabilizationConfig):
        self.config = config
        
    def smooth(self, poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Smooth camera trajectory and compute compensations."""
        print("=> Smoothing trajectory...")
        
        # Convert to trajectory vectors
        # Poses are 3x4 matrices [R|t]
        trajectory = self._poses_to_vectors(poses)
        
        # Apply smoothing
        smooth_trajectory = self._gaussian_smooth(trajectory)
        
        # Convert back to poses
        smooth_poses = self._vectors_to_poses(smooth_trajectory)
        
        # Compute compensation transforms
        # Compensation = Smooth^-1 * Original
        compensations = self._compute_compensations(poses, smooth_poses)
        
        return smooth_poses, compensations
        
    def _poses_to_vectors(self, poses):
        """Convert 3x4 matrices to 7D vectors [tx, ty, tz, qx, qy, qz, qw]."""
        n = len(poses)
        vectors = np.zeros((n, 7))
        
        for i in range(n):
            pose = poses[i]
            if pose.shape == (4, 4):
                pose = pose[:3]
                
            t = pose[:3, 3]
            r_mat = pose[:3, :3]
            r = R.from_matrix(r_mat)
            q = r.as_quat()
            
            vectors[i, :3] = t
            vectors[i, 3:] = q
            
        return vectors
        
    def _vectors_to_poses(self, vectors):
        """Convert 7D vectors back to 3x4 matrices."""
        n = len(vectors)
        poses = np.zeros((n, 3, 4))
        
        for i in range(n):
            t = vectors[i, :3]
            q = vectors[i, 3:]
            r = R.from_quat(q)
            r_mat = r.as_matrix()
            
            poses[i, :3, :3] = r_mat
            poses[i, :3, 3] = t
            
        return poses
        
    def _gaussian_smooth(self, trajectory: np.ndarray) -> np.ndarray:
        """Apply Gaussian weighted moving average."""
        n_frames = len(trajectory)
        smoothed = np.zeros_like(trajectory)
        
        half_window = self.config.smooth_window // 2
        
        for i in range(n_frames):
            # Get window bounds
            start = max(0, i - half_window)
            end = min(n_frames, i + half_window + 1)
            
            # Generate Gaussian weights
            window_size = end - start
            # We need to center the gaussian on the current frame 'i'
            # The full window would be 2*half_window + 1
            # We take a slice of the gaussian corresponding to our available frames
            
            full_window_size = 2 * half_window + 1
            full_weights = scipy.signal.windows.gaussian(
                full_window_size, 
                self.config.stability
            )
            
            # Determine which part of the gaussian to use
            # i corresponds to index 'half_window' in full_weights
            # start corresponds to i - (i - start)
            
            offset_start = half_window - (i - start)
            offset_end = half_window + (end - i)
            
            weights = full_weights[offset_start:offset_end]
            weights /= weights.sum()
            
            # Weighted average
            smoothed[i] = self._weighted_pose_average(
                trajectory[start:end], 
                weights
            )
            
        return smoothed
        
    def _weighted_pose_average(self, poses, weights):
        """Average poses with weights (handles rotation specially)."""
        # Translation: simple weighted average
        t_avg = np.average(poses[:, :3], axis=0, weights=weights)
        
        # Rotation: use scipy Rotation averaging
        # Note: Scipy's Rotation.mean() handles quaternion averaging correctly
        rotations = R.from_quat(poses[:, 3:])
        r_avg = rotations.mean(weights=weights)
        
        return np.concatenate([t_avg, r_avg.as_quat()])
        
    def _compute_compensations(self, original_poses, smooth_poses):
        """Compute transform from original to smooth pose."""
        # We want: Smooth = Compensation * Original
        # So: Compensation = Smooth * Original^-1
        # Wait, usually we warp FROM original TO smooth
        # Point in original frame -> Point in world -> Point in smooth frame
        # P_world = T_orig * P_orig
        # P_smooth = T_smooth^-1 * P_world
        # P_smooth = T_smooth^-1 * T_orig * P_orig
        # So Compensation = T_smooth^-1 * T_orig
        
        n = len(original_poses)
        compensations = np.zeros((n, 4, 4))
        
        for i in range(n):
            # Construct 4x4 matrices
            T_orig = np.eye(4)
            if original_poses[i].shape == (4, 4):
                T_orig = original_poses[i]
            else:
                T_orig[:3] = original_poses[i]
            
            T_smooth = np.eye(4)
            T_smooth[:3] = smooth_poses[i]
            
            T_smooth_inv = np.linalg.inv(T_smooth)
            
            comp = T_smooth_inv @ T_orig
            compensations[i] = comp
            
        return compensations
