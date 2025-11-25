import os
import torch
import numpy as np
import cv2
import glob
from tqdm import tqdm
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
import torchvision.transforms.functional as F
from .config import StabilizationConfig

class OpticalFlowGenerator:
    def __init__(self, config: StabilizationConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = self._load_raft_model()
        
    def _load_raft_model(self):
        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights, progress=False)
        model = model.to(self.device)
        model.eval()
        return model
        
    def generate(self, frames_dir: str) -> str:
        """Generate optical flow for all frame intervals."""
        if self.config.temp_dir:
            flows_dir = os.path.join(self.config.temp_dir, "flows")
            os.makedirs(flows_dir, exist_ok=True)
        else:
            import tempfile
            self.temp_dir_obj = tempfile.TemporaryDirectory(prefix="deep3d_flows_")
            flows_dir = self.temp_dir_obj.name
            
        print(f"=> Generating optical flow to {flows_dir}")
        frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
        
        for interval in self.config.intervals:
            self._generate_interval(frame_paths, interval, flows_dir)
            
        return flows_dir
        
    def _generate_interval(self, frame_paths, interval, output_dir):
        """Generate flow at specific interval."""
        interval_dir = os.path.join(output_dir, str(interval))
        os.makedirs(interval_dir, exist_ok=True)
        
        # Target size for flow computation (to avoid OOM)
        W_target, H_target = self.config.target_flow_resolution
        
        # Batch size 1 to be safe
        batch_size = 1
        
        print(f"   Processing interval {interval}...")
        
        for batch_begin in tqdm(range(0, len(frame_paths), batch_size)):
            # Check if we have enough frames for this interval
            if batch_begin + interval >= len(frame_paths):
                break
                
            current_batch_end = min(len(frame_paths), batch_begin + batch_size + interval)
            current_frame_paths = frame_paths[batch_begin : current_batch_end]
            
            # Load and resize frames
            frames_orig = [cv2.imread(p) for p in current_frame_paths]
            frames = []
            for f in frames_orig:
                f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                f = cv2.resize(f, (W_target, H_target))
                frames.append(f)
                
            H_orig, W_orig = frames_orig[0].shape[:2]
            
            # Compute flow
            flow_fwd, flow_bwd = self._process_batch(frames, interval)
            
            if flow_fwd is None:
                continue
                
            # Convert to numpy
            flow_fwd = flow_fwd.permute(0, 2, 3, 1).cpu().numpy()
            flow_bwd = flow_bwd.permute(0, 2, 3, 1).cpu().numpy()
            
            # Save flows
            for i in range(len(flow_fwd)):
                frame_idx = batch_begin + i
                if frame_idx >= len(frame_paths) - interval:
                    break
                    
                # Resize flow back to original resolution and scale values
                f_fwd = flow_fwd[i]
                f_bwd = flow_bwd[i]
                
                f_fwd_resized = cv2.resize(f_fwd, (W_orig, H_orig))
                f_bwd_resized = cv2.resize(f_bwd, (W_orig, H_orig))
                
                scale_x = W_orig / float(W_target)
                scale_y = H_orig / float(H_target)
                
                f_fwd_resized[..., 0] *= scale_x
                f_fwd_resized[..., 1] *= scale_y
                f_bwd_resized[..., 0] *= scale_x
                f_bwd_resized[..., 1] *= scale_y
                
                flow_combined = np.stack([f_fwd_resized, f_bwd_resized], axis=0)
                save_path = os.path.join(interval_dir, f"{frame_idx:05d}.npy")
                if not os.path.exists(interval_dir):
                    print(f"WARNING: Directory {interval_dir} does not exist! Recreating...")
                    os.makedirs(interval_dir, exist_ok=True)
                np.save(save_path, flow_combined)

    def _process_batch(self, frames, interval):
        img1_batch = []
        img2_batch = []
        
        # Forward pairs
        for i in range(len(frames) - interval):
            img1 = F.to_tensor(frames[i]).to(self.device)
            img2 = F.to_tensor(frames[i + interval]).to(self.device)
            img1_batch.append(img1)
            img2_batch.append(img2)
            
        # Backward pairs
        for i in range(len(frames) - interval):
            img1 = F.to_tensor(frames[i + interval]).to(self.device)
            img2 = F.to_tensor(frames[i]).to(self.device)
            img1_batch.append(img1)
            img2_batch.append(img2)
            
        if not img1_batch:
            return None, None

        img1_batch = torch.stack(img1_batch)
        img2_batch = torch.stack(img2_batch)
        
        # Normalize
        transforms = Raft_Small_Weights.DEFAULT.transforms()
        img1_batch, img2_batch = transforms(img1_batch, img2_batch)
        
        with torch.no_grad():
            list_of_flows = self.model(img1_batch, img2_batch)
            predicted_flows = list_of_flows[-1]
            
        num_pairs = (len(frames) - interval)
        flow_fwd = predicted_flows[:num_pairs]
        flow_bwd = predicted_flows[num_pairs:]
        
        return flow_fwd, flow_bwd
