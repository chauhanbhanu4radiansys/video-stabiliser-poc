import os
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import glob
import cv2
from imageio import imread
from skimage.transform import resize as imresize
from .config import StabilizationConfig
from .models import ResnetEncoder, DepthDecoder, PoseDecoder, Warper, Loss
from .models.warper import pose_vec2mat, inverse_pose

class OptimizationDataset:
    """Simplified dataset loader for optimization, replacing SequenceIO."""
    def __init__(self, config: StabilizationConfig, frames_dir: str, flows_dir: str):
        self.config = config
        self.root = frames_dir
        self.flows_dir = flows_dir
        self.image_names = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
        
        # Get original size from first image
        sample = imread(self.image_names[0])
        self.origin_h, self.origin_w = sample.shape[:2]
        self.h, self.w = config.height, config.width
        
        # Intrinsics
        self.intrinsic = torch.FloatTensor([
            [500., 0, self.origin_w * 0.5], 
            [0, 500., self.origin_h * 0.5], 
            [0, 0, 1]
        ])
        
        self.intrinsic_res = self.intrinsic.clone()
        self.intrinsic_res[0] *= (self.w / self.origin_w)
        self.intrinsic_res[1] *= (self.h / self.origin_h)
        
        self.mean = np.array([config.img_mean] * 3, dtype=np.float32)
        self.std = np.array([config.img_std] * 3, dtype=np.float32)

    def __len__(self):
        return len(self.image_names)
        
    def get_intrinsic(self):
        return self.intrinsic_res

    def load_image(self, index):
        img = imread(self.image_names[index]).astype(np.float32)
        img = imresize(img, (self.h, self.w))
        img = np.transpose(img, (2, 0, 1))
        tensor_img = (torch.from_numpy(img).float() / 255 - self.mean[:, None, None]) / self.std[:, None, None]
        return tensor_img

    def load_flow_snippet(self, begin, end, interval):
        w, h = self.w, self.h
        W, H = self.origin_w, self.origin_h
        
        flows = []
        # We need flow from j to j+interval
        # The snippet range is [begin, end)
        # We can only load flows where j+interval < end?
        # Or does snippet imply we have frames up to end?
        # SequenceIO.load_snippet loads images [begin, end).
        # It loads flows for intervals.
        # Original code: range(begin, end - interval)
        # If end - begin <= interval, this range is empty.
        # But we need flows for the snippet.
        # If the snippet is short, maybe we can't compute loss for large intervals?
        # In original code, batch size is large enough.
        # Here we use batch_size=1.
        # If batch_size=1, begin=0, end=1. Interval=1. Range(0, 0) -> Empty.
        # So we can't use large intervals with small batch size if we strictly follow this?
        # But we need to compute loss.
        # If batch_size < interval, we can't compute flow loss for that interval within this batch?
        # Original code: if i >= bs: continue (in Loss.compute_loss_terms)
        # So if interval >= batch_size, we skip that interval for loss.
        # So we should return empty flows or handle it.
        
        # Let's return dummy flows if empty, and let Loss handle it (it checks batch size)
        
        for j in range(begin, end - interval):
            flow_path = os.path.join(self.flows_dir, str(interval), f"{j:05d}.npy")
            flows.append(np.load(flow_path))
            
        if not flows:
            # Return dummy tensor of correct shape (0, 2, h, w, 2)
            return torch.zeros((0, 2, self.h, self.w, 2))
            
        flows = np.stack(flows, 0) # (B, 2, H, W) or similar?
        # Check flow format from optical_flow.py: (2, H, W, 2) -> (2, H, W, 2)
        # Wait, optical_flow.py saves as (2, H, W, 2) where 0 is fwd, 1 is bwd
        # But SequenceIO expects something else?
        # SequenceIO: flows = np.stack([np.load...]) -> (B, 2, H, W, 2) ?
        # Let's check optical_flow.py save format:
        # flow_combined = np.stack([f_fwd_resized, f_bwd_resized], axis=0) -> (2, H, W, 2)
        # So flows here is (B, 2, H, W, 2)
        
        # SequenceIO logic:
        # flows[..., 0] = flows[..., 0] / W * w
        # flows[..., 1] = flows[..., 1] / H * h
        
        # My optical_flow.py already resizes flow to original resolution (W, H)
        # SequenceIO seems to resize it to (w, h) (model input size)
        
        b = flows.shape[0]
        
        # Resize flow values to model resolution
        flows[..., 0] = flows[..., 0] / W * w
        flows[..., 1] = flows[..., 1] / H * h
        
        # Interpolate flow field to model resolution
        # flows shape: (B, 2, H, W, 2)
        # We need to permute to use F.interpolate
        
        flows_tensor = torch.from_numpy(flows).float()
        # (B, 2, H, W, 2) -> (B*2, 2, H, W) -> (B*2, 2, h, w)
        flows_tensor = flows_tensor.view(b*2, H, W, 2).permute(0, 3, 1, 2)
        flows_tensor = torch.nn.functional.interpolate(flows_tensor, (h, w), mode='area')
        flows_tensor = flows_tensor.permute(0, 2, 3, 1).view(b, 2, h, w, 2)
        
        # Add grid
        grid_x = torch.arange(0, w).view(1, 1, 1, w).expand(b, 2, h, w).float()
        grid_y = torch.arange(0, h).view(1, 1, h, 1).expand(b, 2, h, w).float()
        
        flows_tensor[..., 0] += grid_x
        flows_tensor[..., 1] += grid_y
        
        # Normalize to [-1, 1]
        flows_tensor[..., 0] = 2 * (flows_tensor[..., 0] / (w - 1) - 0.5)
        flows_tensor[..., 1] = 2 * (flows_tensor[..., 1] / (h - 1) - 0.5)
        
        # Split into fwd and bwd
        # flows_tensor is (B, 2, h, w, 2)
        # fwd is flows_tensor[:, 0], bwd is flows_tensor[:, 1]
        
        return flows_tensor

    def load_snippet(self, begin, end):
        items = {}
        items['imgs'] = torch.stack([self.load_image(i) for i in range(begin, end)], 0)
        
        for i in self.config.intervals:
            flows = self.load_flow_snippet(begin, end, i)
            items[('flow_fwd', i)] = flows[:, 0]
            items[('flow_bwd', i)] = flows[:, 1]
            
        return items

class GeometryEstimator:
    def __init__(self, config: StabilizationConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize networks
        self.load_model()
        
    def load_model(self):
        # DispNet
        self.dispnet = {
            'encoder': ResnetEncoder(18, True, 3).to(self.device),
            'decoder': DepthDecoder(
                [64, 64, 128, 256, 512], # num_ch_enc for ResNet18
                scales=range(1), # Only scale 0
                num_output_channels=1,
                h=self.config.height, 
                w=self.config.width
            ).to(self.device)
        }
        
        # PoseNet
        self.posenet = {
            'encoder': ResnetEncoder(18, True, 3).to(self.device),
            'decoder': PoseDecoder(
                [64, 64, 128, 256, 512], 
                num_input_features=1, 
                num_frames_to_predict_for=1
            ).to(self.device)
        }
        
        # Optimizer
        params = [
            {'params': self.dispnet['encoder'].parameters(), 'initial_lr': self.config.learning_rate},
            {'params': self.dispnet['decoder'].parameters(), 'initial_lr': self.config.learning_rate},
            {'params': self.posenet['encoder'].parameters(), 'initial_lr': self.config.learning_rate},
            {'params': self.posenet['decoder'].parameters(), 'initial_lr': self.config.learning_rate}
        ]
        self.optimizer = optim.Adam(params, betas=(0.9, 0.99))
        
    def optimize(self, frames_dir: str, flows_dir: str) -> tuple[str, np.ndarray]:
        """Optimize depth and poses for all frames."""
        print("=> Optimizing geometry (depth & poses)...")
        
        # Setup dataset
        self.dataset = OptimizationDataset(self.config, frames_dir, flows_dir)
        
        # Setup output dir
        if self.config.temp_dir:
            depths_dir = os.path.join(self.config.temp_dir, "depths")
            os.makedirs(depths_dir, exist_ok=True)
        else:
            import tempfile
            self.temp_dir_obj = tempfile.TemporaryDirectory(prefix="deep3d_depths_")
            depths_dir = self.temp_dir_obj.name
            
        # Setup Warper and Loss
        # We need to mock 'opt' object for Loss class since it expects argparse namespace
        class OptMock:
            pass
        opt = OptMock()
        opt.cuda = self.config.device
        opt.scales = [0] # Only scale 0
        opt.intervals = self.config.intervals
        opt.ssim_weight = self.config.ssim_weight
        opt.width = self.config.width
        opt.height = self.config.height
        opt.photometric_loss = self.config.photometric_weight
        opt.flow_loss = self.config.flow_weight
        opt.geometry_loss = self.config.geometry_weight
        opt.adaptive_alpha = self.config.adaptive_alpha
        opt.adaptive_beta = self.config.adaptive_beta
        opt.min_depth = self.config.min_depth
        opt.max_depth = self.config.max_depth
        opt.rotation_mode = self.config.rotation_mode
        
        self.opt_mock = opt
        
        warper = Warper(opt, self.dataset.get_intrinsic()).to(self.device)
        self.loss_function = Loss(opt, warper)
        
        # Optimization Loop
        num_frames = len(self.dataset)
        batch_size = self.config.batch_size
        intervals = self.config.intervals
        max_interval = max(intervals)
        
        # Calculate snippet length logic from original code
        # snippet_len = 1 + int(np.ceil((len(self.seq_io) - opt.batch_size) / (opt.batch_size - max(opt.intervals))))
        snippet_len = 1 + int(np.ceil((num_frames - batch_size) / (batch_size - max_interval)))
        if snippet_len < 1: snippet_len = 1
        
        print(f"   Processing {snippet_len} snippets...")
        
        self.poses = None
        self.fix_depths = None
        
        # Initialization batch
        begin, end = self._get_batch_indices(0, batch_size, max_interval, num_frames, init=True)
        init_batch = self._load_batch_to_device(begin, end)
        self.loss_function.preprocess_minibatch_weights(init_batch)
        
        for _ in range(self.config.init_num_epochs):
            items = self._optimize_snippet(init_batch, prefix=0)
        self._update_state(items, prefix=0)
            
        # Main loop
        all_depths = []
        all_poses = []
        
        # We need to collect results. The original code saves incrementally.
        # We will save depths to disk and keep poses in memory.
        
        # Reset poses for main loop? Original code keeps self.poses
        
        for batch_idx in tqdm(range(snippet_len)):
            begin, end = self._get_batch_indices(batch_idx, batch_size, max_interval, num_frames)
            batch_items = self._load_batch_to_device(begin, end)
            self.loss_function.preprocess_minibatch_weights(batch_items)
            
            prefix = 0 if batch_idx == 0 else max_interval
            
            for _ in range(self.config.num_epochs):
                items = self._optimize_snippet(batch_items, prefix)
                
            self._update_state(items, prefix)
            
            # Save results
            # items['depths'] is a list of tensors (scales). We want scale 0.
            # items['depths'][0] shape: (B, H, W)
            # We need to save only the new part (excluding prefix)
            
            # Depth
            current_depths = items['depths'][0][-prefix:] if prefix > 0 else items['depths'][0]
            # Wait, original code: 
            # self.fix_depths = [items['depths'][s][-self.prefix:].detach() for s in self.opt.scales]
            # self.seq_io.save_depths(items['depths'], save_indices)
            # save_indices = list(range(start_idx, end))
            # If batch_idx > 0, start_idx is begin + prefix?
            # Original code: begin, end = get_batch_indices...
            # save_indices = list(range(begin, end))
            # But optimize_snippet returns full batch depths.
            # SequenceIO.save_depths iterates indices and saves corresponding depth from batch.
            # If batch has overlap (prefix), we overwrite?
            # Original code logic is tricky.
            # Let's simplify: we just need valid depths for all frames.
            # The overlap ensures continuity.
            
            # Let's save all depths in the batch, overwriting previous ones is fine/expected
            batch_depths = items['depths'][0].detach().cpu().numpy()
            for i, frame_idx in enumerate(range(begin, end)):
                np.save(os.path.join(depths_dir, f"{frame_idx:05d}.npy"), batch_depths[i])
                
            # Pose
            # self.poses accumulates poses.
            # We return the full accumulated poses at the end.
            
        # After loop, self.poses contains all poses?
        # Original code: self.poses = torch.cat([self.poses[:-self.prefix], poses], dim=0)
        # It maintains a growing list of poses.
        
        return depths_dir, self.poses.cpu().numpy()

    def _get_batch_indices(self, batch_idx, batch_size, max_interval, num_frames, init=False):
        if init or batch_idx <= 0:
            begin = 0
        else:
            begin = self.end - max_interval
            
        end = min(begin + batch_size, num_frames)
        self.end = end
        return begin, end

    def _load_batch_to_device(self, begin, end):
        items = self.dataset.load_snippet(begin, end)
        for k in items.keys():
            if isinstance(items[k], torch.Tensor):
                items[k] = items[k].to(self.device)
        return items

    def _optimize_snippet(self, items, prefix):
        # DispNet
        # Input: images. If prefix > 0, we might want to fix some depths?
        # Original code: d_features = self.dispnet['encoder'](items['imgs'][self.prefix:])
        # It only computes depth for new frames!
        
        if prefix > 0:
            imgs_input = items['imgs'][prefix:]
        else:
            imgs_input = items['imgs']
            
        d_features = self.dispnet['encoder'](imgs_input)
        d_outputs = self.dispnet['decoder'](d_features)
        
        depths = [d_outputs[('disp', 0)]] # Only scale 0
        # Scale depth
        depths = [d * self.opt_mock.max_depth + self.opt_mock.min_depth for d in depths]
        
        if prefix > 0:
            # Cat with fixed depths
            # self.fix_depths should be stored from previous iteration
            depths = [torch.cat([self.fix_depths[0][-prefix:], depths[0]], 0)]
            
        items['depths'] = depths
        
        # PoseNet
        # Input: images. 
        # Original: p_features = self.posenet['encoder'](items['imgs'][max(0, self.prefix-1):])
        # It includes one frame overlap for continuity?
        
        p_input_start = max(0, prefix - 1)
        p_features = self.posenet['encoder'](items['imgs'][p_input_start:])
        poses = self.posenet['decoder'](p_features)
        
        # Convert to matrix
        poses = pose_vec2mat(poses, self.opt_mock.rotation_mode)
        
        # Accumulate poses
        # poses is relative? No, pose_vec2mat returns transformation matrix.
        # Original code logic:
        # poses = inverse_pose(poses[0].view(-1, 4, 4)).expand_as(poses) @ poses
        # try: poses = self.poses[-1].expand_as(poses).to(device) @ poses
        
        # Normalize first pose to identity relative to itself?
        poses = inverse_pose(poses[0].view(-1, 4, 4)).expand_as(poses) @ poses
        
        if hasattr(self, 'poses') and self.poses is not None:
            # Chain with previous pose
            last_pose = self.poses[-1].to(self.device)
            poses = last_pose.expand_as(poses) @ poses
            
        if prefix > 0:
            # Concatenate with existing poses
            prev_poses = self.poses[-prefix:].to(self.device)
            poses = torch.cat([prev_poses, poses[1:]], 0)
            
        items['poses'] = poses
        items['poses_inv'] = inverse_pose(items['poses'])
        
        # Compute Loss
        loss_items = self.loss_function(items)
        
        self.optimizer.zero_grad()
        if isinstance(loss_items['full'], torch.Tensor):
            loss_items['full'].backward()
            self.optimizer.step()
        else:
            # No loss computed (e.g. batch too small for intervals)
            pass
        
        return items

    def _update_state(self, items, prefix):
        # Update fix_depths
        self.fix_depths = [d.detach() for d in items['depths']]
        
        # Update self.poses
        poses_detach = items['poses'].detach().cpu()
        if self.poses is None:
            self.poses = poses_detach
        else:
            if prefix == 0:
                self.poses = poses_detach
            else:
                self.poses = torch.cat([self.poses[:-prefix], poses_detach], dim=0)
