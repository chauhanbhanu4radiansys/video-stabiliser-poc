import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import glob
from tqdm import tqdm
from .config import StabilizationConfig
from .models.warper import Warper
from .models import inverse_pose

class FrameWarper:
    def __init__(self, config: StabilizationConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Intrinsics (same as GeometryEstimator)
        # In a real implementation, we should share this or load it properly
        # For now, we replicate the logic
        self.origin_w = 0
        self.origin_h = 0
        
    def _init_intrinsics(self, w, h):
        # We need original dimensions to set intrinsics correctly
        # If we don't have them, we assume standard aspect ratio or use config
        # But Warper needs correct intrinsics
        
        # Let's assume we get original dimensions from the first frame
        self.origin_w = w
        self.origin_h = h
        
        self.intrinsic = torch.FloatTensor([
            [500., 0, self.origin_w * 0.5], 
            [0, 500., self.origin_h * 0.5], 
            [0, 0, 1]
        ])
        
        self.intrinsic_res = self.intrinsic.clone()
        self.intrinsic_res[0] *= (self.config.width / self.origin_w)
        self.intrinsic_res[1] *= (self.config.height / self.origin_h)
        
        # Initialize Warper
        # Warper expects 'opt' with height, width, min_depth
        class OptMock:
            pass
        opt = OptMock()
        opt.height = self.config.height
        opt.width = self.config.width
        opt.min_depth = 1.0 # Default
        
        self.warper = Warper(opt, self.intrinsic_res).to(self.device)
        
    def warp(self, frames_dir: str, depths_dir: str, compensations: np.ndarray, video_reader=None, robust_transforms=None) -> list[np.ndarray]:
        """Warp all frames using compensation transforms."""
        print("=> Warping frames...")
        
        if self.config.hybrid_stabilization and robust_transforms is not None:
            print(f"=> Hybrid stabilization enabled. Threshold: {self.config.warping_threshold}")
        
        if video_reader:
            num_frames = len(video_reader)
            frame_paths = list(range(num_frames))
            
            # Init intrinsics based on video properties
            h_orig = video_reader.height
            w_orig = video_reader.width
            self._init_intrinsics(w_orig, h_orig)
        else:
            ext = self.config.temp_frame_format
            frame_paths = sorted(glob.glob(os.path.join(frames_dir, f"*.{ext}")))
            
            if not frame_paths:
                return []
                
            # Init intrinsics based on first frame
            sample = cv2.imread(frame_paths[0])
            h_orig, w_orig = sample.shape[:2]
            self._init_intrinsics(w_orig, h_orig)
        
        depth_files = sorted(glob.glob(os.path.join(depths_dir, "*.npy")))
        
        # Compute warp maps
        warp_maps, crop_bounds = self._compute_warp_maps(depth_files, compensations)
        
        # Apply warps
        stabilized_frames = []
        
        crop_t, crop_b, crop_l, crop_r = crop_bounds
        
        # Validate crop bounds
        if crop_t >= crop_b or crop_l >= crop_r:
            print(f"Warning: Invalid crop bounds detected: t={crop_t}, b={crop_b}, l={crop_l}, r={crop_r}")
            print("Falling back to full frame (no crop).")
            crop_t, crop_b = 0, self.origin_h
            crop_l, crop_r = 0, self.origin_w
            
        crop_w = crop_r - crop_l
        crop_h = crop_b - crop_t
        
        print(f"Final crop: t={crop_t}, b={crop_b}, l={crop_l}, r={crop_r} ({crop_w}x{crop_h})")
        
        # Resize warp maps to original resolution for final warping
        # Original code: 
        # warp = F.interpolate(warp_maps[batch_begin:batch_end]..., (H, W), mode='bilinear'...)
        # reproj_imgs = F.grid_sample(imgs, warp)
        
        batch_size = self.config.batch_size
        
        for batch_begin in tqdm(range(0, len(frame_paths), batch_size)):
            batch_end = min(len(frame_paths), batch_begin + batch_size)
            
            # Load images
            imgs = []
            for i in range(batch_begin, batch_end):
                if video_reader:
                    # video_reader returns RGB
                    img = video_reader.get_frame(i)
                    # cv2.imread returns BGR, and we convert to RGB below.
                    # If we already have RGB, we skip conversion?
                    # Original code: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # So if video_reader gives RGB, we are good.
                else:
                    img = cv2.imread(frame_paths[i])
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                # Normalize? Original code: ((imgs * std + mean) * 255)
                # Normalize? Original code: ((imgs * std + mean) * 255)
                # SequenceIO loads as normalized tensor.
                # Here we load as uint8 numpy.
                # grid_sample expects float tensor.
                
                # We need to normalize to [0, 1] or use whatever grid_sample expects for pixel values
                # Actually grid_sample interpolates whatever values are there.
                # But we should convert to float tensor (B, C, H, W)
                
                img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
                imgs.append(img_tensor)
                
            imgs = torch.stack(imgs).to(self.device)
            
            # Get warp maps for this batch
            # warp_maps is (N, H, W, 2)
            # We need to slice it correctly
            current_warps = warp_maps[batch_begin:batch_end].to(self.device)
            
            if len(current_warps) != len(imgs):
                print(f"WARNING: Batch size mismatch! Warps: {len(current_warps)}, Imgs: {len(imgs)}")
                # This shouldn't happen if logic is correct
                # But if warp_maps has fewer frames than images?
                # warp_maps computation might have skipped frames?
                # Let's check _compute_warp_maps logic.
                pass
            
            # Resize warps to original resolution
            # warp_maps are (B, h, w, 2)
            # We need (B, H, W, 2)
            
            current_warps = current_warps.permute(0, 3, 1, 2) # (B, 2, h, w)
            current_warps = F.interpolate(
                current_warps, 
                (h_orig, w_orig), 
                mode='bilinear', 
                align_corners=False
            )
            current_warps = current_warps.permute(0, 2, 3, 1) # (B, H, W, 2)
            
            # Hybrid Stabilization Logic
            if self.config.hybrid_stabilization and robust_transforms is not None:
                # Get corresponding 2D transforms for this batch
                batch_indices = range(batch_begin, batch_end)
                batch_robust = robust_transforms[batch_indices] # (B, 3) [dx, dy, da]
                
                # Convert 2D transforms to dense warp maps (B, H, W, 2) in [-1, 1]
                robust_warps = self._create_robust_warps(batch_robust, h_orig, w_orig)
                robust_warps = robust_warps.to(self.device)
                
                # Calculate distortion metric for Deep3D warps
                # Metric: Mean absolute difference between Deep3D warp and 2D warp
                # This measures how much the 3D warp deviates from a rigid 2D transform
                # We can compute per-frame distortion
                
                # current_warps: (B, H, W, 2)
                # robust_warps: (B, H, W, 2)
                
                diff = torch.abs(current_warps - robust_warps)
                distortion = diff.mean(dim=(1, 2, 3)) # (B,)
                
                # Blend based on threshold
                # If distortion > threshold, fade to robust_warps
                # Soft transition?
                # alpha = clamp((distortion - thresh) / margin, 0, 1)
                # final = (1-alpha)*deep3d + alpha*robust
                
                thresh = self.config.warping_threshold
                margin = thresh * 0.5 # Transition range
                
                alpha = torch.clamp((distortion - thresh) / margin, 0.0, 1.0)
                
                # Expand alpha for broadcasting: (B, 1, 1, 1)
                alpha_expanded = alpha.view(-1, 1, 1, 1)
                
                current_warps = (1.0 - alpha_expanded) * current_warps + alpha_expanded * robust_warps
                
                # Debug print for high distortion
                if alpha.max() > 0:
                    high_dist_idx = torch.where(alpha > 0)[0]
                    # print(f"  Hybrid: Blending {len(high_dist_idx)} frames. Max alpha: {alpha.max().item():.2f}")

            # Apply warp
            reproj_imgs = F.grid_sample(imgs, current_warps, align_corners=False)
            
            # Crop
            reproj_imgs = reproj_imgs[..., crop_t:crop_b, crop_l:crop_r]
            
            # Convert back to numpy uint8
            reproj_imgs = (reproj_imgs * 255.0).clamp(0, 255).byte()
            reproj_imgs = reproj_imgs.permute(0, 2, 3, 1).cpu().numpy()
            
            for i in range(len(reproj_imgs)):
                stabilized_frames.append(reproj_imgs[i])
                
        return stabilized_frames

    def _compute_warp_maps(self, depth_files, compensations):
        """Compute warp maps for all frames."""
        print("   Computing warp maps...")
        
        warp_maps = []
        
        # Load depths and compensations
        # Process in batches to avoid OOM
        batch_size = self.config.batch_size
        num_frames = len(depth_files)
        
        compensate_poses = torch.from_numpy(compensations).float().to(self.device)
        
        # Track crop bounds
        crop_t, crop_b = 0, self.origin_h
        crop_l, crop_r = 0, self.origin_w
        
        # For crop calculation, we need to scale warps to original resolution
        # Original code logic:
        # batch_warps[..., 0] *= (W - 1)
        # batch_warps[..., 1] *= (H - 1)
        # t, b, l, r = get_cropping_area(...)
        
        for batch_begin in range(0, num_frames, batch_size):
            batch_end = min(num_frames, batch_begin + batch_size)
            
            # Load depths
            depths = []
            for i in range(batch_begin, batch_end):
                d = np.load(depth_files[i])
                depths.append(d)
            depths = np.stack(depths)
            depths = torch.from_numpy(depths).float().to(self.device)
            
            # Compute warps
            # warper.project_pixel returns src_pixel_coords in [-1, 1]
            batch_warps, _, _, _, _ = self.warper.project_pixel(
                depths, 
                compensate_poses[batch_begin:batch_end]
            )
            
            # Calculate cropping area
            # Convert to [0, W-1] and [0, H-1]
            batch_warps_pixel = (batch_warps + 1) / 2
            batch_warps_pixel[..., 0] *= (self.origin_w - 1)
            batch_warps_pixel[..., 1] *= (self.origin_h - 1)
            
            t, b, l, r = self._get_cropping_area(batch_warps_pixel, self.origin_h, self.origin_w)
            crop_t = max(crop_t, t)
            crop_b = min(crop_b, b)
            crop_l = max(crop_l, l)
            crop_r = min(crop_r, r)
            
            # Inverse flow for grid_sample
            # Original code: inverse_warps = warper.inverse_flow(batch_warps)
            # batch_warps here should be in pixel coords of SMALL resolution (w, h)
            # Wait, project_pixel returns [-1, 1].
            # inverse_flow expects pixel coords?
            # Original code:
            # batch_warps[..., 0] *= (w - 1) / (W - 1)  <-- scales back to small res?
            # batch_warps[..., 1] *= (h - 1) / (H - 1)
            # inverse_warps = warper.inverse_flow(batch_warps)
            
            # Let's follow original code exactly
            # 1. Scale to large res for crop calculation
            # (Done above in batch_warps_pixel)
            
            # 2. Scale to small res for inverse flow
            batch_warps_small = batch_warps_pixel.clone()
            batch_warps_small[..., 0] *= (self.config.width - 1) / (self.origin_w - 1)
            batch_warps_small[..., 1] *= (self.config.height - 1) / (self.origin_h - 1)
            
            inverse_warps = self.warper.inverse_flow(batch_warps_small)
            
            # Normalize inverse warps to [-1, 1] for grid_sample
            inverse_warps[..., 0] = inverse_warps[..., 0] * 2 / (self.config.width - 1) - 1
            inverse_warps[..., 1] = inverse_warps[..., 1] * 2 / (self.config.height - 1) - 1
            
            warp_maps.append(inverse_warps.detach().cpu())
            
        warp_maps = torch.cat(warp_maps, 0)
        
        # Apply user crop ratio if needed, or use calculated crop
        # Original code uses calculated crop.
        # But if crop is too aggressive, we might want to limit it?
        # For now, use calculated crop.
        
        return warp_maps, (crop_t, crop_b, crop_l, crop_r)

    def _get_cropping_area(self, warp_maps, h, w):
        # warp_maps: (B, H, W, 2) in pixel coords
        
        # Top border: max y value at top row
        border_t = warp_maps[:, 0, :, 1]
        # Filter valid values (>= 0)
        border_t = border_t[border_t >= 0]
        
        border_b = warp_maps[:, -1, :, 1]
        border_b = border_b[border_b >= 0]
        
        border_l = warp_maps[:, :, 0, 0]
        border_l = border_l[border_l >= 0]
        
        border_r = warp_maps[:, :, -1, 0]
        border_r = border_r[border_r >= 0]
        
        t = int(torch.ceil(torch.clamp(torch.max(border_t), 0, h))) if border_t.numel() > 0 else 0
        b = int(torch.floor(torch.clamp(torch.min(border_b), 0, h))) if border_b.numel() > 0 else h
        l = int(torch.ceil(torch.clamp(torch.max(border_l), 0, w))) if border_l.numel() > 0 else 0
        r = int(torch.floor(torch.clamp(torch.min(border_r), 0, w))) if border_r.numel() > 0 else w
        
        return t, b, l, r

    def _create_robust_warps(self, transforms, h, w):
        """Convert 2D affine transforms [dx, dy, da] to dense warp maps (B, H, W, 2) in [-1, 1]."""
        B = len(transforms)
        
        # Create meshgrid
        # grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        # grid = torch.stack([grid_x, grid_y], dim=-1).float() # (H, W, 2)
        # grid = grid.unsqueeze(0).repeat(B, 1, 1, 1) # (B, H, W, 2)
        
        # Actually, we can just construct the affine matrices and use F.affine_grid?
        # F.affine_grid takes (B, 2, 3) and returns (B, H, W, 2) in [-1, 1]
        # But F.affine_grid expects the matrix to map output [-1, 1] to input [-1, 1].
        
        # Our transforms are [dx, dy, da] in pixel space.
        # dx, dy are translation in pixels. da is rotation in radians.
        # We need to convert this to normalized coordinates [-1, 1].
        
        # Pixel space: x_new = cos(da)*x - sin(da)*y + dx
        # Normalized space: u = (2*x / (w-1)) - 1
        # This conversion is tedious.
        
        # Easier: Construct pixel-space grid, apply transform, normalize.
        
        device = self.device
        
        # 1. Create pixel grid (B, H, W, 3) homogeneous
        y_range = torch.arange(h, device=device)
        x_range = torch.arange(w, device=device)
        grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')
        ones = torch.ones_like(grid_x)
        grid = torch.stack([grid_x, grid_y, ones], dim=-1).float() # (H, W, 3)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1) # (B, H, W, 3)
        
        # 2. Construct affine matrices (B, 3, 3) inverse?
        # We want the warp map: for each output pixel (x,y), where does it come from in input?
        # Our 'transforms' are corrections: Input -> Stabilized.
        # So Stabilized = M * Input.
        # We want Input = M^-1 * Stabilized.
        # So we need the inverse of the correction transform.
        # Wait, 'transforms' from RobustStabilizer are 'difference' = Smoothed - Trajectory.
        # As established, 'difference' is the translation to apply to the frame.
        # So x_new = x_old + diff.
        # x_old = x_new - diff.
        # So we need to apply -diff.
        
        # Let's build the matrix for -diff.
        # dx, dy, da from transforms.
        # We want inverse: -dx, -dy, -da.
        # Approximately. For rotation, inverse is -angle.
        
        dx = -transforms[:, 0]
        dy = -transforms[:, 1]
        da = -transforms[:, 2]
        
        # Build matrices (B, 2, 3)
        # [ cos  -sin   dx ]
        # [ sin   cos   dy ]
        
        cos_a = np.cos(da)
        sin_a = np.sin(da)
        
        # Use torch
        cos_a = torch.from_numpy(cos_a).float().to(device)
        sin_a = torch.from_numpy(sin_a).float().to(device)
        dx = torch.from_numpy(dx).float().to(device)
        dy = torch.from_numpy(dy).float().to(device)
        
        # Compute grid coordinates
        # x_src = x_dst * cos - y_dst * sin + dx
        # y_src = x_dst * sin + y_dst * cos + dy
        
        # grid[..., 0] is x, grid[..., 1] is y
        
        x_dst = grid[..., 0]
        y_dst = grid[..., 1]
        
        # Reshape parameters for broadcasting: (B, 1, 1)
        cos_a = cos_a.view(B, 1, 1)
        sin_a = sin_a.view(B, 1, 1)
        dx = dx.view(B, 1, 1)
        dy = dy.view(B, 1, 1)
        
        x_src = x_dst * cos_a - y_dst * sin_a + dx
        y_src = x_dst * sin_a + y_dst * cos_a + dy
        
        # 3. Normalize to [-1, 1]
        u = (x_src / (w - 1)) * 2 - 1
        v = (y_src / (h - 1)) * 2 - 1
        
        warp = torch.stack([u, v], dim=-1) # (B, H, W, 2)
        
        return warp
