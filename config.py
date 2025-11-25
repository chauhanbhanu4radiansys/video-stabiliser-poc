from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import os

@dataclass
class StabilizationConfig:
    # Required (set by user)
    input_video_path: str
    output_video_path: str
    
    # Optional (with defaults)
    stability: int = 12
    crop_ratio: float = 0.8
    device: str = "cuda:0"
    batch_size: int = 80
    num_epochs: int = 100
    init_num_epochs: int = 300
    
    # Flow intervals
    intervals: List[int] = field(default_factory=lambda: [1, 4, 9])
    
    # Smoothing
    smooth_window: int = 59
    
    # Processing
    target_flow_resolution: Tuple[int, int] = (640, 384)
    img_mean: float = 0.45
    img_std: float = 0.225
    
    # Optimization
    learning_rate: float = 2e-4
    photometric_weight: float = 1.0
    flow_weight: float = 10.0
    geometry_weight: float = 0.5
    ssim_weight: float = 0.5
    adaptive_alpha: float = 1.2
    adaptive_beta: float = 0.85
    
    # Depth & Pose
    min_depth: float = 1e-3
    max_depth: float = 10.0
    rotation_mode: str = 'quat'
    
    # Temporary files
    temp_dir: Optional[str] = None
    keep_intermediates: bool = False
    
    # Output
    optimize_codec: bool = True
    
    # Runtime (filled during processing if None, or set by user)
    fps: Optional[float] = None
    width: Optional[int] = 192
    height: Optional[int] = 128
    
    def validate(self):
        """Validate configuration parameters."""
        assert 0 < self.crop_ratio < 1, "crop_ratio must be in (0, 1)"
        assert self.stability > 0, "stability must be positive"
        if not os.path.exists(self.input_video_path):
            raise FileNotFoundError(f"Input video not found: {self.input_video_path}")
        
    def update_metadata(self, metadata: Dict):
        """Update with video metadata."""
        self.fps = metadata.get('fps')
        self.width = metadata.get('width')
        self.height = metadata.get('height')
