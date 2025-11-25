from typing import Dict, Any, Optional
import os
import shutil
from .config import StabilizationConfig
from .preprocessing import VideoPreprocessor
from .postprocessing import VideoPostprocessor

from .optical_flow import OpticalFlowGenerator
from .geometry_estimation import GeometryEstimator
from .trajectory_smoothing import TrajectorySmoother
from .frame_warping import FrameWarper

class Deep3DStabilizer:
    """Main class that orchestrates the pipeline."""
    
    def __init__(self, config: StabilizationConfig):
        self.config = config
        self.preprocessor = VideoPreprocessor(config)
        self.postprocessor = VideoPostprocessor(config)
        
        # Initialize other components lazily or here
        # self.flow_generator = OpticalFlowGenerator(config)
        # self.geometry_estimator = GeometryEstimator(config)
        # self.trajectory_smoother = TrajectorySmoother(config)
        # self.frame_warper = FrameWarper(config)
        
    def stabilize(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """Execute full pipeline."""
        print(f"Starting Deep3D stabilization for {input_path}")
        
        # Validate config
        self.config.input_video_path = input_path
        self.config.output_video_path = output_path
        self.config.validate()
        
        try:
            # Stage 1: Extract frames
            frames_dir = self.preprocessor.extract_frames(input_path)
            
            # Initialize components that might depend on video metadata
            self.flow_generator = OpticalFlowGenerator(self.config)
            self.geometry_estimator = GeometryEstimator(self.config)
            self.trajectory_smoother = TrajectorySmoother(self.config)
            self.frame_warper = FrameWarper(self.config)
            
            # Stage 2: Generate optical flow
            flows_dir = self.flow_generator.generate(frames_dir)
            
            # Stage 3: Estimate depth and poses
            depths_dir, poses = self.geometry_estimator.optimize(frames_dir, flows_dir)
            
            # Stage 4: Smooth trajectory
            smooth_poses, compensations = self.trajectory_smoother.smooth(poses)
            
            # Stage 5: Warp frames
            stabilized_frames = self.frame_warper.warp(frames_dir, depths_dir, compensations)
            
            # Stage 6: Create output video
            self.postprocessor.create_video(stabilized_frames, output_path)
            
            return {'success': True}
            
        except Exception as e:
            print(f"Stabilization failed: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
        finally:
            # Cleanup temp dir if created by preprocessor
            if hasattr(self.preprocessor, 'temp_dir_obj'):
                self.preprocessor.temp_dir_obj.cleanup()
            # Cleanup other temp dirs if created
            if hasattr(self, 'flow_generator') and hasattr(self.flow_generator, 'temp_dir_obj'):
                self.flow_generator.temp_dir_obj.cleanup()
            if hasattr(self, 'geometry_estimator') and hasattr(self.geometry_estimator, 'temp_dir_obj'):
                self.geometry_estimator.temp_dir_obj.cleanup()

def stabilize_video_deep3d(input_video_path: str, output_video_path: str, **kwargs) -> Dict[str, Any]:
    """
    Stabilize video using Deep3D method.
    
    Args:
        input_video_path: Path to input shaky video
        output_video_path: Path to save stabilized video
        **kwargs: Configuration parameters (see StabilizationConfig)
    """
    config = StabilizationConfig(input_video_path, output_video_path, **kwargs)
    stabilizer = Deep3DStabilizer(config)
    return stabilizer.stabilize(input_video_path, output_video_path)
