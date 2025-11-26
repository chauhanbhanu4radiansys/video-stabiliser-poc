from typing import Dict, Any, Optional
import os
import shutil
import time
from .config import StabilizationConfig
from .preprocessing import VideoPreprocessor
from .postprocessing import VideoPostprocessor

from .optical_flow import OpticalFlowGenerator
from .geometry_estimation import GeometryEstimator
from .trajectory_smoothing import TrajectorySmoother
from .frame_warping import FrameWarper

# Try to import psutil for memory tracking, fallback to basic tracking if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Memory tracking will be limited. Install with: pip install psutil")

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
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in MB."""
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            return {
                'rss': mem_info.rss / 1024 / 1024,  # Resident Set Size in MB
                'vms': mem_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
            }
        else:
            # Fallback: return zero if psutil not available
            return {'rss': 0, 'vms': 0}
    
    def _get_directory_size(self, dir_path: str) -> float:
        """Get total size of directory in MB."""
        if not os.path.exists(dir_path):
            return 0.0
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(dir_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except Exception:
            pass
        return total_size / 1024 / 1024  # Convert to MB
    
    def _format_time(self, seconds: float) -> str:
        """Format time in human-readable format."""
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.2f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}h {minutes}m {secs:.2f}s"
    
    def _cleanup_directory(self, dir_path: str, description: str = ""):
        """Clean up a directory if keep_intermediates is False."""
        if not self.config.keep_intermediates and dir_path and os.path.exists(dir_path):
            try:
                # Normalize paths for comparison
                dir_path_abs = os.path.abspath(dir_path)
                temp_dir_abs = os.path.abspath(self.config.temp_dir) if self.config.temp_dir else None
                
                # Check if it's a subdirectory of temp_dir (safe to delete)
                if temp_dir_abs and dir_path_abs.startswith(temp_dir_abs):
                    print(f"=> Cleaning up {description} ({dir_path})")
                    shutil.rmtree(dir_path)
                elif not self.config.temp_dir:
                    # If using system temp (temp_dir is None), it's safe to delete
                    # This handles TemporaryDirectory cases
                    print(f"=> Cleaning up {description} ({dir_path})")
                    shutil.rmtree(dir_path)
            except Exception as e:
                print(f"Warning: Failed to cleanup {dir_path}: {e}")
        
    def stabilize(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """Execute full pipeline."""
        print(f"Starting Deep3D stabilization for {input_path}")
        print("=" * 60)
        
        # Initialize timing and memory tracking
        start_time = time.time()
        initial_memory = self._get_memory_usage()
        stage_times = {}
        stage_memories = {}
        peak_memory = initial_memory.copy()
        intermediate_sizes = {}
        
        # Validate config
        self.config.input_video_path = input_path
        self.config.output_video_path = output_path
        self.config.validate()
        
        try:
            # Stage 1: Extract frames or Initialize VideoReader
            stage_start = time.time()
            
            if self.config.direct_video_read:
                print(f"=> Direct video reading enabled. Skipping frame extraction.")
                from .video_io import VideoReader
                self.video_reader = VideoReader(input_path)
                frames_dir = None # No frames directory
                frames_size = 0.0
                
                # Verify video properties match config if needed, or update config?
                # Actually config might have defaults, but we should respect video properties
                # But config.width/height are optimization targets, not video size.
                pass
            else:
                frames_dir = self.preprocessor.extract_frames(input_path)
                frames_size = self._get_directory_size(frames_dir)
                self.video_reader = None
                
            stage_time = time.time() - stage_start
            stage_times['frame_extraction'] = stage_time
            stage_memory = self._get_memory_usage()
            stage_memories['frame_extraction'] = stage_memory
            intermediate_sizes['frames'] = frames_size
            peak_memory['rss'] = max(peak_memory['rss'], stage_memory['rss'])
            print(f"  Time: {self._format_time(stage_time)} | Memory: {stage_memory['rss']:.1f} MB | Frames size: {frames_size:.1f} MB")
            
            # Initialize components that might depend on video metadata
            self.flow_generator = OpticalFlowGenerator(self.config)
            self.geometry_estimator = GeometryEstimator(self.config)
            self.trajectory_smoother = TrajectorySmoother(self.config)
            self.frame_warper = FrameWarper(self.config)
            
            # Stage 2: Generate optical flow
            stage_start = time.time()
            flows_dir = self.flow_generator.generate(frames_dir, self.video_reader)
            stage_time = time.time() - stage_start
            stage_times['optical_flow'] = stage_time
            stage_memory = self._get_memory_usage()
            stage_memories['optical_flow'] = stage_memory
            flows_size = self._get_directory_size(flows_dir)
            intermediate_sizes['flows'] = flows_size
            peak_memory['rss'] = max(peak_memory['rss'], stage_memory['rss'])
            print(f"  Time: {self._format_time(stage_time)} | Memory: {stage_memory['rss']:.1f} MB | Flows size: {flows_size:.1f} MB")
            
            # Stage 3: Estimate depth and poses
            # Note: Both frames_dir and flows_dir are needed for depth estimation
            stage_start = time.time()
            depths_dir, poses = self.geometry_estimator.optimize(frames_dir, flows_dir, self.video_reader)
            stage_time = time.time() - stage_start
            stage_times['geometry_estimation'] = stage_time
            stage_memory = self._get_memory_usage()
            stage_memories['geometry_estimation'] = stage_memory
            depths_size = self._get_directory_size(depths_dir)
            intermediate_sizes['depths'] = depths_size
            peak_memory['rss'] = max(peak_memory['rss'], stage_memory['rss'])
            print(f"  Time: {self._format_time(stage_time)} | Memory: {stage_memory['rss']:.1f} MB | Depths size: {depths_size:.1f} MB")
            
            # Cleanup flows after depth estimation (no longer needed)
            self._cleanup_directory(flows_dir, "optical flows")
            
            # Stage 4: Smooth trajectory
            stage_start = time.time()
            smooth_poses, compensations = self.trajectory_smoother.smooth(poses)
            stage_time = time.time() - stage_start
            stage_times['trajectory_smoothing'] = stage_time
            stage_memory = self._get_memory_usage()
            stage_memories['trajectory_smoothing'] = stage_memory
            peak_memory['rss'] = max(peak_memory['rss'], stage_memory['rss'])
            print(f"  Time: {self._format_time(stage_time)} | Memory: {stage_memory['rss']:.1f} MB")
            
            # Stage 5: Warp frames
            # Note: frames_dir is still needed for warping (loads frames from disk)
            stage_start = time.time()
            # Hybrid Stabilization: Get 2D transforms if enabled
            robust_transforms = None
            if self.config.hybrid_stabilization:
                print("=> Running Robust 2D Stabilizer for hybrid mode...")
                from .robust_2d.stabilizer import RobustStabilizer
                # We need a temporary output path for RobustStabilizer init, though we only use get_correction_transforms
                # Just use a dummy path
                dummy_out = os.path.join(self.config.temp_dir, "dummy_robust.mp4")
                rs = RobustStabilizer(self.config.input_video_path, dummy_out)
                robust_transforms = rs.get_correction_transforms()
                print("=> Robust 2D transforms calculated.")

            # 4. Warp frames
            stabilized_frames = self.frame_warper.warp(
                frames_dir, 
                depths_dir, 
                compensations, 
                video_reader=self.video_reader,
                robust_transforms=robust_transforms
            )
            stage_time = time.time() - stage_start
            stage_times['frame_warping'] = stage_time
            stage_memory = self._get_memory_usage()
            stage_memories['frame_warping'] = stage_memory
            peak_memory['rss'] = max(peak_memory['rss'], stage_memory['rss'])
            print(f"  Time: {self._format_time(stage_time)} | Memory: {stage_memory['rss']:.1f} MB")
            
            # Cleanup depths after warping (no longer needed)
            self._cleanup_directory(depths_dir, "depth maps")
            
            # Cleanup frames after warping (no longer needed)
            self._cleanup_directory(frames_dir, "frames")
            
            # Stage 6: Create output video
            stage_start = time.time()
            self.postprocessor.create_video(stabilized_frames, output_path)
            stage_time = time.time() - stage_start
            stage_times['video_creation'] = stage_time
            stage_memory = self._get_memory_usage()
            stage_memories['video_creation'] = stage_memory
            peak_memory['rss'] = max(peak_memory['rss'], stage_memory['rss'])
            print(f"  Time: {self._format_time(stage_time)} | Memory: {stage_memory['rss']:.1f} MB")
            
            # Calculate totals
            total_time = time.time() - start_time
            final_memory = self._get_memory_usage()
            
            # Print summary
            print("=" * 60)
            print("PROCESSING SUMMARY")
            print("=" * 60)
            print(f"Total processing time: {self._format_time(total_time)}")
            print(f"\nStage breakdown:")
            for stage, duration in stage_times.items():
                percentage = (duration / total_time) * 100
                print(f"  {stage.replace('_', ' ').title()}: {self._format_time(duration)} ({percentage:.1f}%)")
            
            print(f"\nMemory usage:")
            print(f"  Initial memory: {initial_memory['rss']:.1f} MB")
            print(f"  Peak memory: {peak_memory['rss']:.1f} MB")
            print(f"  Final memory: {final_memory['rss']:.1f} MB")
            print(f"  Memory increase: {final_memory['rss'] - initial_memory['rss']:.1f} MB")
            
            print(f"\nIntermediate file sizes (before cleanup):")
            total_intermediate = sum(intermediate_sizes.values())
            for name, size in intermediate_sizes.items():
                print(f"  {name}: {size:.1f} MB")
            print(f"  Total: {total_intermediate:.1f} MB")
            print("=" * 60)
            
            return {
                'success': True,
                'timing': {
                    'total_time': total_time,
                    'stage_times': stage_times
                },
                'memory': {
                    'initial': initial_memory,
                    'peak': peak_memory,
                    'final': final_memory
                },
                'intermediate_sizes': intermediate_sizes
            }
            
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
            if hasattr(self, 'video_reader') and self.video_reader:
                self.video_reader.close()

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
