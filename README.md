# Video Stabilizer POC - Deep3D Stabilization

A proof-of-concept implementation for video stabilization using Deep3D methods. This project stabilizes shaky videos by estimating depth and camera poses, then applying trajectory smoothing and frame warping.

## Features

- **Deep3D Stabilization**: Uses depth estimation and pose optimization for high-quality stabilization
- **Optical Flow Generation**: Utilizes RAFT (Recurrent All-Pairs Field Transforms) for accurate flow estimation
- **Trajectory Smoothing**: Applies Gaussian smoothing to camera trajectories
- **Frame Warping**: Warps frames based on estimated depth and compensation transforms
- **GPU Acceleration**: Supports CUDA for faster processing

## Prerequisites

### System Requirements

- **Python**: 3.8 or higher
- **CUDA**: Compatible GPU with CUDA support (optional but recommended)
- **FFmpeg**: Required for video processing
  ```bash
  # Ubuntu/Debian
  sudo apt-get install ffmpeg
  
  # macOS
  brew install ffmpeg
  ```

### Python Dependencies

The project requires the following Python packages:

- `torch` (PyTorch)
- `torchvision`
- `numpy`
- `opencv-python` (cv2)
- `tqdm`
- `imageio`
- `scikit-image`
- `scipy`

## Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd video-stabiliser-poc
```

### 2. Create a Conda Environment (Recommended)

```bash
# Create a new conda environment
conda create -n stabvideo python=3.10

# Activate the environment
conda activate stabvideo
```

### 3. Install Dependencies

#### Option A: Using pip with requirements.txt (Recommended)

```bash
# First, install PyTorch with CUDA support (adjust CUDA version as needed)
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only:
# pip install torch torchvision torchaudio

# Then install all other dependencies from requirements.txt
pip install -r requirements.txt
```

#### Option B: Using Conda

```bash
# Install PyTorch with CUDA support (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
conda install numpy opencv tqdm imageio scikit-image scipy -c conda-forge
```

#### Option C: Manual pip installation

```bash
# Install PyTorch (adjust based on your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies manually
pip install numpy>=1.21.0 scipy>=1.7.0 opencv-python>=4.5.0 scikit-image>=0.19.0 imageio>=2.9.0 tqdm>=4.62.0
```

### 4. Verify Installation

```bash
# Check if PyTorch can detect your GPU (if available)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Usage

### Basic Usage

Run the video stabilizer with a single command:

```bash
python run_video_stabiliser.py /path/to/input/video.mp4
```

### Command-Line Options

```bash
# Basic usage (output will be saved to output/ directory)
python run_video_stabiliser.py input_video.mp4

# Specify custom output path
python run_video_stabiliser.py input_video.mp4 --output custom_output.mp4

# Or use short form
python run_video_stabiliser.py input_video.mp4 -o custom_output.mp4

# Use absolute paths
python run_video_stabiliser.py /absolute/path/to/input.mp4
```

### Output

- **Stabilized video**: Saved to `output/` directory by default
  - Default naming: `{input_filename}_deep3d.{extension}`
  - Example: `input.mp4` → `output/input_deep3d.mp4`

- **Temporary files**: Created in `temp/` directory during processing
  - `temp/frames/`: Extracted video frames
  - `temp/flows/`: Optical flow data
  - `temp/depths/`: Depth maps
  - **Note**: Intermediate files are automatically cleaned up after each stage to save disk space

### Performance Metrics

The script automatically tracks and displays performance metrics:

- **Processing Time**: Total time and per-stage breakdown
  - Frame extraction time
  - Optical flow generation time
  - Geometry estimation time (usually the longest stage)
  - Trajectory smoothing time
  - Frame warping time
  - Video creation time

- **Memory Usage**: Tracks memory consumption throughout processing
  - Initial memory usage
  - Peak memory usage (maximum during processing)
  - Final memory usage
  - Memory increase from start to finish

- **Intermediate File Sizes**: Shows disk space used by temporary files
  - Frames directory size
  - Flows directory size
  - Depths directory size
  - Total intermediate storage

**Example output:**
```
============================================================
PROCESSING SUMMARY
============================================================
Total processing time: 5m 23.45s

Stage breakdown:
  Frame Extraction: 12.34s (3.8%)
  Optical Flow: 1m 45.67s (32.5%)
  Geometry Estimation: 2m 58.12s (55.2%)
  Trajectory Smoothing: 0.23s (0.1%)
  Frame Warping: 15.89s (4.9%)
  Video Creation: 11.20s (3.5%)

Memory usage:
  Initial memory: 512.3 MB
  Peak memory: 2847.6 MB
  Final memory: 523.1 MB
  Memory increase: 10.8 MB

Intermediate file sizes (before cleanup):
  frames: 1250.4 MB
  flows: 456.2 MB
  depths: 234.8 MB
  Total: 1941.4 MB
============================================================
```

**Note**: Install `psutil` for accurate memory tracking:
```bash
pip install psutil
```
Without `psutil`, memory metrics will show as 0 MB.

### Configuration

You can modify stabilization parameters in `run_video_stabiliser.py`:

```python
result = stabilize_video_deep3d(
    input_video, 
    output_video, 
    stability=12,          # Smoothing strength (higher = smoother)
    crop_ratio=0.8,       # Output crop ratio (0-1)
    temp_dir=temp_dir,    # Temporary files directory
    num_epochs=5,         # Optimization epochs
    init_num_epochs=10,   # Initial optimization epochs
    batch_size=2,         # Batch size for processing
    intervals=[1]         # Flow intervals
)
```

## Project Structure

```
video-stabiliser-poc/
├── __init__.py              # Package initialization
├── config.py                # Configuration dataclass
├── stabilizer.py            # Main stabilization orchestrator
├── preprocessing.py         # Video frame extraction
├── optical_flow.py          # Optical flow generation (RAFT)
├── geometry_estimation.py   # Depth and pose estimation
├── trajectory_smoothing.py  # Camera trajectory smoothing
├── frame_warping.py         # Frame warping and compensation
├── postprocessing.py         # Video reconstruction
├── models/                  # Deep learning models
│   ├── __init__.py
│   ├── resnet_encoder.py
│   ├── depth_decoder.py
│   ├── pose_decoder.py
│   ├── warper.py
│   ├── loss.py
│   └── layers.py
├── run_video_stabiliser.py  # Entry point script
├── requirements.txt         # Python dependencies
├── temp/                    # Temporary files (gitignored)
├── output/                  # Output videos (gitignored)
└── README.md
```

## How It Works

1. **Frame Extraction**: Extracts frames from input video using FFmpeg
2. **Optical Flow**: Generates optical flow between frames using RAFT model
3. **Geometry Estimation**: Optimizes depth maps and camera poses using the flow data
4. **Trajectory Smoothing**: Applies Gaussian smoothing to camera trajectory
5. **Frame Warping**: Warps frames based on depth and compensation transforms
6. **Video Reconstruction**: Combines stabilized frames into output video

## Troubleshooting

### CUDA Out of Memory

If you encounter CUDA out of memory errors:

- Reduce `batch_size` in `run_video_stabiliser.py`
- Reduce `target_flow_resolution` in `config.py`
- Process shorter video segments

### FFmpeg Not Found

Ensure FFmpeg is installed and available in your PATH:

```bash
ffmpeg -version
```

### Import Errors

If you encounter import errors, ensure:

1. All dependencies are installed
2. You're using the correct conda environment
3. The package structure is intact

## Notes

- Processing time depends on video length, resolution, and hardware
- GPU acceleration significantly speeds up processing
- Temporary files in `temp/` can be large for long videos
- The `output/` and `temp/` directories are gitignored

## License

[Add your license information here]

## Contributing

[Add contribution guidelines if applicable]

