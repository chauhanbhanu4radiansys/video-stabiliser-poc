import os
import sys
import importlib
import importlib.util
from pathlib import Path

# Setup: Make the current directory importable as a package
# This ensures all relative imports (from .config, from .models, etc.) work correctly
package_dir = Path(__file__).parent.absolute()
parent_dir = package_dir.parent

# Add parent to path so we can import the package
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# The package name (directory name with hyphens replaced)
package_name = package_dir.name.replace('-', '_')

# Create package module structure for relative imports to work
# Use importlib machinery to properly set up the package so Python's import system recognizes it
import types
if package_name not in sys.modules:
    # Create package spec to properly initialize the package
    init_file = package_dir / "__init__.py"
    pkg_spec = importlib.util.spec_from_file_location(package_name, init_file)
    if pkg_spec is None:
        # Fallback to manual creation
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [str(package_dir)]
        pkg.__package__ = package_name
        pkg.__file__ = str(init_file)
        sys.modules[package_name] = pkg
    else:
        # Use the spec to create the package module properly
        pkg = importlib.util.module_from_spec(pkg_spec)
        pkg.__path__ = [str(package_dir)]
        pkg.__package__ = package_name
        sys.modules[package_name] = pkg

# Also ensure models subpackage is set up for nested imports
# This must be done BEFORE loading the main __init__.py
# The key is to ensure Python's import system can resolve relative imports
models_dir = package_dir / "models"
models_pkg_name = f"{package_name}.models"
if models_dir.exists() and models_pkg_name not in sys.modules:
    # Create the models package module structure
    # Python's import system will use __path__ to find modules within this package
    models_pkg = types.ModuleType(models_pkg_name)
    models_pkg.__path__ = [str(models_dir)]
    models_pkg.__package__ = models_pkg_name
    models_pkg.__file__ = str(models_dir / "__init__.py")
    # Ensure parent package is in sys.modules (it should be from above)
    if package_name not in sys.modules:
        sys.modules[package_name] = pkg
    sys.modules[models_pkg_name] = models_pkg
    
    # Manually load and execute models/__init__.py into the existing module object
    # We must do this because importlib.import_module() will just return the empty 
    # module we put in sys.modules without executing the file
    try:
        models_init_file = models_dir / "__init__.py"
        if models_init_file.exists():
            # Create spec
            spec = importlib.util.spec_from_file_location(models_pkg_name, models_init_file)
            if spec and spec.loader:
                # Execute the module using the existing module object
                # This populates the module in sys.modules
                spec.loader.exec_module(models_pkg)
    except Exception as e:
        print(f"Warning: Failed to execute models/__init__.py: {e}")
        # Don't raise here, see if it works anyway (maybe partial load)
        pass

# Load the package using importlib.import_module to properly handle all relative imports
# This ensures Python's import system correctly resolves all relative imports including nested packages
# Import the package's __init__ module explicitly to get the exports
init_module = importlib.import_module(f"{package_name}.__init__")
stabilize_video_deep3d = init_module.stabilize_video_deep3d

import argparse

# ... (imports section)

# ... (package setup section)

# Helper to parse arguments
parser = argparse.ArgumentParser(description="Verify Deep3D Stabilization")
parser.add_argument("input_video", nargs="?", default="unstable.mp4", help="Path to input video (absolute or relative)")
parser.add_argument("--output", "-o", default=None, help="Path to output video")
parser.add_argument("--stability", type=int, default=12, help="Stability score (lower = less smoothing, less warping)")
parser.add_argument("--smooth_window", type=int, default=59, help="Smoothing window size")
parser.add_argument("--crop_ratio", type=float, default=0.8, help="Crop ratio (lower = more crop)")
args = parser.parse_args()

input_video = args.input_video

# Create temp directory relative to the script's location (dynamic, not fixed path)
script_dir = os.path.dirname(os.path.abspath(__file__))
temp_dir = os.path.join(script_dir, "temp")

# Define output directory relative to script's location
output_dir = os.path.join(script_dir, "output")
os.makedirs(output_dir, exist_ok=True)

# Handle output path logic
if args.output:
    # If user specified output, check if it's just a filename or a path
    if os.path.dirname(args.output):
        output_video = args.output
    else:
        # If just a filename, put it in output directory
        output_video = os.path.join(output_dir, args.output)
else:
    # Default output naming based on input, placed in output directory
    input_path = Path(input_video)
    output_filename = f"{input_path.stem}_deep3d{input_path.suffix}"
    output_video = os.path.join(output_dir, output_filename)

print(f"Verifying Deep3D standalone function...")
print(f"Input: {input_video}")
print(f"Output: {output_video}")
print(f"Temp directory: {temp_dir}")
print(f"Output directory: {output_dir}")

# Check if input exists before running
if not os.path.exists(input_video):
    print(f"Error: Input video not found at {input_video}")
    sys.exit(1)

# Run stabilization
result = stabilize_video_deep3d(
    input_video, 
    output_video, 
    stability=args.stability,
    crop_ratio=args.crop_ratio,
    smooth_window=args.smooth_window,
    temp_dir=temp_dir,
    num_epochs=5, # Reduced for verification speed
    init_num_epochs=10, # Reduced for verification speed
    batch_size=2, # Reduced to fit in memory
    intervals=[1] # Reduced intervals to allow small batch size
)

if result['success']:
    print("Stabilization successful!")
    if os.path.exists(output_video):
        print(f"Output video created at {output_video}")
    else:
        print("Error: Output video file not found!")
else:
    print("Stabilization failed!")
    print(f"Error: {result.get('error')}")
