import argparse
import os
import sys
from pathlib import Path

# Add parent dir to path
sys.path.append(str(Path(__file__).parent.parent))

from robust_2d.stabilizer import RobustStabilizer

def main():
    parser = argparse.ArgumentParser(description="Robust 2D Video Stabilizer")
    parser.add_argument("input_video", help="Path to input video")
    parser.add_argument("--output", "-o", default=None, help="Path to output video")
    parser.add_argument("--smoothing", type=int, default=30, help="Smoothing radius")
    parser.add_argument("--crop", type=float, default=0.8, help="Crop ratio")
    
    args = parser.parse_args()
    
    input_path = args.input_video
    if args.output:
        output_path = args.output
    else:
        p = Path(input_path)
        output_path = str(p.parent / f"{p.stem}_robust2d{p.suffix}")
        
    stabilizer = RobustStabilizer(input_path, output_path, args.smoothing, args.crop)
    stabilizer.stabilize()
    
if __name__ == "__main__":
    main()
