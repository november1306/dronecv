"""
Environment setup verification script.
Run this script to check if all required dependencies are properly installed.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import matplotlib
import tqdm

def check_dependencies():
    print("Checking Python version...")
    print(f"Python version: {sys.version}")
    
    print("\nChecking OpenCV installation...")
    print(f"OpenCV version: {cv2.__version__}")
    
    print("\nChecking NumPy installation...")
    print(f"NumPy version: {np.__version__}")
    
    print("\nChecking Matplotlib installation...")
    print(f"Matplotlib version: {matplotlib.__version__}")
    
    print("\nChecking tqdm installation...")
    print(f"tqdm version: {tqdm.__version__}")
    
    print("\nChecking video directory...")
    video_dir = Path(__file__).parent.parent / "video"
    if video_dir.exists():
        print(f"Video directory found at: {video_dir}")
        video_files = list(video_dir.glob("*.avi"))
        if video_files:
            print(f"Found {len(video_files)} video files:")
            for video in video_files:
                print(f"  - {video.name}")
        else:
            print("No .avi video files found in video directory")
    else:
        print("Video directory not found. Please create a 'video' directory in the project root")
    
    print("\nChecking output directory...")
    output_dir = Path(__file__).parent.parent / "output"
    if not output_dir.exists():
        output_dir.mkdir(exist_ok=True)
        print("Created output directory")
    else:
        print("Output directory exists")

if __name__ == "__main__":
    print("DroneCV Environment Setup Check")
    print("==============================")
    try:
        check_dependencies()
        print("\nAll dependency checks completed successfully!")
        print("You can now run the detection tests using:")
        print("python -m src.test_detection")
    except Exception as e:
        print(f"\nError during environment check: {str(e)}")
        print("Please fix the above error and try again") 