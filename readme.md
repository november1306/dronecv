# Long-Range Flying Object Detection and Tracking

## Project Overview

This project implements a proof-of-concept (POC) computer vision system for long-range detection and tracking of flying objects, such as drones. The system is designed for use with a handheld camera and utilizes a small scope in the center of the screen for target locking.

### Key Features

- Real-time tracking of distant flying objects
- Handheld camera operation
- Small, fixed-size scope for target acquisition
- Multiple detection and tracking methods
- Flexible pipeline for testing various components

## Motivation

The primary goals of this project are:
- Implement a CV/ML system for long-range object detection and tracking
- Focus on a proof-of-concept with scalability in mind
- Achieve a primary detection range of 200m, with potential for improvement up to 1km

## Technical Approach

Our system incorporates the following components:

1. **Video Processing**:
   - Frame extraction and processing from video files
   - Support for different frame rates and formats

2. **Object Detection**:
   - Multiple detection methods:
      - Background Subtraction (MOG2)
      - Frame Differencing
      - Optical Flow
   - Fixed-size scope for region of interest (ROI)

3. **Tracking**:
   - Implementation of MOSSE and CSRT trackers
   - Kalman filter for trajectory prediction

4. **Software Architecture**:
   - Video Capture Module
   - Object Detection Module
   - Object Tracking Module
   - Kalman Filter Module
   - Main Control Logic
   - Output Module
   - Optimization Layer

## Usage

1. Run the detection test:
   ```
   python test_detection.py
   ```

2. Adjust parameters in `test_detection.py` for different videos or detection settings:
   - `VIDEO_PATH`: Path to the input video file
   - `SCOPE_CENTER`: Center of the fixed detection scope
   - `SCOPE_SIZE`: Size of the detection scope
   - `NUM_FRAMES`: Number of frames to process
   - `START_FRAME`: Starting frame for processing
   - `FPS`: Frames per second to process (set to None for all frames)
   - `detector_type`: Type of detector to use ('mog2', 'frame_diff', or 'optical_flow')

## Results

### Strengths
- Multiple detection methods for robustness
- Fixed-size scope improves performance and focuses on relevant areas
- Integration of tracking for continuous object following
- Kalman filter for trajectory prediction
- Modular architecture for easy updates and improvements

### Limitations
- Currently limited to 200m range
- Potential false positives in complex backgrounds
- Computational limitations on current hardware

## Future Improvements

- Extend detection range up to 1km
- Implement and integrate deep learning-based object detection
- Optimize for real-time processing on target hardware
- Develop adaptive scope management for multiple or dynamic scopes
- Improve tracking robustness for temporary object disappearances
- Implement multi-object tracking capabilities
- Enhance visualization and user interface
- Conduct extensive field tests under various conditions

## Contributing

Contributions to improve the project are welcome. Please ensure to update tests as appropriate and adhere to the existing coding style.

## License

[MIT License](LICENSE)