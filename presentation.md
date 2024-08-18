# Long-Range Flying Object Detection and Tracking

## Motivation
- Implement a computer vision system for long-range detection and tracking of flying objects
- Focus on proof-of-concept (POC) with scalability in mind
- Primary detection range: 20m, with potential improvement up to 1km

## Introduction
Our project aims to develop a real-time tracking system for distant flying objects, such as drones. Key features include:

- Handheld camera operation
- Small scope in the center of the screen for target locking
- Manual target acquisition by the operator

This system is designed with practical applications in mind, particularly for anti-drone aiming systems. 

For the current proof-of-concept (POC), we have focused on:

- Implementing a flexible pipeline for testing various components
- Evaluating different detectors, trackers, and computer vision tools
- Using pre-recorded videos for initial assessment and optimization

This approach allows us to lay the groundwork for a robust, real-world solution while providing valuable insights into system performance and areas for improvement.

## Description
Technical approach used in the project:
1. Video Processing:
   - Frame extraction and processing from video files
   - Support for different frame rates and formats

2. Object Detection:
   - Multiple detection methods:
     - Background Subtraction (MOG2)
     - Frame Differencing
     - Optical Flow
   - Fixed-size scope for region of interest (ROI)

3. Tracking:
   - Implementation of MOSSE and CSRT trackers
   - Kalman filter for trajectory prediction

4. Software Architecture:
   - Video Capture Module
   - Object Detection Module
   - Object Tracking Module
   - Kalman Filter Module
   - Main Control Logic
   - Output Module
   - Optimization Layer

## Demo
(Here, you would include screenshots, GIFs, or links to video demonstrations of the system in action)

## Results
Evaluation of results and system effectiveness:

Strengths:
- Multiple detection methods for robustness
- Fixed-size scope improves performance and focuses on relevant areas
- Integration of tracking for continuous object following
- Kalman filter for trajectory prediction
- Modular architecture for easy updates and improvements

Weaknesses:
- Limited to 20m range currently
- Potential false positives in complex backgrounds
- Computational limitations on current hardware

## Conclusions and Further Improvements
Future work and possible improvements:
- Extend detection range up to 1km
- Implement and integrate deep learning-based object detection
- Optimize for real-time processing on target hardware
- Develop adaptive scope management for multiple or dynamic scopes
- Improve tracking robustness for temporary object disappearances
- Implement multi-object tracking capabilities
- Enhance visualization and user interface for easier system operation
- Conduct extensive field tests under various environmental conditions

### Introduction to Existing Approaches
Existing approaches and relevant works in the field:
- Background subtraction methods (e.g., MOG2)
- Frame differencing techniques
- Optical flow for motion detection
- Machine learning-based object detection (e.g., YOLO, SSD)
- Multi-object tracking algorithms (e.g., SORT, DeepSORT)

These existing approaches provide a foundation for further improvements and can be integrated or compared with our current implementation to enhance the system's capabilities.
