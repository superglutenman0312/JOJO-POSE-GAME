# JoJo Pose AI: Real-Time Edge Pose Recognition System 🕺✨

A lightweight, real-time human pose recognition game optimized for edge devices (e.g., NVIDIA Jetson Xavier). Players are challenged to strike classic *JoJo's Bizarre Adventure* poses within a time limit, while the system evaluates their accuracy and immerses them into the anime background using real-time segmentation.

## 🚀 Engineering Highlights

This project was built with a strong focus on **performance optimization for resource-constrained edge devices**:

* **Lightweight Posture-Matching Algorithm:** Bypassed resource-heavy deep learning classification models. Instead, the system extracts 12-dimensional joint angle vectors via MediaPipe and calculates the Minimum Squared Error (MSE) considering 360-degree wrapping. This pure-math/geometric approach solves scale/distance variance and ensures high FPS on edge hardware.
* **Edge Device Deployment:** Successfully tested and deployed using a Linux host-to-target workflow on NVIDIA Jetson Xavier.
* **Real-Time End-to-End Pipeline:** Seamlessly integrates MediaPipe (Pose & Selfie Segmentation), OpenCV (dynamic video processing), and Pygame (audio/UI feedback) into a robust, low-latency system.

## 🛠️ Tech Stack
* **Language:** Python
* **Computer Vision:** OpenCV, MediaPipe (Pose Landmark & Selfie Segmentation)
* **Math & Matrix Operations:** NumPy
* **Audio & Multimedia:** Pygame

## 📂 Project Structure

Before running the game, ensure you have the required assets (images and sounds) in your project directory:

```text
├── play.py                 # Main game loop and UI rendering
├── compare.py              # MSE calculation and vector matching logic
├── compute_angle.py        # Converts 2D landmarks into 12D angle vectors
├── utils.py                # Helper functions for drawing, processing, and debugging
├── backgrounds/            # Directory for character background images
└── image_resize/           # Directory for pose hint images
