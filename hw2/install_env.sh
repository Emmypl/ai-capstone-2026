#!/bin/bash
# Environment setup for Open3D reconstruction pipeline

echo "Setting up Open3D reconstruction environment..."

# Update package lists
sudo apt-get update

# Install system dependencies required for Open3D and OpenCV headless
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0

# Install Python packages
pip3 install --upgrade pip
pip3 install numpy scipy opencv-python open3d

echo "Installation complete! You can now run the reconstruction pipeline:"
echo "python3 reconstruct.py"
