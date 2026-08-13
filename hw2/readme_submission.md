# Homework 2: 3D Scene Reconstruction Pipeline

## Setup
Run the provided script: `./install_env.sh`

## How to Run

The main reconstruction script supports different floors via command-line arguments.

### First Floor
```bash
python reconstruct.py -f 1
```

### Second Floor
```bash
python reconstruct.py -f 2
```

## Project Structure
- `reconstruct.py`: Main pipeline.
- `load.py`: For data collection.
- `data_collection/`: Contains RGB and Depth frames, and Ground Truth poses.
- `install_env.sh`: Automated environment setup script.