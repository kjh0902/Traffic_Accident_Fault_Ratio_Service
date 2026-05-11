# Traffic_Accident_Fault_Ratio_Service
AI-based traffic accident fault ratio assessment service using computer vision and deep learning

# Traffic Accident Fault Ratio Service

This repository contains early-stage code for a traffic accident analysis project.

Current modules:
1. Object detection training for traffic accident images
2. Video-based multi-task classification for accident scene and accident type prediction
3. Data preprocessing for detection
4. Data preprocessing for video classification

## Project Structure
- detect/process.py: converts image annotations into COCO-style train/val/test files
- detect/train_detector.py: trains a Faster R-CNN detector
- classification/process_video_data.py: matches accident videos with labels and creates split json files
- classification/train_video_classifier.py: trains a video classifier using R2Plus1D

## Installation
Install PyTorch according to your CUDA version first.
Then:

pip install -r requirements.txt

## Usage

### Detection preprocessing
python detect/process.py

### Detection training
python detect/train_detector.py

### Video classification preprocessing
python classification/process_video_data.py

### Video classification training
python classification/train_video_classifier.py
