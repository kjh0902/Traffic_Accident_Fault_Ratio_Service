# Traffic Accident Fault Ratio Service

AI-based traffic accident fault ratio assessment service using computer vision and deep learning.

This repository contains early-stage code for a traffic accident analysis project.  
The current implementation focuses on two main tasks:

1. Object detection from traffic accident images
2. Multi-task video classification for accident-related factors

The final goal of this project is to support traffic accident fault ratio assessment by analyzing accident videos and images.

---

## Dataset

This project uses the AI Hub Traffic Accident Video Dataset.

Dataset source: AI Hub - 교통사고 영상 데이터

The dataset contains traffic accident videos and images for fault ratio measurement and evaluation. According to AI Hub, the dataset includes video and image data, JSON annotations, image bounding-box labels, and accident-related classification labels. The dataset was built for developing AI models that can support traffic accident fault ratio measurement and evaluation. :contentReference[oaicite:0]{index=0}

In this repository, only the following subset is used: 차대차_직선도로

The AI Hub page reports that the full dataset contains both video and image data, with 21,895 videos and 3,284,250 images in total. For the 차대차 직선 도로 category, it reports 9,089 videos and 1,363,350 bounding-box images.

---

## Project Structure

```text
Traffic_Accident_Fault_Ratio_Service/
├── README.md
├── requirements.txt
├── .gitignore
│
├── classification/
│   ├── process.py
│   ├── train_video.py
│   ├── video_data/
│   │   ├── raw/
│   │   └── processed/
│   └── video_classification_outputs/
│
├── detect/
│   ├── process.py
│   ├── train_detector.py
│   ├── img_data/
│   │   ├── raw/
│   │   └── processed/
│   └── detection_outputs/
```

## Expected Data Structure

```text
classification/video_data/raw/
├── VS_차대차_영상_직선도로/
│   ├── *.mp4
│   └── ...
└── VL_차대차_영상_직선도로/
    ├── *.json
    └── ...
```

```text
detect/img_data/raw/
├── VS_차대차_이미지_직선도로/
│   ├── *.png
│   └── ...
└── VL_차대차_이미지_직선도로/
    ├── *.json
    └── ...
```

## Installation

```text
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
