# Face Mask Detection - Quick Commands

## Virtual Environment Setup

### Activate Existing Virtual Environment
```bash
.\.venv\Scripts\activate
```

### Create New Virtual Environment
```bash
C:\Users\Ankit\AppData\Local\Programs\Python\Python312\python.exe -m venv .venv
.\.venv\Scripts\activate
```

## Essential Commands

### Run Detection
```bash
python detect_mask_video.py
```
Main detection script for real-time face mask detection.

### Convert Model Format
```bash
python convert_model.py
```
Converts old `.model` format to new `.keras` format (if needed).

### Train Detector
```bash
python train_mask_detector.py
```
Train the mask detector model on your dataset.

