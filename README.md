# Wildlife Object Detection using YOLO26

A portfolio-quality deep learning project that fine-tunes **YOLO26** for wildlife object detection across **72 animal classes**. The project covers the complete machine learning pipeline—from raw dataset preparation and conversion to YOLO format, exploratory data analysis (EDA), model training, evaluation, and inference on images and videos.

---

## Features

* End-to-end object detection pipeline
* Dataset conversion to YOLO format
* Automatic train/validation/test split
* Dataset integrity checks
* Class distribution analysis
* Fine-tuning of pretrained YOLO26 models
* Model evaluation using standard detection metrics
* Image and video inference
* Exportable trained weights

---

## Project Structure

```text
├── wildlife_yolo.ipynb          # Complete project notebook
├── dataset.yaml                 # YOLO dataset configuration
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── testing/
│   ├── deer.jpg
│   ├── wildlife.mp4
│   └── wildlife2.mp4
├── yolo26s_72class_finetune_v1/
│   └── weights/
│       └── best.pt
└── README.md
```

---

## Technologies

* Python
* Ultralytics YOLO26
* PyTorch
* OpenCV
* NumPy
* Matplotlib
* Pillow
* PyYAML

---

## Training

The model is initialized from pretrained YOLO26 weights before being fine-tuned on the wildlife dataset.

Example training configuration:

```python
results = model.train(
    data="dataset.yaml",
    epochs=20,
    imgsz=512,
    batch=96,
    device=0,
    workers=8,
    amp=True,
    compile=True
)
```

---

## Evaluation

The trained model is evaluated using standard object detection metrics:

* Precision = 70.64%
* Recall = 66.09%
* mAP@50 = 70.78%
* mAP@50–95 = 62.94%

---

## Exploratory Data Analysis

The notebook includes:

* Class frequency visualization
* Dataset statistics
* Mean and median samples per class
* Detection of underrepresented classes
* Identification of empty annotations
* Verification of annotation consistency

---

## Getting Started

Clone the repository:

```bash
git clone https://github.com/yourusername/wildlife-yolo.git
cd wildlife-yolo
```

Install dependencies:

```bash
pip install ultralytics opencv-python matplotlib pyyaml pillow numpy
```

Run the notebook or perform inference using the provided weights.

---
