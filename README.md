Real-Time Semantic Segmentation for Autonomous Driving

Project Overview

This project focuses on semantic segmentation of road scenes using the SegFormer-B0 deep learning architecture. The system identifies and classifies different objects and regions in road images to support scene understanding for autonomous driving applications.

The project uses a Cityscapes-related dataset and performs pixel-level classification across 19 semantic classes.

Problem Statement

Understanding road environments is essential for autonomous driving systems. Traditional image classification methods identify objects at image level but do not provide detailed information about their locations.

This project addresses this challenge by implementing semantic segmentation, which assigns a class label to individual pixels in an input road image.

Objectives

Implement semantic segmentation using SegFormer-B0.

Prepare and preprocess road images and segmentation masks.

Classify road scenes into 19 semantic categories.

Train and evaluate the deep learning model.

Develop a Streamlit-based interface to visualize segmentation predictions.

Technologies Used

Technology

Purpose

Python

Programming language

PyTorch

Deep learning framework

SegFormer-B0

Semantic segmentation architecture

Streamlit

Web application interface

NumPy

Numerical operations

PIL / OpenCV

Image processing

Matplotlib

Visualization

Dataset

The project uses a Cityscapes-related road-scene image and segmentation-mask dataset.

Dataset preparation includes:

Image and mask loading.

Image preprocessing and normalization.

Mapping mask colours to 19 semantic class IDs.

Training and validation split.

Dataset size: 2,975 image-mask samples
Training samples: 2,380
Validation samples: 595

The dataset files are not included in this repository.

Model Architecture

SegFormer-B0 is used as the semantic segmentation model. It is a transformer-based architecture designed for efficient semantic segmentation.

The model is trained using:

Optimizer: AdamW

Learning rate: 0.00005

Loss function: Cross Entropy Loss

Training epochs: 10

Batch size: 4

Ignored mask label: 255

Model Evaluation Results

The model was evaluated using pixel accuracy and mean Intersection over Union (mIoU).

Evaluation Metric

Result

Pixel Accuracy

91.92%

Mean IoU (mIoU)

44.12%

Evaluated Classes

19

These results represent the validation output recorded after the tenth training epoch.

Streamlit Application

The project includes a Streamlit-based interface that allows users to upload road images and visualize the segmentation results.

The application displays:

Original input image.

Predicted segmentation mask.

Segmentation overlay.

Detected semantic categories.

Pixel-share percentages for the predicted categories.

Repository Structure

real-time-semantic-segmentation/
│
├── app.py
├── evaluate_model.py
├── requirements.txt
└── README.md

Installation and Execution

1. Clone the repository

git clone https://github.com/sherlyravi2-cmyk/real-time-semantic-segmentation.git

2. Open the project directory

cd real-time-semantic-segmentation

3. Install dependencies

pip install -r requirements.txt

4. Add the trained model checkpoint

The trained model checkpoint is not included in this repository. Place the checkpoint in the location expected by app.py before running the application.

5. Run the Streamlit application

streamlit run app.py

Future Enhancements

Improve segmentation performance across different road scenes.

Evaluate inference speed and model efficiency.

Extend testing to additional road images.

Explore deployment with live camera input.

Conclusion

This project demonstrates the application of transformer-based deep learning for semantic segmentation in autonomous driving scenarios. SegFormer-B0 is used to classify road-scene images into 19 semantic categories, and a Streamlit interface provides visual access to the model's predictions.

Project: Real-Time Semantic Segmentation for Autonomous Driving
Model: SegFormer-B0
Dataset: Cityscapes-related road-scene dataset
