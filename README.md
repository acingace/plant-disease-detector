# Plant Disease Detector Web App

This repository contains a web-based application for automatic detection of plant diseases from images of leaves using deep learning. 
It leverages pre-trained ResNet models and ensemble techniques for improved accuracy, packaged into a Flask web application.

## Dataset

Name: PlantVillage Dataset
Description: 70,000+ labeled plant leaf images (healthy & diseased) across 9 species. Used ~10,500 images for training and ~2,000 for validation/testing due to hardware limitations.

## Features

- Upload plant leaf images for disease classification
- Ensemble of 5 fine-tuned ResNet18 models for accurate predictions
- Detects 38 different plant categories (healthy & diseased)
- Majority voting among models for prediction confidence
- Displays predicted disease, plant species, and model confidence

## Tech Stack
Backend: Python, Flask, PyTorch
Frontend: HTML/CSS (Flask templates)
Model: Fine-tuned ResNet18, ensemble of 5 models
Deployment: Local 

## Model Performance
Overall models have an accuracy of 95% and above

## How To Run The Project
- Download the dataset (https://www.kaggle.com/datasets/mohitsingh1804/plantvillage)
- Extract it and place it in an accessible directory on your machine.
- Install Dependencies (use a virtual machine)
- Run model.py to train the 5 base models
  - Where it says "Set directories and data", input the path of your downloaded dataset.
- Run the Flask App (app.py)
