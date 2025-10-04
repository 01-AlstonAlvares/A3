Car Price Prediction Project - MLOps Pipeline
This repository contains a machine learning project to predict the price category of used cars. It features a complete MLOps pipeline using GitHub Actions for Continuous Integration (CI) and Continuous Deployment (CD), with all experiments tracked on a remote MLflow server.

Project Overview
The core of this project is a multinomial logistic regression model that classifies used cars into one of four price brackets. The model is built from scratch in Python and trained on a preprocessed version of the car price dataset.

The main focus of this repository is the automated CI/CD pipeline which handles:

Continuous Integration (CI): Automatically testing the registered "Staging" model every time a new version is pushed.

Continuous Deployment (CD): Automatically building a Docker image and deploying the model-serving application to a remote server.

Project Structure
The repository is organized as follows:

.github/workflows/main.yml: The main GitHub Actions workflow file that defines the entire CI/CD pipeline.

app/: This directory contains all the application and testing code.

app.py: A Flask application that loads the trained model from MLflow and serves predictions via a REST API.

test_model.py: An automated unit test script that validates the "Staging" model from the MLflow registry.

requirements.txt: A list of all Python packages required to run the application and tests.

A3_Classification_Final_Improved.ipynb: The main Jupyter Notebook used for data preprocessing, feature engineering, model training, and experiment tracking with MLflow.

custom_classifier.py: A custom Python class containing the from-scratch implementation of the LogisticRegression model.

Dockerfile: Instructions for building a containerized version of the Flask application.

Model & Experiment Tracking
Model: A custom LogisticRegression model with L2 (Ridge) regularization, trained to solve a 4-class classification problem.

Feature Engineering: The model uses several engineered features for improved accuracy, including car_age and km_per_year.

Experiment Tracking: All training runs, including hyperparameter tuning, metrics, and model artifacts, are logged to a centralized MLflow server located at http://mlflow.ml.brain.cs.ait.ac.th/.

CI/CD Pipeline Workflow
The pipeline is defined in the .github/workflows/main.yml file and automates the entire release process.

Trigger
The workflow is automatically triggered whenever a new Git tag starting with v is pushed to the repository. This is a standard practice for creating versioned releases.

Example of how to trigger a new deployment:

# After committing your changes, create a new version tag
git tag v1.0.1

# Push the tag to your GitHub repository
git push origin v1.0.1
