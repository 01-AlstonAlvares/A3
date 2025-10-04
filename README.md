# 🚗 A3 Car Price Prediction – MLOps Pipeline

A complete **MLOps project** demonstrating the training, tracking, and deployment of a machine learning model for predicting the **price category of used cars**.  
This repository implements a **fully automated CI/CD pipeline** powered by **GitHub Actions** and **MLflow**.

---

## 📖 Project Overview

The project centers on a **Multinomial Logistic Regression model**, built **from scratch in Python**, to classify used cars into one of four price brackets.  
The workflow covers the **entire machine learning lifecycle** — from **data preprocessing** and **feature engineering** to **hyperparameter tuning**, **evaluation**, and **deployment**.

All experiments, metrics, and artifacts are tracked using a **remote MLflow server**, ensuring complete reproducibility.

### 🔧 Core Objectives
- Build a **custom Logistic Regression model** with L2 regularization.  
- Automate **training, testing, and deployment** through GitHub Actions.  
- Deploy the trained model as a **REST API** using Flask and Docker.  
- Track experiments and model versions via **MLflow**.

---

## ⚙️ CI/CD Pipeline Overview

The CI/CD pipeline automates the **end-to-end machine learning workflow**, including continuous testing, model validation, and production deployment.

### 🧩 Continuous Integration (CI)
- Automatically runs **unit tests** against the “Staging” model on each version update.
- Ensures that only **validated models** proceed to deployment.

### 🚀 Continuous Deployment (CD)
- Builds a **Docker image** of the Flask-based model API.
- Deploys the containerized model to a **remote server** for production use.

### 🏷️ Triggering a Deployment
To create a new deployment, push a **Git tag** to the repository:

```bash
# After committing your changes, create a new version tag
git tag v1.0.1

# Push the tag to trigger CI/CD pipeline
git push origin v1.0.1
```

---

## 📂 Project Structure

| File / Directory | Description |
|------------------|-------------|
| `.github/workflows/main.yml` | Defines the GitHub Actions CI/CD workflow. |
| `app/` | Contains the Flask app and automated test scripts. |
| ├── `app.py` | Flask application that loads the model from MLflow and serves predictions via REST API. |
| ├── `test_model.py` | Unit tests to validate the “Staging” model from MLflow. |
| └── `requirements.txt` | Dependencies required for running the app and tests. |
| `A3_Classification_Final_Improved.ipynb` | Main Jupyter Notebook for preprocessing, training, and tracking experiments. |
| `custom_classifier.py` | Custom implementation of the Logistic Regression classifier. |
| `Dockerfile` | Instructions to build the containerized model-serving application. |

---

## 🧪 Model & Experiment Tracking

### 🧠 Model
- **Type:** Multinomial Logistic Regression (from scratch)
- **Regularization:** L2 (Ridge) to reduce overfitting
- **Target:** Price category classification

### 🔍 Feature Engineering
- **`car_age`** – Derived from the manufacturing year  
- **`km_per_year`** – Represents usage intensity  

These features significantly improve model accuracy by capturing meaningful relationships.

### 📊 Experiment Tracking
- **Platform:** MLflow  
- **Tracked Items:** Parameters, metrics, models, and artifacts  
- **Benefit:** Enables full reproducibility and model comparison  

MLflow UI: [http://mlflow.ml.brain.cs.ait.ac.th/](http://mlflow.ml.brain.cs.ait.ac.th/)

---

## 🐳 Deployment

The deployment uses **Docker** for consistent, isolated environments:

```bash
# Build Docker image
docker build -t car-price-predictor .

# Run container locally
docker run -p 5000:5000 car-price-predictor
```

Once deployed, the Flask API can serve real-time car price predictions via simple HTTP requests.

---

## 🧠 Key Highlights

- ✅ Custom-built ML model (no scikit-learn dependency)
- ✅ Automated model validation and CI/CD pipeline
- ✅ Full experiment tracking via MLflow
- ✅ Scalable deployment using Docker and Flask

---

## 👥 Contributors

**Author:** Alston Alvares


**Student ID:** st126488 


**Institution:** Asian Institute of Technology (AIT)  


**Department:** Data Science and Artificial Intelligence  
