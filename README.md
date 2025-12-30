# MLOps

# Course Completion Prediction – MLOps Project

This project demonstrates how a machine learning model can be transformed from an experimental training script into a **production-ready system** using **MLOps Level 2 (Automated CI/CD Pipeline)** principles.

---

## Project Overview

The objective of this project is to predict whether a student will complete an online course using machine learning models.  
Beyond model training, the project focuses on **automation, reproducibility, and deployment readiness**, which are core responsibilities of an **MLOps Engineer**.

The system includes:
- Automated model training
- Model versioning and governance
- Containerized serving
- CI/CD-based validation

---

## MLOps Architecture (Level 2)

The project is designed according to **MLOps Level 2**, where the entire pipeline is automatically triggered by code changes without manual intervention.

### Technologies Used

- **Version Control:** Git & GitHub  
- **Model Training:** scikit-learn, XGBoost  
- **Experiment Tracking & Registry:** MLflow  
- **Serving:** FastAPI (REST API)  
- **Containerization:** Docker  
- **CI/CD Automation:** GitHub Actions  

---

## Model Lifecycle Management

- Models are trained using an existing ML Engineer pipeline.
- During training, all experiments, parameters, and metrics are logged to **MLflow**.
- Trained models are automatically registered in the **MLflow Model Registry**.
- Static artifacts such as `model.pkl` are intentionally excluded from the repository.
- The Model Registry acts as the **single source of truth** for all model versions.

---

## Serving Strategy

- The FastAPI service dynamically loads the production model from MLflow:
- The API follows a **stateless serving pattern**.
- Training–serving skew caused by feature hashing is resolved using **dynamic feature padding** at the API layer.
- A fallback `DummyClassifier` is implemented to ensure service availability in case of model loading or inference failures.

---

## CI/CD Pipeline

A fully automated CI/CD pipeline is implemented using **GitHub Actions**.

### Pipeline Behavior

- Triggered on every push to the `main` branch
- Automatically:
- Installs dependencies
- Runs the training pipeline
- Logs models to MLflow Model Registry
- Performs syntax and integrity checks
- Validates Docker build for serving readiness

This setup ensures that the system remains **reproducible, testable, and production-ready** at all times.

---

## MLOps Maturity Level

This project fulfills the requirements of **MLOps Level 2 (CI/CD Pipeline Automation)**:

- No manual intervention after code changes
- Automated training and model registration
- Registry-based model serving
- Dockerized and environment-agnostic execution

---

## Role Separation

- **ML Engineer:** Model architecture, feature engineering, and training logic  
- **MLOps Engineer:** Experiment tracking, model registry integration, CI/CD automation, serving architecture, and system robustness

---

## Conclusion

This project demonstrates how machine learning systems can move beyond notebooks and become **reliable, automated, and maintainable products** through proper MLOps practices.
