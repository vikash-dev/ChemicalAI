#  Household Ingredients AI Safety Checker

An end-to-end **Machine Learning application** designed to predict the **toxicity and safety impact of ingridents of household products** using their molecular structures.

This project demonstrates how raw chemical data can be transformed into meaningful predictions through a complete ML pipeline — from data preprocessing to model deployment with a user-friendly web interface(temp). I need this model to integrate it with one of my Ingredients scaneer app 😺

---

##  Project Architecture Overview

The application is built in **two major phases**:

### 1️⃣ Model Training (Building the “Brain”)
This phase focuses on preparing chemical data, extracting molecular features, and training a machine learning model capable of predicting toxicity.

### 2️⃣ Application Deployment (Building the “Interface”)
This phase exposes the trained model through a simple web-based interface where users can input chemical names and instantly receive predictions.

---

##  End-to-End ML Pipeline

1. Chemical Input (Name / SMILES string)
2. Data Preprocessing
3. Feature Engineering (Molecular Descriptors)
4. Model Training (Random Forest)
5. Model Serialization
6. Web Application for Prediction

---

##  Glossary of Key Components & Tools

| Component | Description | Tools Used |
|--------|------------|-----------|
| **Virtual Environment (Venv)** | Isolated Python environment to manage dependencies and avoid version conflicts | Python `venv` |
| **Data Preprocessing** | Cleaning, organizing, and preparing raw chemical datasets | Pandas, NumPy |
| **Feature Engineering** | Converts chemical structures into numerical features the model can understand | RDKit, PubChemPy |
| **AI Model** | Learns patterns from molecular features to predict toxicity | Scikit-learn (Random Forest) |
| **Serialization** | Saves the trained model to disk for fast reuse | joblib |
| **User Interface (UI)** | Web-based interface for chemical input and predictions | Streamlit |

---

##  Key Features

- End-to-end machine learning pipeline
- Real-world chemical safety use case
- Molecular feature extraction using RDKit
- Fast and lightweight deployment with Streamlit
- Easily extensible to additional chemical properties

---

##  Tech Stack

- **Python**
- **Pandas, NumPy**
- **RDKit, PubChemPy**
- **Scikit-learn**
- **Streamlit**

---

##  Future Improvements

- Support for multiple toxicity categories
- Integration with larger chemical databases
- Deep learning models for improved accuracy
- Cloud deployment (Docker / AWS / Azure)
- User authentication and prediction history
