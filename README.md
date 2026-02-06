# Data Mining - WM9QG Individual Assignment

## Diabetes Risk Prediction and Health Insights

This project applies data mining and machine learning techniques to the CDC Diabetes Health Indicators Dataset to:
1. **Classify** individuals into diabetic/non-diabetic groups using supervised learning
2. **Cluster** the population into meaningful health risk segments using unsupervised learning

## Dataset
- **Source**: CDC BRFSS 2015 Survey (253,680 responses, 22 features)
- **Target**: `Diabetes_012` (0=No diabetes, 1=Prediabetes, 2=Diabetes)

## Project Structure
```
├── data/                  # Dataset files
├── notebooks/
│   ├── 01_EDA.ipynb              # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb    # Data Preprocessing
│   ├── 03_Clustering.ipynb       # Clustering Analysis
│   └── 04_Classification.ipynb   # Classification Models
├── requirements.txt       # Python dependencies
└── README.md
```

## Setup
```bash
pip install -r requirements.txt
```

## Author
WM9QG-15 Fundamentals of AI and Data Mining - Individual Assessment 2025/26
