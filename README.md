# Data Mining - WM9QG Individual Assignment

## Diabetes Risk Prediction and Health Insights

This project applies data mining and machine learning techniques to the **CDC Diabetes Health Indicators Dataset (BRFSS 2015)** to:
1. **Explore** the data through comprehensive EDA (distributions, correlations, statistical tests)
2. **Preprocess** the data with binarisation, scaling, and SMOTE for class imbalance
3. **Cluster** the population into meaningful health risk segments (K-Means, DBSCAN)
4. **Classify** individuals as diabetic/non-diabetic using supervised learning (Logistic Regression, Random Forest)

---

## Dataset

| Property | Detail |
|----------|--------|
| **Source** | CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015 |
| **Samples** | 253,680 survey responses |
| **Features** | 21 health/lifestyle indicators (14 binary, 4 ordinal, 3 continuous) |
| **Target** | `Diabetes_012` → binarised to 0 (No Diabetes, 84.2%) / 1 (Diabetes/Prediabetes, 15.8%) |

**Key features:** HighBP, HighChol, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income

---

## Project Structure

```
├── data/
│   ├── CDC Diabetes Dataset.csv          # Raw dataset
│   ├── Dataset Description.txt           # Feature descriptions
│   └── processed/                        # Scaled & SMOTE-balanced data
│       ├── X_train_smote.csv
│       ├── y_train_smote.csv
│       ├── X_test_scaled.csv
│       ├── y_test.csv
│       ├── X_full_scaled.csv
│       └── y_full.csv
├── notebooks/
│   ├── 01_EDA.ipynb                      # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb            # Data Preprocessing Pipeline
│   ├── 03_Clustering.ipynb               # Unsupervised Clustering
│   ├── 04_Classification.ipynb           # Supervised Classification
│   └── figures/                          # Saved plots (PNG)
├── requirements.txt                      # Python dependencies
├── .gitignore
└── README.md
```

---

## Notebooks Overview

### 01 — Exploratory Data Analysis
- Data quality check (no missing values, 23,899 duplicates retained)
- Target distribution analysis (severe class imbalance: 84%/2%/14%)
- Feature distributions (binary, ordinal, continuous)
- Correlation heatmap
- Feature vs target analysis (box plots, bar charts)
- Outlier detection (IQR method)
- Mutual Information ranking
- PCA 2D projection & variance analysis (17 components for 90%)
- Chi-Square & ANOVA statistical significance tests

### 02 — Preprocessing
- Target binarisation: merged prediabetes + diabetes → class 1
- Stratified train/test split (80/20): 202,944 / 50,736
- StandardScaler (fit on train only to prevent data leakage)
- SMOTE on training set: 170,962 vs 170,962 (balanced)

### 03 — Clustering
- **K-Means** (k=3 via elbow method + silhouette analysis):
  - Cluster 0 — Healthy/Low-risk (70%, diabetes rate 9.3%)
  - Cluster 1 — High-risk/Chronic (25%, diabetes rate 34.2%)
  - Cluster 2 — Uninsured/At-risk (5%, diabetes rate 13.3%)
  - Silhouette blade plot, profile heatmap, diabetes rate comparison
- **DBSCAN** (eps=5.0, min_samples=30): 3 clusters, silhouette 0.258
- Algorithm comparison and interpretation

### 04 — Classification
- **Logistic Regression**: Accuracy 73.1%, Recall **75.8%**, AUC 0.817
- **Random Forest** (200 trees): Accuracy **80.7%**, F1 **48.2%**, AUC 0.817
- Confusion matrices, ROC curves, Precision-Recall curves
- Feature importance comparison — top predictors: GenHlth, BMI, Age, HighBP, HighChol

---

## Key Results

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|-------|----------|-----------|--------|-----|---------|
| Logistic Regression | 0.731 | 0.341 | **0.758** | 0.470 | 0.817 |
| Random Forest | **0.807** | **0.418** | 0.569 | **0.482** | **0.817** |

- Both models achieve **AUC ~0.82**, indicating good discriminative ability
- Logistic Regression has higher recall (better for screening — fewer missed cases)
- Random Forest has higher accuracy and precision (more balanced predictions)

---

## Setup

```bash
pip install -r requirements.txt
```

**Python version:** 3.10+  
**Dependencies:** pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn, scipy

---

## How to Run

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run notebooks in order: `01_EDA` → `02_Preprocessing` → `03_Clustering` → `04_Classification`
4. Notebook 02 generates `data/processed/` files used by notebooks 03 and 04

---

## Author

WM9QG-15 Fundamentals of AI and Data Mining — Individual Assessment 2025/26
