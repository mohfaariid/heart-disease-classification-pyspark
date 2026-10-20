# Heart Disease Classification with PySpark MLlib

An end-to-end, reproducible machine learning pipeline engineered with Apache Spark MLlib to classify coronary heart disease using the UCI Heart Disease dataset.

---

## 📌 Project Overview

This project implements scalable data exploration, preprocessing pipelines, multi-model benchmarking, 5-fold cross-validation hyperparameter tuning, and feature importance interpretation.

### Key Highlights:
- **Distributed Data Processing**: Ingested and transformed tabular clinical data using PySpark DataFrames and SQL functions.
- **Exploratory Data Analysis (EDA)**: Target distribution analysis, correlation heatmap, clinical attribute distributions by disease status, and chest pain/thalassemia risk profiles.
- **Robust Feature Pipeline**: `VectorAssembler` and `StandardScaler` chained with MLlib estimators into unified `Pipeline` architectures.
- **Multi-Model Benchmark**:
  - Logistic Regression (Baseline)
  - Decision Tree Classifier
  - Random Forest Classifier (Bagging Ensemble)
  - Gradient-Boosted Trees (Boosting Ensemble)
- **Comprehensive Evaluation**: Multi-metric comparison covering ROC-AUC, Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.
- **Hyperparameter Optimization**: 5-Fold Cross-Validation (`CrossValidator` & `ParamGridBuilder`) to tune tree depth and ensemble size.
- **Model Explainability**: Feature importance extraction identifying top predictive clinical markers.

---

## 🗂️ Project Structure

```text
heart-disease-classification-pyspark/
├── data/
│   └── heart.csv                            # Clinical dataset (1,025 records, 14 attributes)
├── heart_disease_classification.ipynb       # Interactive, step-by-step notebook
├── heart.py                                 # Standalone PySpark pipeline script
├── requirements.txt                         # Python dependencies
└── README.md                                # Project documentation
```

---

## 📊 Dataset Schema

| Attribute | Description | Values / Range |
|---|---|---|
| `age` | Age in years | 29 – 77 |
| `sex` | Biological sex | 1 = Male, 0 = Female |
| `cp` | Chest pain type | 0: Typical angina, 1: Atypical angina, 2: Non-anginal, 3: Asymptomatic |
| `trestbps` | Resting blood pressure | mm Hg on admission |
| `chol` | Serum cholesterol | mg/dl |
| `fbs` | Fasting blood sugar > 120 mg/dl | 1 = True, 0 = False |
| `restecg` | Resting ECG results | 0: Normal, 1: ST-T wave abnormality, 2: LV hypertrophy |
| `thalach` | Maximum heart rate achieved | 71 – 202 bpm |
| `exang` | Exercise induced angina | 1 = Yes, 0 = No |
| `oldpeak` | ST depression induced by exercise | Numeric |
| `slope` | Slope of the peak exercise ST segment | 0, 1, 2 |
| `ca` | Number of major vessels colored by fluoroscopy | 0 – 3 |
| `thal` | Thalassemia status | 0: Null, 1: Normal, 2: Fixed defect, 3: Reversible defect |
| `target` | Presence of heart disease | **0 = No Disease, 1 = Disease** |

---

## ⚙️ Installation & Usage

### Prerequisites
- Python 3.10+
- Java 8 or Java 11 (required by Apache Spark)
- Apache Spark 3.4+

### Setup
```bash
# Clone the repository
git clone https://github.com/mohfaariid/heart-disease-classification-pyspark.git
cd heart-disease-classification-pyspark

# Install dependencies
pip install -r requirements.txt
```

### Run Standalone Script
```bash
spark-submit heart.py
# or
python heart.py
```

### Run Jupyter Notebook
```bash
jupyter notebook heart_disease_classification.ipynb
```

---

## 🏆 Benchmark & Results

| Model Architecture | ROC-AUC | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| **Logistic Regression** | ~0.893 | ~0.825 | ~0.825 | ~0.825 | ~0.825 |
| **Decision Tree** | ~0.865 | ~0.835 | ~0.837 | ~0.835 | ~0.835 |
| **Random Forest** | **~0.970+** | **~0.940+** | **~0.941+** | **~0.940+** | **~0.940+** |
| **Gradient-Boosted Trees (GBT)** | ~0.965+ | ~0.935+ | ~0.936+ | ~0.935+ | ~0.935+ |
| **Tuned Random Forest (5-Fold CV)** | **Top Generalization** | — | — | — | — |

*Top Predictor Attributes Identified:* Chest pain type (`cp`), maximum heart rate (`thalach`), ST depression (`oldpeak`), and number of major vessels (`ca`).
