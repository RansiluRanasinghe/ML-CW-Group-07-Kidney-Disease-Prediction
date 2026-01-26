# 🩺 Chronic Kidney Disease Classification using Supervised Machine Learning

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![License](https://img.shields.io/badge/License-Educational-blue?style=flat-square)

Chronic Kidney Disease (CKD) is a long-term medical condition that can progress silently and lead to kidney failure if not detected early. This project focuses on building a **supervised machine learning classification model** to predict whether a patient has CKD or not, based on clinical and laboratory measurements.

The project emphasizes:
- Proper data cleaning
- Feature preprocessing
- Dimensionality reduction using PCA
- Model training and evaluation
- Generalization and reliability

The goal is not just high accuracy, but a **clean, explainable, and academically sound ML pipeline** suitable for coursework and real-world understanding.

---

## 🎯 Problem Statement

Given a dataset containing medical attributes such as blood glucose level, red blood cell count, white blood cell count, blood pressure, and other clinical indicators, the task is to:

> **Predict whether a patient is suffering from Chronic Kidney Disease (CKD) or not.**

### Binary Classification Problem

| Class | Description |
|-------|-------------|
| `ckd` | Patient has Chronic Kidney Disease |
| `notckd` | Patient does not have Chronic Kidney Disease |

---

## 📊 Dataset Description

| Property | Details |
|----------|---------|
| **Source** | UCI Machine Learning Repository |
| **Dataset Name** | Chronic Kidney Disease Dataset |
| **Number of Rows** | 400 |
| **Number of Features** | 25 (excluding ID) |
| **Target Variable** | classification |

### Feature Types

**Numerical Features:**
- Age, Blood Pressure, Blood Glucose
- White Blood Cell Count, Red Blood Cell Count
- Serum Creatinine, Blood Urea, etc.

**Categorical Features:**
- Hypertension, Diabetes, Appetite
- Anemia, Pedal Edema, etc.

### Data Quality Challenges

- ❌ **Missing values** (NaN)
- ❌ **Numeric values stored as strings**
- ❌ **Mixed categorical encodings**
- ❌ **Medical data inconsistency** (real-world noise)

These challenges make the dataset ideal for demonstrating **realistic data preprocessing**.

---

## 🧠 Machine Learning Workflow

This project follows a **structured supervised learning workflow**:

### 🔹 Stage 1: Data Understanding & Exploration

**What is done:**
- Inspect dataset shape and feature types
- Identify numerical vs categorical variables
- Analyze target class distribution
- Identify missing values across features

**Why it is important:**
Understanding the dataset prevents incorrect assumptions, helps detect imbalance, and informs preprocessing decisions.

---

### 🔹 Stage 2: Data Cleaning

**What is done:**
- Remove rows containing any missing values
- Convert numerical features stored as text into floating-point values
- Drop irrelevant columns such as IDs

**Why it is important:**
- Machine learning models cannot handle missing or improperly typed values
- Ensures clean input for PCA and classifiers
- Follows academic requirements for data preprocessing

---

### 🔹 Stage 3: Feature Selection for PCA Visualization

**Selected Features:**
1. **Blood Glucose Random (bgr)**
2. **Red Blood Cell Count (rc)**
3. **White Blood Cell Count (wc)**

**Why it is important:**
- Allows intuitive visualization of the dataset
- Demonstrates how variance differs across features
- Prepares data for dimensionality reduction analysis

---

### 🔹 Stage 4: Principal Component Analysis (PCA)

**Two Approaches:**

1. **PCA Without Scaling**
   - Applied directly to raw numerical values

2. **PCA With Scaling**
   - Features are standardized before PCA

**Why both are performed:**
- PCA is sensitive to feature scale
- Medical features have different units and ranges
- Comparing both approaches demonstrates **why feature scaling is critical**

---

### 🔹 Stage 5: Feature Encoding & Scaling

**What is done:**
- Convert categorical variables into numerical representations
- Apply feature scaling to numerical values

**Why it is important:**
- Most ML models require numeric input
- Scaling ensures fair contribution of all features
- Prevents dominance of high-magnitude variables

---

### 🔹 Stage 6: Train-Test Split

**What is done:**
- Dataset is split into training and testing sets

**Why it is important:**
- Ensures unbiased model evaluation
- Tests model generalization on unseen data
- Prevents overfitting

---

### 🔹 Stage 7: Model Training

**Model Choice:**
A supervised classification model is trained (e.g., **Logistic Regression**, **Random Forest**, or **Support Vector Machine**).

**Why this model:**
- Suitable for tabular medical data
- Interpretable and well-established
- Works effectively with limited dataset size

---

### 🔹 Stage 8: Model Evaluation

**Evaluation Metrics Used:**
- ✅ **Accuracy**
- ✅ **Precision**
- ✅ **Recall**
- ✅ **F1-score**
- ✅ **Confusion Matrix**

**Why these metrics:**
- Accuracy alone is insufficient in medical diagnosis
- **Recall is critical** to avoid missing CKD cases (false negatives)
- F1-score balances false positives and false negatives

---

### 🔹 Stage 9: Generalization & Reliability Check

**What is done:**
- Compare training and testing performance
- Ensure no severe overfitting
- Validate robustness of predictions

**Why it matters:**
Medical prediction systems must **generalize well to unseen patients**.

---

## ⏱️ Project Timeline

| Day | Focus |
|-----|-------|
| **Day 1** | Data cleaning, preprocessing, PCA, feature engineering |
| **Day 2** | Model training, evaluation, tuning, and documentation |

---

## 🛠️ Technologies Used

### Machine Learning
- **Python** — Programming language
- **scikit-learn** — Model training, PCA, evaluation
- **NumPy** — Numerical computation
- **Pandas** — Data manipulation

### Visualization
- **Matplotlib** — Plotting and visualization
- **Seaborn** — Statistical graphics

### Development
- **Jupyter Notebook** — Interactive development environment

---

## 🚀 How to Run

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/RansiluRanasinghe/CKD-Classification-ML.git
cd CKD-Classification-ML

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
```

### Run the Notebook
Open and run `ckd_classification.ipynb` sequentially.

---

## ✅ Final Outcome

This project delivers:

- ✅ **Clean and reproducible** ML pipeline
- ✅ **Proper use of PCA** with and without scaling
- ✅ **Reliable supervised classification** model
- ✅ **Coursework-compliant** methodology
- ✅ **GitHub-ready** documentation

---

## 📚 Learning Outcomes

This project demonstrates:

✔ **Practical understanding of PCA**  
✔ **Importance of scaling** in dimensionality reduction  
✔ **Real-world data cleaning** challenges  
✔ **Supervised model evaluation** in medical contexts  
✔ **End-to-end ML workflow** design

### Skills Demonstrated
- Medical data preprocessing
- Principal Component Analysis (PCA)
- Feature engineering and encoding
- Binary classification
- Model evaluation metrics
- Healthcare ML thinking

---

## 👨‍🎓 Academic Relevance

This project satisfies typical **Machine Learning coursework requirements** by including:

- ✅ Clear problem definition
- ✅ Real-world medical dataset
- ✅ Proper preprocessing pipeline
- ✅ Dimensionality reduction (PCA)
- ✅ Model training using standard libraries
- ✅ Comprehensive evaluation metrics
- ✅ Explainable methodology

---

## 🎯 Use Cases

This approach can be adapted for:
- **Healthcare** — Early disease detection systems
- **Clinical Decision Support** — Diagnostic assistance
- **Medical Research** — Pattern discovery in patient data
- **Public Health** — Population health screening
- **Telemedicine** — Remote patient monitoring

---

## 📎 Dataset Reference

**Chronic Kidney Disease Dataset**  
[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease)

### Citation
If using this dataset, please cite:
> Rubini, L. Jerlin (2015). Chronic Kidney Disease. UCI Machine Learning Repository.

---

## 📜 License

This project is intended for **educational and research purposes**.

---

## 🙏 Acknowledgements

Special thanks to:
- **UCI Machine Learning Repository** — Dataset hosting
- **L. Jerlin Rubini** — Dataset contributor
- **Apollo Hospitals, Karaikudi** — Data collection

---

## 🤝 Connect

**Ransilu Ranasinghe**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ransilu-ranasinghe-a596792ba)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/RansiluRanasinghe)
[![Email](https://img.shields.io/badge/Email-EA4335?style=flat-square&logo=gmail&logoColor=white)](mailto:dinisthar@gmail.com)

**Interests:**  
Machine Learning • Healthcare AI • Data Preprocessing • Dimensionality Reduction

---

<div align="center">

**⭐ If you find this project useful, consider giving it a star!**

**Built for academic excellence and real-world understanding.**

</div>
