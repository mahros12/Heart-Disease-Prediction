# ❤️ Heart Disease Prediction Using Machine Learning

A complete Machine Learning project for predicting heart disease using medical and lifestyle patient data.  
This project demonstrates a full ML workflow including data preprocessing, feature engineering, model training, hyperparameter tuning, visualization, and performance evaluation.

---

# 📌 Project Overview

Heart disease is one of the leading causes of death worldwide.  
The goal of this project is to build and evaluate multiple Machine Learning models capable of predicting whether a patient is likely to have heart disease based on medical attributes.

The project focuses not only on model accuracy, but also on:
- Proper data preprocessing
- Feature engineering
- Model comparison
- Hyperparameter tuning
- Visualization and analysis

---

# 🧠 Machine Learning Workflow

## 1️⃣ Data Preprocessing
The dataset was cleaned and prepared using several preprocessing techniques:

- Removing unnecessary columns
- Handling missing values
- Removing duplicates
- Detecting and removing outliers using IQR
- Log transformation for skewed features
- One-Hot Encoding categorical variables
- Feature scaling using StandardScaler

---

## 2️⃣ Feature Engineering
Several custom features were engineered to improve model performance:

| Feature | Description |
|---|---|
| `HR_diff` | Difference between expected and actual heart rate |
| `Age_group` | Categorized patient age ranges |
| `High_Risk` | High BP + High Cholesterol indicator |
| `stress_ratio` | Stress measurement using ECG and MaxHR |

---

# 📊 Models Used

The following classification models were trained and evaluated:

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

---

# ⚙️ Hyperparameter Tuning

Different hyperparameters were tested to optimize performance:

| Model | Hyperparameters |
|---|---|
| Logistic Regression | `C` |
| Decision Tree | `max_depth` |
| KNN | `n_neighbors` |
| SVM | `C` |

---

# 🏆 Best Model Performance

## ✅ Decision Tree Classifier

The Decision Tree model achieved the best overall performance:

| Metric | Value |
|---|---|
| Accuracy | **87.5%** |

The model balanced complexity and generalization effectively while minimizing False Positives and False Negatives.

---

# 📈 Visualizations Included

This project includes multiple visualizations such as:

- Correlation Heatmaps
- Feature Importance Analysis
- Confusion Matrices
- Hyperparameter Tuning Graphs
- Decision Tree Visualization

---

# 🛠️ Technologies Used

## Programming Language
- Python

## Libraries
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

# 📂 Project Structure

```bash
Heart-Disease-Prediction/
│
├── data/
│   └── heart.csv
│
├── images/
│   ├── heatmap.png
│   ├── confusion_matrix_lr.png
│   ├── decision_tree.png
│   └── ...
│
├── report/
│   └── Heart Disease Prediction.pdf
│
├── src/
│   └── heart_disease_prediction.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Heart-Disease-Prediction.git
```

Navigate to the project directory:

```bash
cd Heart-Disease-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python src/heart_disease_prediction.py
```

---

# 📷 Sample Results

## Logistic Regression
- Accuracy: 82.5%

## Decision Tree
- Accuracy: 87.5%

## KNN
- Accuracy: 85%

## SVM
- Accuracy: 77.5%

---

# 📖 Key Learning Outcomes

Through this project, I gained practical experience in:

- Data preprocessing techniques
- Feature engineering
- Classification algorithms
- Model evaluation
- Hyperparameter tuning
- Data visualization
- Healthcare data analysis

---

# 🎯 Future Improvements

Potential future improvements include:

- Cross-validation
- Ensemble models
- XGBoost / Random Forest
- Deep Learning approaches
- Deployment using Flask or Streamlit
- Real-time prediction web application

---

## 👨‍💻 Author

Ahmed Mahros

Faculty of Computers and Information
Ain Shams University


