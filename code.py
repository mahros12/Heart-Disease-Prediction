import pandas as pd
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# =========================
# FUNCTIONS
# =========================

def remove_outliers_iqr(df, cols):

    for i in cols:

        Q1 = df[i].quantile(0.25)
        Q3 = df[i].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df = df[
            (df[i] >= lower) &
            (df[i] <= upper)
        ]

    return df


def show_null_columns(df):

    null_counts = df.isnull().sum()

    null_cols = null_counts[null_counts > 0]

    print(null_cols)

    return null_cols


# =========================
# LOAD DATA
# =========================

train = pd.read_csv('train_data.csv')
test = pd.read_csv('test_data.csv')

train.info()

train = train.drop('id', axis=1)


# =========================
# DUPLICATES
# =========================

print(train.duplicated().sum())

train = train.drop_duplicates()

print(train.duplicated().sum())


# =========================
# MISSING VALUES VISUALIZATION
# =========================

sns.heatmap(train.isnull(), cbar=False)

plt.title("Missing Values")

plt.show()

print(train.isnull().sum())


# =========================
# NUMERICAL COLUMNS
# =========================

num_cols = [
    'Age',
    'BP',
    'Cholesterol',
    'Max HR',
    'ST depression'
]

for col in num_cols:

    sns.histplot(train[col], kde=True)

    plt.title(f"{col} Distribution")

    plt.show()


# =========================
# OUTLIER REMOVAL
# =========================

sns.boxplot(data=train[num_cols])

plt.title('Boxplot of Numerical Columns')

plt.show()

train = remove_outliers_iqr(train, num_cols)

sns.boxplot(data=train[num_cols])

plt.title('Boxplot after Removing Outliers')

plt.show()


# =========================
# HANDLE MISSING VALUES
# =========================

show_null_columns(train)

train['Age'] = train['Age'].fillna(
    train['Age'].mean()
)

categorical = [
    'Gender',
    'smoking_status',
    'work_type'
]

for i in categorical:

    train[i] = train[i].fillna(
        train[i].mode()[0]
    )

for i in categorical:

    sns.countplot(x=i, data=train)

    plt.title(f"{i} Distribution")

    plt.show()

print(train.isnull().any())


# =========================
# LOG TRANSFORMATION
# =========================

train['ST_log'] = np.log1p(
    train['ST depression']
)

train = train.drop('ST depression', axis=1)

sns.histplot(train['ST_log'], kde=True)

plt.title("ST_log Distribution")

plt.show()

num_cols.remove('ST depression')

num_cols.append('ST_log')


# =========================
# CORRELATION MATRIX
# =========================

corr = train[num_cols].corr()

sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Matrix")

plt.show()


# =========================
# FEATURE ENGINEERING
# =========================

train['Age_group'] = pd.cut(
    train['Age'],
    bins=[0, 25, 50, 70, 100],
    labels=[0, 1, 2, 3]
)

train['HR_diff'] = (
    (220 - train['Age']) -
    train['Max HR']
)

train['High_Risk'] = (
    (train['BP'] > 140) &
    (train['Cholesterol'] > 240)
).astype(int)

train['stress_ratio'] = (
    train['ST_log'] /
    np.maximum(train['Max HR'], 1)
)


# =========================
# ENCODING
# =========================

train = pd.get_dummies(
    train,
    drop_first=True
)

bool_cols = train.select_dtypes(
    include='bool'
).columns

train[bool_cols] = train[bool_cols].astype(int)


# =========================
# FEATURE CORRELATION
# =========================

corr_target = train.corr()['Heart Disease_Yes'].sort_values(
    ascending=False
)

print(corr_target)

corr_target.drop('Heart Disease_Yes').sort_values().plot(
    kind='barh',
    figsize=(10, 8)
)

plt.title('Feature Correlation with Heart Disease')

plt.xlabel('Correlation Value')

plt.show()

train.info()

print(train.head())


# =========================
# SPLIT FEATURES & TARGET
# =========================

X = train.drop('Heart Disease_Yes', axis=1)

y = train['Heart Disease_Yes']


# =========================
# TRAIN VALID SPLIT
# =========================

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("X_train shape:", X_train.shape)
print("X_valid shape:", X_valid.shape)


# =========================
# FEATURE SCALING
# =========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_valid_scaled = scaler.transform(X_valid)


# =========================
# LOGISTIC REGRESSION
# =========================

lr_model = LogisticRegression(
    max_iter=3000,
    C=0.5,
    solver='liblinear'
)

lr_model.fit(X_train_scaled, y_train)

lr_pred = lr_model.predict(X_valid_scaled)

print("\n===== Logistic Regression =====")

print(
    "Accuracy:",
    accuracy_score(y_valid, lr_pred)
)

print("\nClassification Report:\n")

print(classification_report(y_valid, lr_pred))

cm = confusion_matrix(y_valid, lr_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Logistic Regression Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()


# =========================
# DECISION TREE
# =========================

dt_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42
)

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_valid)

print("\n===== Decision Tree =====")

print(
    "Accuracy:",
    accuracy_score(y_valid, dt_pred)
)

print("\nClassification Report:\n")

print(classification_report(y_valid, dt_pred))

cm = confusion_matrix(y_valid, dt_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Greens'
)

plt.title("Decision Tree Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()


# =========================
# KNN
# =========================

knn_model = KNeighborsClassifier(
    n_neighbors=3,
    weights='distance',
    metric='minkowski'
)

knn_model.fit(X_train_scaled, y_train)

knn_pred = knn_model.predict(X_valid_scaled)

print("\n===== KNN =====")

print(
    "Accuracy:",
    accuracy_score(y_valid, knn_pred)
)

print("\nClassification Report:\n")

print(classification_report(y_valid, knn_pred))

cm = confusion_matrix(y_valid, knn_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Oranges'
)

plt.title("KNN Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()


# =========================
# SVM
# =========================

svm_model = SVC(
    kernel='rbf',
    C=10,
    gamma='scale'
)

svm_model.fit(X_train_scaled, y_train)

svm_pred = svm_model.predict(X_valid_scaled)

print("\n===== SVM =====")

print(
    "Accuracy:",
    accuracy_score(y_valid, svm_pred)
)

print("\nClassification Report:\n")

print(classification_report(y_valid, svm_pred))

cm = confusion_matrix(y_valid, svm_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Purples'
)

plt.title("SVM Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()


# =========================
# HYPERPARAMETER EXPERIMENTS
# =========================

print("\n============================")
print("HYPERPARAMETER EXPERIMENTS")
print("============================")


# =========================
# LOGISTIC REGRESSION C TEST
# =========================

print("\nLogistic Regression C Experiments\n")

lr_scores = []

for c_value in [0.01, 0.1, 0.5, 1, 10]:

    temp_lr = LogisticRegression(
        max_iter=3000,
        C=c_value,
        solver='liblinear'
    )

    temp_lr.fit(X_train_scaled, y_train)

    temp_pred = temp_lr.predict(X_valid_scaled)

    acc = accuracy_score(y_valid, temp_pred)

    lr_scores.append(acc)

    print(f"C = {c_value} -> Accuracy = {acc:.4f}")

plt.figure(figsize=(8, 5))

plt.plot(
    [0.01, 0.1, 0.5, 1, 10],
    lr_scores,
    marker='o'
)

plt.title('Logistic Regression Accuracy vs C')

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.grid()

plt.show()

print("""
Explanation:
- Small C values apply stronger regularization.
- Medium C values balance learning and generalization.
- Very large C values may cause overfitting.
""")


# =========================
# DECISION TREE DEPTH TEST
# =========================

print("\nDecision Tree max_depth Experiments\n")

for depth in [2, 3, 4, 5, 7, 10]:

    temp_dt = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    temp_dt.fit(X_train, y_train)

    temp_pred = temp_dt.predict(X_valid)

    acc = accuracy_score(y_valid, temp_pred)

    print(f"max_depth = {depth} -> Accuracy = {acc:.4f}")

plt.figure(figsize=(20, 10))

tree.plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=['No Disease', 'Disease'],
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("Decision Tree")

plt.show()

print("""
Explanation:
- Very small depth values underfit the data.
- max_depth = 3 achieved the best balance between complexity and generalization.
- Larger depth values caused overfitting and reduced validation accuracy.
""")


# =========================
# KNN NEIGHBORS TEST
# =========================

print("\nKNN n_neighbors Experiments\n")

knn_scores = []

for k in [1, 3, 5, 7, 9]:

    temp_knn = KNeighborsClassifier(
        n_neighbors=k
    )

    temp_knn.fit(X_train_scaled, y_train)

    temp_pred = temp_knn.predict(X_valid_scaled)

    acc = accuracy_score(y_valid, temp_pred)

    knn_scores.append(acc)

    print(f"n_neighbors = {k} -> Accuracy = {acc:.4f}")

plt.figure(figsize=(8, 5))

plt.plot(
    [1, 3, 5, 7, 9],
    knn_scores,
    marker='o'
)

plt.title('KNN Accuracy vs n_neighbors')

plt.xlabel('n_neighbors')

plt.ylabel('Accuracy')

plt.xticks([1, 3, 5, 7, 9])

plt.grid()

plt.show()

print("""
Explanation:
- Smaller K values make the model sensitive to noise.
- Medium K values balance generalization.
- Larger K values create smoother decision boundaries.
""")


# =========================
# SVM C VALUE TEST
# =========================

print("\nSVM C Experiments\n")

svm_scores = []

for c_value in [0.01, 0.1, 1, 10, 50]:

    temp_svm = SVC(
        kernel='rbf',
        C=c_value
    )

    temp_svm.fit(X_train_scaled, y_train)

    temp_pred = temp_svm.predict(X_valid_scaled)

    acc = accuracy_score(y_valid, temp_pred)

    svm_scores.append(acc)

    print(f"C = {c_value} -> Accuracy = {acc:.4f}")

plt.figure(figsize=(8, 5))

plt.plot(
    [0.01, 0.1, 1, 10, 50],
    svm_scores,
    marker='o'
)

plt.title('SVM Accuracy vs C')

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.grid()

plt.show()

print("""
Explanation:
- Small C values allow simpler decision boundaries.
- Larger C values try harder to classify all training samples correctly.
- Very large C values may cause overfitting.
""")


# =========================
# BEST MODEL SELECTION
# =========================

model_scores = {
    'Logistic Regression': accuracy_score(y_valid, lr_pred),
    'Decision Tree': accuracy_score(y_valid, dt_pred),
    'KNN': accuracy_score(y_valid, knn_pred),
    'SVM': accuracy_score(y_valid, svm_pred)
}

best_model_name = max(
    model_scores,
    key=model_scores.get
)

best_model_score = model_scores[best_model_name]

print("\n============================")
print("BEST MODEL")
print("============================")

print(f"Best Model: {best_model_name}")

print(f"Best Accuracy: {best_model_score:.4f}")

print("""
Explanation:
- Decision Tree achieved the highest validation accuracy.
- max_depth = 3 created the best balance between simplicity and generalization.
- The model avoided overfitting while maintaining strong prediction performance.
""")


# =========================
# MODEL ACCURACY COMPARISON
# =========================

model_names = [
    'Logistic Regression',
    'Decision Tree',
    'KNN',
    'SVM'
]

model_accuracies = [
    accuracy_score(y_valid, lr_pred),
    accuracy_score(y_valid, dt_pred),
    accuracy_score(y_valid, knn_pred),
    accuracy_score(y_valid, svm_pred)
]

plt.figure(figsize=(8, 5))

sns.barplot(
    x=model_names,
    y=model_accuracies
)

plt.title('Model Accuracy Comparison')

plt.ylabel('Accuracy')

plt.ylim(0, 1)

plt.show()


# =========================
# PREPROCESS TEST DATA
# =========================

test = test.drop('id', axis=1)

test['Age'] = test['Age'].fillna(
    train['Age'].mean()
)

for i in categorical:

    test[i] = test[i].fillna(
        test[i].mode()[0]
    )

test['ST_log'] = np.log1p(
    test['ST depression']
)

test = test.drop('ST depression', axis=1)

test['Age_group'] = pd.cut(
    test['Age'],
    bins=[0, 25, 50, 70, 100],
    labels=[0, 1, 2, 3]
)

test['HR_diff'] = (
    (220 - test['Age']) -
    test['Max HR']
)

test['High_Risk'] = (
    (test['BP'] > 140) &
    (test['Cholesterol'] > 240)
).astype(int)

test['stress_ratio'] = (
    test['ST_log'] /
    np.maximum(test['Max HR'], 1)
)

test = pd.get_dummies(test)

bool_cols_test = test.select_dtypes(
    include='bool'
).columns

test[bool_cols_test] = test[bool_cols_test].astype(int)

test = test.reindex(
    columns=X.columns,
    fill_value=0
)


# =========================
# FINAL PREDICTIONS
# =========================

final_predictions = dt_model.predict(test)

print("\nFinal Predictions:\n")

print(final_predictions)


# =========================
# SAVE SUBMISSION FILE
# =========================

submission = pd.DataFrame({
    'Prediction': final_predictions
})

submission.to_csv(
    'submission.csv',
    index=False
)

print("\nsubmission.csv file saved successfully!")


# =========================
# FINAL CONCLUSION
# =========================

print("""
============================
FINAL CONCLUSION
============================

This project successfully applied machine learning techniques
to predict heart disease using patient medical data.

Several preprocessing techniques were applied including:
- handling missing values
- removing outliers
- feature engineering
- log transformation
- encoding categorical variables
- feature scaling

Four classification models were trained and evaluated:
- Logistic Regression
- Decision Tree
- KNN
- SVM

Hyperparameter experiments were also performed to study
the impact of model complexity and hyperparameter tuning.

Decision Tree achieved the best overall performance because
it balanced model simplicity and generalization effectively.

Important features such as:
- Thallium
- Chest pain type
- stress_ratio
- Max HR

had strong influence on heart disease prediction.

This project demonstrates how machine learning can support
medical diagnosis by helping identify patients at risk of
heart disease accurately and efficiently.
""")