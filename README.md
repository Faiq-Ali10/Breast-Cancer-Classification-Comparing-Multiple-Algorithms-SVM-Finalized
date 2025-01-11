# Breast Cancer Classification: Comparing Multiple Algorithms with Final SVM Model

This project is focused on classifying breast cancer data into two categories: **Benign** and **Malignant**. The dataset used is the popular Breast Cancer Wisconsin (Diagnostic) dataset, which is available through the `sklearn.datasets` module.

The objective of this project is to compare multiple machine learning algorithms and determine the best one for predicting the malignancy of tumors in breast cancer cases. After experimenting with different models, **Support Vector Machine (SVM)** with a polynomial kernel was selected as the final model due to its superior performance.

## Dataset

The dataset contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. Each feature represents a specific attribute, such as the size or texture of the tumor, and the target label is binary:
- **1**: Benign
- **0**: Malignant

The dataset includes 30 feature columns with various attributes, such as:
- Mean radius
- Mean texture
- Mean smoothness
- Mean fractal dimension
- Standard error of the mean radius
- and more...

It contains 569 samples, with 357 benign and 212 malignant cases.

## Feature Engineering and Preprocessing

To ensure the accuracy and effectiveness of the models, the following preprocessing and feature engineering steps were performed:

### 1. **Data Loading**
   - The Breast Cancer dataset was loaded from `sklearn.datasets`.

### 2. **Handling Skewness with Log Transformation**
   - Some of the features in the dataset had a high skewness, which can adversely affect the performance of machine learning algorithms. 
   - To address this, a **log transformation** was applied to those skewed features to make their distributions more Gaussian-like, improving the performance of the models.

### 3. **Feature Scaling with Min-Max Normalization**
   - After handling skewness, all the features were scaled using **Min-Max scaling**, ensuring that all feature values lie between 0 and 1. 
   - This transformation ensures that features are on a comparable scale, which is crucial for algorithms that rely on distances, like SVM.

### 4. **Feature Selection**
   - Some features with very low correlation to the target variable (malignant vs benign) were removed from the dataset to improve model performance and reduce computational complexity. This step ensures that only the most relevant features were used to train the model.

### 5. **Data Splitting**
   - The dataset was split into training and testing sets:
     - **Training set**: 80% (455 samples)
     - **Test set**: 20% (114 samples)
   
   - This was done using `train_test_split` from `sklearn`.

### 6. **Dimensionality Reduction (if applicable)**
   - In this project, no explicit dimensionality reduction was required, as the dataset was relatively small with 30 features. However, techniques like PCA (Principal Component Analysis) can be explored for larger datasets to reduce overfitting.

## Algorithms Used

The following machine learning models were compared for this project:

- **Logistic Regression**
- **Random Forest**
- **Support Vector Machine (SVM)** with multiple kernels:
  - Linear
  - Polynomial
  - Radial Basis Function (RBF)
  - Sigmoid
- **K-Nearest Neighbors (KNN)**
- **Decision Tree**
- **XGBoost**

After evaluating the performance of each algorithm, the **SVM with a Polynomial kernel** was found to provide the best results, with high accuracy and minimal overfitting.

## Model Evaluation

The models were evaluated using metrics such as:
- Accuracy
- Precision
- Recall
- F1-score

The final model selected, **SVM with Polynomial kernel**, achieved the following performance:

- **Training Accuracy**: 99%
- **Testing Accuracy**: 99%
- **Precision (Test Set)**: 100% (Benign), 99% (Malignant)
- **Recall (Test Set)**: 100% (Benign), 98% (Malignant)
- **F1-Score (Test Set)**: 99%

## Conclusion

The project demonstrates that the **Support Vector Machine** with a polynomial kernel is highly effective for classifying breast cancer as either benign or malignant. The SVM model was selected due to its balanced performance across various evaluation metrics, especially its high accuracy on both the training and testing sets.

## Getting Started

To run this project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/Faiq-Ali10/Breast-Cancer-Classification-Comparing-Multiple-Algorithms-SVM-Finalized.git
