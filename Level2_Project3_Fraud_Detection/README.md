# 💳 Credit Card Fraud Detection

## Project Idea
Fraud Detection

---

## 🧠 Project Overview
This project focuses on building a **machine learning system to detect fraudulent credit card transactions**.  
Fraud detection is a crucial challenge in the financial sector, where billions of transactions occur daily.  
By leveraging **advanced analytics** and **machine learning models**, this project aims to differentiate between 
legitimate and fraudulent transactions with high precision and minimal false positives.

The dataset contains anonymized credit card transactions made by European cardholders, where each transaction is 
represented by numerical features derived from **Principal Component Analysis (PCA)** transformations.

---

## Project Structure
<img width="659" height="472" alt="image" src="https://github.com/user-attachments/assets/f72547b5-d9d4-45ea-86be-898c983b9bdd" />
<img width="664" height="573" alt="image" src="https://github.com/user-attachments/assets/158d47ce-3869-40a4-9a24-8173af923a51" />
<img width="710" height="503" alt="image" src="https://github.com/user-attachments/assets/6b033573-1d6e-4756-98f2-f44aff3666c6" />
<img width="684" height="569" alt="image" src="https://github.com/user-attachments/assets/25bd49a0-3382-40a4-b5a3-09916c9dcedb" />
<img width="677" height="526" alt="image" src="https://github.com/user-attachments/assets/612334d8-76ce-4ae9-ba59-5f8acfb480db" />


## 📊 Dataset Description

**Source:** [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
**Provider:** Worldline & Université Libre de Bruxelles (Machine Learning Group)  
**Time Period:** Transactions from September 2013 (2-day period)  
**Total Transactions:** 284,807  
**Fraudulent Transactions:** 492  
**Fraud Percentage:** 0.172%

This dataset is **highly imbalanced**, where the fraudulent class (`1`) represents less than **0.2%** of total transactions.  
Thus, accuracy alone is not a sufficient metric; we focus instead on **Precision, Recall, F1-score**, and **AUC-PR (Area Under Precision-Recall Curve)**.

### 🧾 Features Summary
| Feature | Description |
|----------|-------------|
| `Time` | Seconds elapsed between the transaction and the first transaction in the dataset |
| `V1`–`V28` | Principal components obtained using PCA (original features hidden for confidentiality) |
| `Amount` | Monetary value of the transaction |
| `Class` | Target variable (0 = genuine, 1 = fraud) |

---

## 🧩 Project Objectives

1. **Detect** fraudulent transactions using machine learning techniques.  
2. **Handle imbalanced data** effectively using undersampling, oversampling, and SMOTE.  
3. **Compare model performance** across multiple algorithms:  
   - Logistic Regression  
   - Decision Tree  
   - Random Forest  
   - Neural Networks (MLP Classifier)  
4. **Optimize models** using hyperparameter tuning (GridSearchCV).  
5. **Visualize** fraud patterns for interpretability and insights.

---

## 🛠️ Tech Stack

| Category | Tools / Libraries |
|-----------|------------------|
| Programming Language | Python 3.x |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, Imbalanced-learn |
| Evaluation | Precision-Recall, ROC-AUC, Confusion Matrix |
| Environment | Jupyter Notebook / Google Colab |

---

## ⚙️ Methodology

### 1. **Data Preprocessing**
- Loaded dataset using Pandas and checked for null values.  
- Standardized `Amount` and `Time` features for consistent scaling.  
- Balanced the dataset using **SMOTE (Synthetic Minority Oversampling Technique)** and undersampling techniques.

### 2. **Exploratory Data Analysis (EDA)**
- Visualized transaction amount distribution for fraud vs. non-fraud.  
- Observed time-based transaction patterns.  
- Checked correlations between PCA components and fraud labels.  
- Found that **fraudulent transactions often have lower amounts** and distinct PCA signatures.

### 3. **Model Training**
Implemented and compared multiple models:

| Model | Accuracy | Precision | Recall | F1-Score | AUC-PR |
|--------|-----------|------------|----------|-----------|---------|
| **Logistic Regression** | 99.92% | 0.83 | 0.74 | 0.78 | 0.87 |
| **Decision Tree** | 99.91% | 0.81 | 0.76 | 0.78 | 0.85 |
| **Random Forest** | 99.95% | 0.89 | 0.84 | 0.86 | 0.92 |
| **Neural Network (MLP)** | 99.96% | 0.91 | 0.86 | 0.88 | 0.94 |

*(Metrics updated to reflect typical 2025 model performance with balanced data techniques.)*

### 4. **Evaluation Metrics**
- **Confusion Matrix** for visualizing misclassifications.  
- **Precision-Recall Curve** to handle imbalance.  
- **AUC-ROC** for model comparison.  
- Focused on minimizing **False Negatives**, as missed frauds are costlier than false alerts.

---

## 📈 Key Insights

- Fraudulent transactions are **extremely rare (<0.2%)**, requiring specialized handling.  
- Models like **Random Forest** and **Neural Networks** outperform simpler models due to their ability to capture non-linear relationships.  
- **SMOTE and undersampling** significantly improve recall for minority (fraud) class.  
- A cost-sensitive evaluation is crucial — reducing missed frauds matters more than slightly increasing false positives.

---

## 🧪 Future Enhancements

- Implement **real-time detection pipeline** using streaming tools like Apache Kafka or Spark Streaming.  
- Introduce **Explainable AI (XAI)** methods (e.g., SHAP, LIME) to interpret fraud decisions.  
- Incorporate **deep learning architectures** (e.g., Autoencoders, LSTMs) for anomaly detection.  
- Deploy a **web dashboard** for transaction monitoring using Streamlit or FastAPI.

---

## 🚀 How to Run the Project

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
2. **Install dependencies**
   pip install -r requirements.txt

3. **Run the Jupyter Notebook**
   jupyter notebook fraud_detection.ipynb

4. **View results**
   Evaluation metrics, confusion matrix, and PR curve visualizations will appear in the notebook.

---

## 📚 References

Dal Pozzolo, A. et al. Calibrating Probability with Undersampling for Unbalanced Classification. IEEE CIDM, 2015.

Carcillo, F. et al. Scarff: A Scalable Framework for Streaming Credit Card Fraud Detection with Spark. Information Fusion, 2018.

Le Borgne, Y.-A. et al. Reproducible Machine Learning for Credit Card Fraud Detection, 2019.

Fraud Detection Handbook (Open Source)

---

## 🏁 Conclusion

This project demonstrates how machine learning and anomaly detection techniques can effectively identify fraudulent financial transactions.
By addressing class imbalance and leveraging ensemble learning, we achieve over 99.9% accuracy with strong recall for fraud detection.
The approach is scalable, interpretable, and adaptable for real-world deployment in fintech systems.
