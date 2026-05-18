# 🚀 Production-Ready Credit Card Fraud Detection Pipeline

An end-to-end, high-performance machine learning pipeline built to identify fraudulent financial transactions across **2.7 million+ records**. This project benchmarks multiple architectures—ranging from classic linear baselines to deep neural networks—and solves severe class imbalance ($99.7\%$ legitimate vs. $0.3\%$ fraud) through custom domain-specific feature engineering.

---

## 🏆 Key Performance Milestone
*   **Winning Model:** Random Forest Classifier
*   **Precision:** `1.0000` (**0 False Alarms**)
*   **Recall (Sensitivity):** `0.9963` (**Caught 1,637 out of 1,643 actual fraud cases**)
*   **F1-Score:** `0.9982`

While deep learning architectures (Neural Networks) and gradient boosters (XGBoost) successfully achieved high recall, they cast too wide a net, introducing tens of thousands of operational false alarms. By engineering custom balance-error features, the **Random Forest achieved operational near-perfection with microsecond inference speeds.**

---

## 📊 Pipeline Architecture & Methodology

### 1. Advanced Feature Engineering (The Game Changer)
Instead of relying blindly on raw transaction data or complex black-box architectures, this pipeline exposes the structural patterns of fraud by calculating origin and destination account discrepancy metrics:
*   `errorBalanceOrig` = $NewBalanceOrig + Amount - OldBalanceOrig$
*   `errorBalanceDest` = $OldBalanceDest + Amount - NewBalanceDest$

These engineered features mathematically isolate transaction anomalies, allowing tree-based ensembles to cleanly split the data.

### 2. Model Benchmarking
We scaled the data using `StandardScaler` and evaluated four distinct paradigms:
1.  **Logistic Regression (Baseline):** Established an initial threshold but suffered from poor precision due to high class imbalance.
2.  **Random Forest Ensemble:** The champion architecture. Flawless precision, near-perfect recall, and production-ready inference speeds.
3.  **XGBoost Classifier:** Highly aggressive defender utilizing `scale_pos_weight`; caught fraud but produced more false positives.
4.  **Neural Network (Keras/TensorFlow):** A deep fully-connected network with custom class weight penalties optimized for maximizing coverage.

---

## 📉 Results Summary

| Model Name | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest (Champion)** | **1.0000** | **1.0000** | **0.9963** | **0.9982** |
| XGBoost Classifier | 0.9977 | 0.5613 | 0.9951 | 0.7177 |
| Neural Network (Keras) | 0.9630 | 0.0715 | 0.9592 | 0.1332 |
| Logistic Regression (Baseline) | 0.9349 | 0.0387 | 0.8795 | 0.0742 |

---

## 🛠️ Repository Structure & Artifacts

*   `Fraud_Detection_Pipeline.ipynb`: Complete, documented interactive development notebook.
*   `Fraud_Detection_Pipeline.py`: Production-ready execution script.
*   `winning_random_forest_model.pkl`: Serialized, trained champion model ready for instant deployment.
*   `production_scaler.pkl`: Serialized preprocessing standardizer to maintain data consistency in production.

---

## 🚀 How to Load and Run in Production

You don't need to retrain the models. You can load the pipeline artifacts directly in Python to score new incoming transaction streams instantly:

```python
import pickle
import pandas as pd

# 1. Load the production artifacts
with open("production_scaler.pkl", "rb") as sf:
    scaler = pickle.load(sf)
    
with open("winning_random_forest_model.pkl", "rb") as mf:
    model = pickle.load(mf)

# 2. Predict on new incoming data (Ensure your custom feature math is applied!)
# raw_data = pd.DataFrame(...) 
# scaled_data = scaler.transform(raw_data)
# predictions = model.predict(scaled_data)
