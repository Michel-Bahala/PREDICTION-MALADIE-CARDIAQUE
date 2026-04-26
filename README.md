# 🚀 Heart Disease Prediction App (ML + Streamlit)

## 🧠 Overview

This project focuses on **early detection of heart disease risk** using real-world health data from the BRFSS 2015 dataset (229,781 records).

In healthcare, missing a high-risk patient can have severe consequences.
👉 This model is therefore **optimized for high recall (sensitivity)** to minimize false negatives.

## 🎯 Key Features

* End-to-end machine learning pipeline
* Advanced feature engineering on medical indicators
* Class imbalance handling (balanced dataset strategy)
* Hyperparameter optimization using Optuna
* Deployment-ready interface with Streamlit

## 🧪 Model Performance (Test Set)

* Accuracy: **74.82%**
* Recall: **81.84%** ✅ *(priority metric)*
* Precision: **71.60%**
* F1 Score: **76.38%**
* ROC-AUC: **0.827**

📌 **Key insight:** The most important feature was
👉 **Age × BMI interaction**, highlighting the power of feature engineering.


## 🔬 Feature Engineering Highlights

* Age × BMI (most impactful)
* Alcohol × Smoking
* Income × Hypertension
* BMI transformations (log, squared)

## 🖥️ Streamlit App (Interactive Demo)

An interactive web app was built using Streamlit to make the model accessible to non-technical users.

### 💡 What the app does:

* Users input health indicators (BMI, age, smoking habits, etc.)
* Model predicts heart disease risk
* Displays probability + risk interpretation

### ▶️ Run locally:

```bash id="runlocal02"

# Navigate into the project folder
cd heart-disease-ml

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

## 📸 App Preview

*(Add screenshots or GIF demo here — highly recommended for recruiters)*


## ⚙️ Tech Stack

* Python (Pandas, NumPy, Scikit-learn)
* Optuna (hyperparameter tuning)
* Streamlit (deployment)
* Matplotlib / Seaborn (EDA)

## 🏥 Real-World Relevance

This model is suitable for:

* Early disease screening systems
* Clinical decision support tools
* Preventive healthcare analytics

## 📁 Project Structure

```id="structure02"
├── data/
├── notebooks/
├── src/
├── model/
├── app.py
└── README.md

## 🌍 Career Focus

I am actively seeking opportunities as a:

* Machine Learning Engineer
* Data Scientist

🎯 Focus: Applied AI, Healthcare, Predictive Modeling, Real-world ML Systems
🌐 Open to on-site, hybrid, or remote roles

## 🤝 Let’s Connect

I’m open to:

* Feedback on this project
* Collaboration opportunities
* Internships / Junior roles
