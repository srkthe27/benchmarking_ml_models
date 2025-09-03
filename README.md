# Cyber Attack Detection using Machine Learning

This project applies various **machine learning models** to detect and classify cyber-attacks using the `Enhanced_Synthetic_Cyber_Attack_Dataset.csv`.  
It includes **data preprocessing, visualization, feature balancing (SMOTE), model training, and performance evaluation**.

---

## 📂 Project Structure

- `notebook.ipynb` → Main Jupyter Notebook with code and analysis  
- `Enhanced_Synthetic_Cyber_Attack_Dataset.csv` → Input dataset  
- `plots/` → Generated visualizations (e.g., Protocol vs Attack Type, Packet Size vs Duration)  

---

## ⚙️ Requirements

Install the required libraries before running the notebook:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
```

---

## 🚀 Workflow

1. **Data Loading & Preprocessing**
   - Handle missing values using `SimpleImputer`
   - Encode categorical variables using `LabelEncoder`
   - Standardize numerical features using `StandardScaler`

2. **Data Visualization**
   - Protocol vs Attack Type (Barplot)
   - Packet Size vs Duration (Scatterplot)
   - Additional exploratory plots

3. **Balancing Dataset**
   - Applied **SMOTE** to handle class imbalance

4. **Model Training**
   Implemented multiple classification models:
   - Random Forest Classifier
   - Decision Tree Classifier
   - K-Nearest Neighbors (KNN)
   - Logistic Regression
   - Support Vector Machine (SVM)
   - Naive Bayes (GaussianNB)
   - XGBoost Classifier

5. **Evaluation**
   - Accuracy, Precision, Recall, F1-Score
   - ROC Curve & AUC
   - Classification Report

---

## 📊 Results

- Multiple ML models were trained and compared
- Metrics such as **accuracy, precision, recall, and F1-score** were used for evaluation
- ROC curves and AUC scores provided further insight into model performance

---

## ▶️ Usage

1. Clone this repository or download the project files
2. Place `Enhanced_Synthetic_Cyber_Attack_Dataset.csv` in the same folder as the notebook
3. Run the Jupyter Notebook:

```bash
jupyter notebook a0f699ab-2462-4a58-8024-084c702cef98.ipynb
```

4. Visualizations will be saved inside the `plots/` directory

---

## 🏆 Key Learnings

- Handling imbalanced datasets using **SMOTE**
- Comparing multiple **machine learning models** for cyber security tasks
- Generating meaningful **visual insights** from raw data

---

## 📌 Future Enhancements

- Integrate **deep learning models** (e.g., LSTMs, CNNs)
- Deploy as an **API or Web Application**
- Extend dataset for **real-time intrusion detection**

---

## 👨‍💻 Author

Developed as part of a **Cyber Security & Machine Learning project** for analyzing synthetic attack data.
