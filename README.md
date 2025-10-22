# 🧠 Churn-Retention Analysis  

**Churn-Retention Analysis** is an AI-driven web platform that helps businesses **predict customer churn, explain why it happens, and simulate retention strategies** — all in a single, interactive dashboard.  

It combines **machine learning**, **explainable AI (SHAP)**, and **business ROI simulation** to transform customer data into clear, actionable insights for smarter decision-making.  

---

## 🚀 Key Features  

### 🧩 1. Data Upload & Automated EDA  
- Upload datasets in **CSV or Excel** format — no configuration required.  
- Automatically detects file encoding, column types, and missing values.  
- Generates a **Quick EDA Summary** showing:  
  - Dataset shape (rows, columns)  
  - Data types and missing counts  
- Runs a **Full Exploratory Data Analysis (EDA)** with:  
  - Descriptive statistics for numerical features  
  - Correlation heatmap data  
  - Top category distributions for categorical columns  

---

### 🤖 2. Model Training & Evaluation  
- Choose your **target column** (e.g., `Churn`, `Exited`, `Label`) and train predictive models in one click.  
- Supports multiple machine learning algorithms:  
  - Logistic Regression  
  - Random Forest  
  - Decision Tree  
  - Gradient Boosting, AdaBoost, KNN, SVM, and more  
- Performs full **automated data preprocessing**, including:  
  - Cleaning missing values using median imputation  
  - Encoding categorical variables  
  - Scaling numerical features with standardization  
  - Reusing the same transformations during prediction for consistent results  
- Displays real-time **performance metrics and KPIs**, including:  
  - Accuracy, Precision, Recall, F1-score, and ROC-AUC  
  - Overall churn rate and total number of records trained  

---

### 📊 3. Prediction Dashboard  
- Predict churn for the uploaded dataset or upload new data for scoring.  
- Instantly view:  
  - Predicted churn rate (%)  
  - Number of customers analyzed  
  - Average and total revenue (if available)  
  - Estimated business impact (revenue at risk)  
- The dashboard updates dynamically — no downloads or manual refresh needed.  

---

### 💡 4. Explainable AI (Per-Customer Insights)  
- Provides **customer-level churn explanations** using **SHAP (SHapley Additive exPlanations)**.  
- Displays which features influenced a specific prediction and their impact direction.  
- Helps identify **key drivers of churn** and actionable insights for retention.  

---

### 🧮 5. Retention Simulation  
- Simulate business actions such as **discounts**, **plan extensions**, or **incentives**.  
- Instantly visualize how actions affect:  
  - Churn rate (before vs. after)  
  - Customers retained  
  - Revenue saved  
  - ROI (Return on Investment)  
- Quantifies the **financial outcome of interventions** in real time.  

---

### 💬 6. AI Chat Assistant  
- Built-in AI assistant that answers analytical questions about your dataset and model.  
- Ask questions like:  
  - “What is the current churn rate?”  
  - “Which features impact churn the most?”  
  - “How much revenue can be saved after retention?”  
- Optional **LLM-powered mode** (via OpenAI API) for advanced, domain-aware responses and recommendations.  

---

## 💻 Tech Stack  

### Backend & Machine Learning  
- **Python (Flask)** – Web framework & backend controller  
- **Pandas, NumPy** – Data manipulation and preprocessing  
- **Scikit-learn** – Machine learning algorithms  
- **SHAP** – Explainable AI (feature attribution)  
- **XGBoost / CatBoost (optional)** – Advanced gradient boosting models  
- **Gunicorn** – Production-grade WSGI server  

---

## 🔒 Privacy & Security  
- All processing is **in-memory only** — no data is saved or stored.  
- Each user session is **isolated and private**.  
- Data automatically clears when the user closes the tab.  
- Designed with a **privacy-first architecture**, ensuring zero persistence or leakage.  

---

## 💼 Ideal Use Cases  
- Businesses analyzing **customer churn and retention** patterns.  
- **Data science students** demonstrating real-world predictive analytics.  

---

## 🧾 Feature Summary  

| Feature | Description |
|----------|--------------|
| Data Upload | CSV/XLSX upload with automatic detection |
| Quick EDA | Summary of data shape, missing values, and types |
| Full EDA | Correlation matrix and category insights |
| Model Training | Multiple ML models with automatic preprocessing |
| Prediction Dashboard | Real-time churn metrics and KPIs |
| Explainability | SHAP-based feature importance (per-customer) |
| Retention Simulation | ROI and retention effect estimation |
| Chat Assistant | Interactive churn & ROI insights |
| Privacy | In-memory session isolation, no persistent storage |

---

## 🌐 Deployment  
- Backend powered by **Flask + Gunicorn**.  
- Deployable on **Render, Railway, or any cloud platform** supporting Python web apps.  
- Single-page frontend integrated with **HTML, JS, and minimal CSS** for fast performance.  

---

**Developed by:** *Yaswanth Himanshu*  
📊 *An intelligent, privacy-first customer churn prediction & retention optimization system.*


