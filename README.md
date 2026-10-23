# Osamah Al Hatef — Data Science & Analytics Portfolio

M.S. Data Science (University of Michigan). I build end-to-end data projects: warehouse design and SQL analytics, machine-learning pipelines with honest evaluation, and cloud systems on AWS. Every project below runs end to end, and every number in its README is a real output captured in the repo.

---

## 🚀 Projects

### 🐘 PostgreSQL Customer Analytics & dbt Warehouse
Two builds of the same customer warehouse: a **PostgreSQL 16** database (schema design with PKs/FKs/indexes, 241,609 billing rows loaded via Python COPY, native SQL with CTEs and window functions, 7/7 reconciliation checks) and a **dbt + DuckDB** analytics-engineering project (staging → intermediate → marts, 22 data tests, `dbt build` PASS=28). Worst churn segment: month-to-month, 0–6 months at 62.35% vs 3.69% for the safest.
→ [PostgreSQL project](postgres-analytics/post) · [dbt project](dbt-analytics/DB)

### 📉 Customer Churn Analytics Pipeline
SQLite ETL + segmentation SQL, Python EDA, and a churn classifier comparison (logistic regression vs random forest, 5-fold cross-validation; logistic regression won, ROC-AUC 0.75, catches 72% of churners). Key finding: 47.8% of customers churn in their first year.
→ [Churn project](customer-churn-analytics/churn-analytics)

### 🛡️ Real-Time Fraud Detection System
100,000 transactions, shared train/serve feature engineering, cost-tuned classification threshold (precision 0.748, recall 0.957 at the chosen threshold; expected cost $3,246 vs $135,549 no-model baseline), a FastAPI prediction service, batch scoring, PSI drift monitoring, and passing tests.
→ [Fraud project](fraud-detection/00)

### 🔥 Deep Learning from Scratch
Perceptron and deep neural networks implemented in pure NumPy — forward/backpropagation, gradient descent, cross-validation, learning curves.
→ [Deep learning project](deep-learning-from-scratch)

### ☁️ Smart Door OTP System
Cloud visitor-registration and OTP validation system on AWS (API Gateway, Lambda, DynamoDB).
→ [Smart Door project](smart-door-otp-system)

### 🤖 Coffee Chatbot (Live Demo)
Conversational ordering assistant built with AWS Lex, Lambda, DynamoDB, and API Gateway.
[Live demo](http://coffee-chatbot-osamah.s3-website-us-east-1.amazonaws.com/) · [Code](coffee-chatbot-aws)

### 📊 Polynomial Regression
Regression modeling with ridge regularization, cross-validation, and hyperparameter tuning by RMSE.
→ [Polynomial regression project](polynomial-regression)

---

## 🛠 Technical Skills

- **Languages:** Python, SQL, R, Java, C++, Excel
- **Data & ML:** pandas, NumPy, SciPy, scikit-learn, TensorFlow/Keras, PySpark, Hadoop, ETL pipelines, PostgreSQL, dbt, feature engineering, predictive modeling, hypothesis testing, cross-validation
- **BI & Visualization:** Power BI, Tableau, dashboard development, KPI reporting, Matplotlib, Seaborn
- **Cloud & Tools:** AWS (Lambda, API Gateway, DynamoDB, S3, Lex, SNS/SES), REST APIs, FastAPI, SQLite, Git/GitHub

---

## 📫 Contact

LinkedIn: [linkedin.com/in/osamahds](https://www.linkedin.com/in/osamahds) · Dearborn, MI
