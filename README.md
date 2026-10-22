# 🛍️ Customer Segmentation using K-Means Clustering

> **A Machine Learning College Project**  
> **Student:** Krishna Piyush Bhavsar  
> **Domain:** Machine Learning & Data Science  
> **Tech Stack:** Python, Scikit-Learn, Streamlit, Plotly, Pandas  

---

## 📖 1. Project Overview & Problem Statement

In retail stores and e-commerce, customers have different spending habits and income levels. Treating all customers the same leads to wasted marketing budgets and missed sales opportunities.

**Objective:**  
To analyze mall customer data and group shoppers into distinct segments using **Unsupervised Machine Learning (K-Means Clustering)** so that management can launch personalized marketing campaigns (e.g., VIP rewards for high spenders, budget discounts for savers).

---

## 📊 2. Dataset Information

- **Dataset:** `Mall_Customers.csv`
- **Total Records:** 200 customers
- **Features / Attributes:**
  1. `CustomerID`: Unique ID assigned to each shopper.
  2. `Gender`: Male / Female.
  3. `Age`: Customer age (18 to 70 years).
  4. `Annual Income (k$)`: Yearly income in thousands of dollars ($15k to $137k).
  5. `Spending Score (1-100)`: Score assigned by the mall based on customer behavior and purchase frequency.

---

## 🧠 3. Machine Learning Methodology

```
Raw Customer Data
       ↓
Data Preprocessing & Scaling (StandardScaler)
       ↓
Finding Optimal k (Elbow Method & Silhouette Score)
       ↓
Model Training (K-Means with k = 5)
       ↓
Cluster Visualization (2D & 3D Plots with Centroids)
       ↓
Live Web Dashboard & Prediction (Streamlit + Plotly)
```

### Why $k = 5$ Clusters?
- **Elbow Method:** Plotting WCSS (Within-Cluster Sum of Squares) shows a clear bend ("elbow") at **$k = 5$**.
- **Silhouette Score:** Evaluates cluster separation. The highest silhouette score is reached at **$k = 5$** (~0.55), proving that 5 groups offer the best separation.

---

## 👥 4. The 5 Identified Customer Segments

| Group | Segment Name | Income Level | Spending Score | Marketing Strategy |
| :---: | :--- | :---: | :---: | :--- |
| **1** | 🏆 **VIP Customers** | High (> $55k) | High (> 50) | Early access to collections, VIP Club, luxury rewards. |
| **2** | 💎 **Careful Spenders** | High (> $55k) | Low (< 50) | Highlight quality, warranty, and premium bundle offers. |
| **3** | ⚡ **Careless / Impulsive** | Low (< $55k) | High (> 50) | Flash sales, social media promotions, trending items. |
| **4** | 💰 **Budget Seekers** | Low (< $40k) | Low (< 40) | Discount coupons, cashback deals, clearance sales. |
| **5** | 👥 **Standard Shoppers** | Medium ($40k–$70k) | Medium (40–60) | Standard loyalty points, seasonal holiday offers. |

---

## 🚀 5. How to Run the Project

### Step 1: Open Terminal / Command Prompt
Navigate to the project folder:
```powershell
cd "c:\Users\KRISHNA\OneDrive\Desktop\customer_segmentation"
```

### Step 2: Install Required Libraries
```powershell
pip install -r requirements.txt
```

### Step 3: Run the Streamlit Dashboard
```powershell
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser!

### Step 4 (Optional): Run Offline CLI Script
```powershell
python main.py
```
*(Saves static plots to the `outputs/` folder).*

---

## 🖥️ 6. Dashboard Features

1. **📊 Dashboard Overview:** Summary KPI cards (Total Customers, Avg Income, Avg Spending, Avg Age) and pie charts.
2. **📉 Finding Optimal K:** Interactive plots of both the **Elbow Method** and **Silhouette Score**.
3. **🎯 Customer Clusters:** 2D interactive scatter plot with marked cluster centroids (`⭐ Centroid`) and an optional **3D rotation view**.
4. **👥 Customer Groups & Strategies:** Practical marketing recommendations for each cluster.
5. **🤖 Predict New Customer:** Live form where you can enter any Income and Spend score to see which cluster they belong to!
6. **📋 View & Download Data:** Full searchable customer table with download button for `segmented_mall_customers.csv`.

---

## 🎓 7. Common Viva / Interview Questions

<details>
<summary><b>Q1. What is K-Means Clustering?</b></summary>
K-Means is an unsupervised machine learning algorithm that groups similar data points together based on distance to cluster center points called centroids.
</details>

<details>
<summary><b>Q2. Why is feature scaling (StandardScaler) necessary?</b></summary>
Annual income ranges from 15 to 137, while spending score ranges from 1 to 100. Without scaling, features with larger numeric values would dominate distance calculations. StandardScaler standardizes them to mean 0 and variance 1.
</details>

<details>
<summary><b>Q3. How did you choose k=5?</b></summary>
We used two methods:
1. <b>Elbow Method:</b> Found the point where WCSS reduction slows down significantly ($k=5$).
2. <b>Silhouette Score:</b> Achieved the highest cohesion and separation score at $k=5$.
</details>

---

## 👨‍💻 Project Developer
- **Name:** Krishna Piyush Bhavsar
- **Project:** Customer Segmentation using Machine Learning
