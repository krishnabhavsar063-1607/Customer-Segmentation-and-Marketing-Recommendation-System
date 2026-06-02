import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ---------------------------------
# PAGE SETTINGS
# ---------------------------------

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    layout="wide"
)

st.title("🛍 Customer Segmentation & Marketing Recommendation System")

# ---------------------------------
# LOAD DATASET
# ---------------------------------

uploaded_file = st.file_uploader(
    "Upload Customer Dataset",
    type=["csv"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

else:
    try:
        data = pd.read_csv("dataset/Mall_Customers.csv")
    except:
        st.error("Upload a CSV file or place Mall_Customers.csv inside dataset folder.")
        st.stop()

# ---------------------------------
# DATA PREVIEW
# ---------------------------------

st.subheader("Dataset Preview")
st.dataframe(data.head())

# ---------------------------------
# KPI SECTION
# ---------------------------------

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Customers",
    len(data)
)

col2.metric(
    "Average Income",
    round(data['Annual Income (k$)'].mean(), 2)
)

col3.metric(
    "Average Spending Score",
    round(data['Spending Score (1-100)'].mean(), 2)
)

# ---------------------------------
# FEATURES
# ---------------------------------

X = data[
    [
        'Annual Income (k$)',
        'Spending Score (1-100)'
    ]
]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ---------------------------------
# SILHOUETTE SCORE
# ---------------------------------

st.subheader("Best Cluster Selection")

scores = []

for k in range(2, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels
    )

    scores.append(score)

best_k = np.argmax(scores) + 2

st.success(
    f"Best Number of Clusters = {best_k}"
)

# ---------------------------------
# TRAIN MODEL
# ---------------------------------

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

data["Cluster"] = clusters

# ---------------------------------
# CUSTOMER LABELS
# ---------------------------------

cluster_summary = data.groupby("Cluster")[
    "Spending Score (1-100)"
].mean()

cluster_names = {}

sorted_clusters = cluster_summary.sort_values()

cluster_names[sorted_clusters.index[0]] = "Budget Customers"
cluster_names[sorted_clusters.index[-1]] = "Premium Customers"

for cluster in sorted_clusters.index[1:-1]:
    cluster_names[cluster] = "Regular Customers"

data["Customer Type"] = data["Cluster"].map(
    cluster_names
)

# ---------------------------------
# CLUSTER VISUALIZATION
# ---------------------------------

st.subheader("Customer Segmentation")

fig, ax = plt.subplots(figsize=(8, 6))

scatter = ax.scatter(
    data["Annual Income (k$)"],
    data["Spending Score (1-100)"],
    c=data["Cluster"],
    cmap="rainbow"
)

ax.set_xlabel("Annual Income")
ax.set_ylabel("Spending Score")
ax.set_title("Customer Clusters")

st.pyplot(fig)

# ---------------------------------
# CUSTOMER DISTRIBUTION
# ---------------------------------

st.subheader("Customer Type Distribution")

st.dataframe(
    data["Customer Type"]
    .value_counts()
    .reset_index()
)

# ---------------------------------
# AGE ANALYSIS
# ---------------------------------

st.subheader("Age Analysis")

fig2, ax2 = plt.subplots()

ax2.hist(
    data["Age"],
    bins=10
)

ax2.set_title("Age Distribution")

st.pyplot(fig2)

# ---------------------------------
# GENDER ANALYSIS
# ---------------------------------

if "Gender" in data.columns:

    st.subheader("Gender Analysis")

    st.dataframe(
        data["Gender"]
        .value_counts()
        .reset_index()
    )

# ---------------------------------
# CUSTOMER PREDICTION
# ---------------------------------

st.subheader("Predict Customer Type")

income = st.number_input(
    "Annual Income",
    min_value=1,
    value=50
)

score = st.number_input(
    "Spending Score",
    min_value=1,
    max_value=100,
    value=50
)

if st.button("Predict"):

    sample = scaler.transform(
        [[income, score]]
    )

    cluster = kmeans.predict(sample)[0]

    customer_type = cluster_names[cluster]

    st.success(
        f"Predicted Type: {customer_type}"
    )

    if customer_type == "Premium Customers":

        st.info("""
        Recommended Strategy:
        • VIP Membership
        • Luxury Product Offers
        • Premium Rewards
        """)

    elif customer_type == "Budget Customers":

        st.info("""
        Recommended Strategy:
        • Discount Coupons
        • Seasonal Offers
        • Cashback Deals
        """)

    else:

        st.info("""
        Recommended Strategy:
        • Loyalty Program
        • Product Recommendations
        • Email Marketing
        """)

# ---------------------------------
# DOWNLOAD SEGMENTED DATA
# ---------------------------------

st.subheader("Download Results")

csv = data.to_csv(index=False)

st.download_button(
    "Download Segmented Customers",
    csv,
    file_name="segmented_customers.csv",
    mime="text/csv"
)