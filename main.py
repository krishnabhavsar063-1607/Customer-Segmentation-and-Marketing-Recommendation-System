import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
import sys

# ---------------------------
# Check Project Structure
# ---------------------------

print("\n===== CUSTOMER SEGMENTATION PROJECT =====\n")

print("Current Working Directory:")
print(os.getcwd())

print("\nFiles/Folders Found:")
print(os.listdir())

# Dataset paths to check
possible_paths = [
    "Mall_Customers.csv",
    "dataset/Mall_Customers.csv"
]

dataset_path = None

for path in possible_paths:
    if os.path.exists(path):
        dataset_path = path
        break

if dataset_path is None:
    print("\nERROR: Mall_Customers.csv not found!")
    print("\nCreate one of these structures:\n")

    print("Option 1:")
    print("""
customer_segmentation/
│
├── main.py
└── Mall_Customers.csv
""")

    print("Option 2:")
    print("""
customer_segmentation/
│
├── main.py
└── dataset/
    └── Mall_Customers.csv
""")

    sys.exit()

print(f"\nDataset Found: {dataset_path}")

# ---------------------------
# Load Dataset
# ---------------------------

data = pd.read_csv(dataset_path)

print("\nFirst 5 Records:")
print(data.head())

print("\nDataset Shape:")
print(data.shape)

# ---------------------------
# Create Output Folder
# ---------------------------

os.makedirs("outputs", exist_ok=True)

# ---------------------------
# Select Features
# ---------------------------

X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# ---------------------------
# Scaling
# ---------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------
# Elbow Method
# ---------------------------

wcss = []

for i in range(1, 11):

    model = KMeans(
        n_clusters=i,
        init='k-means++',
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    wcss.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)

plt.savefig("outputs/elbow_plot.png")
plt.close()

# ---------------------------
# Final KMeans Model
# ---------------------------

kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

data['Cluster'] = clusters

# ---------------------------
# Visualize Clusters
# ---------------------------

plt.figure(figsize=(10, 6))

plt.scatter(
    data['Annual Income (k$)'],
    data['Spending Score (1-100)'],
    c=data['Cluster'],
    cmap='rainbow',
    s=80
)

centers = scaler.inverse_transform(
    kmeans.cluster_centers_
)

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    s=300,
    c='black',
    marker='X',
    label='Centroids'
)

plt.title("Customer Segmentation")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()

plt.savefig("outputs/customer_clusters.png")
plt.close()

# ---------------------------
# Cluster Summary
# ---------------------------

summary = data.groupby('Cluster').mean(numeric_only=True)

print("\nCluster Summary:")
print(summary)

with open(
    "outputs/cluster_report.txt",
    "w"
) as file:

    file.write("Customer Segmentation Report\n\n")
    file.write(str(summary))

print("\nProject Completed Successfully!")

print("\nGenerated Files:")
print("1. outputs/elbow_plot.png")
print("2. outputs/customer_clusters.png")
print("3. outputs/cluster_report.txt")