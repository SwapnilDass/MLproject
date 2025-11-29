import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load your reduced dataset (make sure the file is in the same folder as your script)
df = pd.read_csv("video_games_sales_100.csv")

# Show first 5 rows
print("Head of the dataset:")
print(df.head())

# Show dataset info
print("\nDataset info:")
print(df.info())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

from sklearn.preprocessing import StandardScaler

# Select only the numeric sales columns
features = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]
X = df[features]

# Fill missing values with the column mean (in case there are any)
X = X.fillna(X.mean())

# Scale the data for clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Preprocessing complete. Shape:", X_scaled.shape)

inertia = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(K_range, inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.show()

# Fit K-Means with chosen k ---
# After viewing the elbow plot, adjust this number
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print("K-Means clustering complete!")
print(df[["Name", "Genre", "Global_Sales", "Cluster"]].head(10))


# Plot dendrogram
plt.figure(figsize=(10, 6))
dendrogram = sch.dendrogram(sch.linkage(X_scaled, method='ward'))
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Samples")
plt.ylabel("Euclidean distance")
plt.show()


# Run Agglomerative Clustering
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
df["Cluster_HC"] = hc.fit_predict(X_scaled)

print("Hierarchical clustering complete!")
print(df[["Name", "Genre", "Global_Sales", "Cluster_HC"]].head(10))

# Silhouette score for K-Means
kmeans_score = silhouette_score(X_scaled, df["Cluster"])
print("Silhouette Score (K-Means):", kmeans_score)

# Silhouette score for Hierarchical Clustering
hc_score = silhouette_score(X_scaled, df["Cluster_HC"])
print("Silhouette Score (Hierarchical Clustering):", hc_score)

# Reduce to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Add PCA columns to dataframe
df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]

# --- Plot K-Means Clusters ---
plt.figure(figsize=(8, 5))
plt.scatter(df["PC1"], df["PC2"], c=df["Cluster"], cmap="viridis")
plt.title("K-Means Clusters (PCA 2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# --- Plot Hierarchical Clustering ---
plt.figure(figsize=(8, 5))
plt.scatter(df["PC1"], df["PC2"], c=df["Cluster_HC"], cmap="plasma")
plt.title("Hierarchical Clusters (PCA 2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()