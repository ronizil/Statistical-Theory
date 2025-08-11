import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="umap")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE, Isomap
from umap import UMAP
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# --- קריאת הקובץ ---
df = pd.read_csv("gym_members_exercise_tracking.csv")

# --- שמירה על עמודות נומריות בלבד ---
df_numeric = df.select_dtypes(include=[np.number]).dropna()

# --- הסרה של עמודת cluster אם קיימת ---
if 'cluster' in df_numeric.columns:
    df_numeric = df_numeric.drop(columns='cluster')

# --- סטנדרטיזציה ---
X_scaled = StandardScaler().fit_transform(df_numeric)

# --- הגדרת שיטות צמצום ממדים ---
dr_methods = {
    'PCA': PCA(n_components=2, random_state=42),
    'TSNE': TSNE(n_components=2, perplexity=30, random_state=42),
    'UMAP': UMAP(n_components=2, random_state=42),
    'UMAP50': UMAP(n_components=2, n_neighbors=50, random_state=42),
    'UMAP15': UMAP(n_components=2, n_neighbors=15, random_state=42),
    'ICA': FastICA(n_components=2, random_state=42),
    'Isomap': Isomap(n_components=2)
}

# --- הגדרת מספר אשכולות ---
K = 3

# --- ציור גרפים ---
fig, axes = plt.subplots(1, len(dr_methods), figsize=(22, 4), squeeze=False)
axes = axes[0]

for ax, (name, dr) in zip(axes, dr_methods.items()):
    try:
        X_dr = dr.fit_transform(X_scaled)
        labels = AgglomerativeClustering(n_clusters=K).fit_predict(X_dr)
        sil = silhouette_score(X_dr, labels) if len(set(labels)) > 1 else None
        sil_text = f"Silhouette = {sil:.2f}" if sil is not None else "Silhouette = N/A"
        ax.scatter(X_dr[:, 0], X_dr[:, 1], c=labels, cmap='tab10', s=10, alpha=0.7)
        ax.set_title(f"{name}\n{sil_text}", fontsize=9)
    except Exception as e:
        ax.set_title(f"{name}\nFAILED", fontsize=9)
        ax.text(0.5, 0.5, str(e), ha='center', va='center', fontsize=6)
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle(f"Agglomerative Clustering (K={K}) on Different Dim. Reduction Methods", fontsize=14, y=1.05)
plt.tight_layout()
plt.show()
