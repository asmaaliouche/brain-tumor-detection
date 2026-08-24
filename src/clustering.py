import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

def perform_dimensionality_reduction(features):
    """
    Standardizes features and applies PCA and t-SNE for dimensionality reduction.
    """
    logger.info("Standardizing features...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    logger.info("Computing PCA decomposition...")
    pca = PCA(n_components=2, random_state=42)
    pca_results = pca.fit_transform(scaled_features)
    logger.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    
    logger.info("Computing t-SNE non-linear projection...")
    # Fix: TSNE uses 'max_iter' instead of 'n_iter'
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42, init='pca')
    tsne_results = tsne.fit_transform(scaled_features)
    
    logger.info("Dimensionality reduction complete.")
    return scaled_features, pca_results, tsne_results

def perform_kmeans_clustering(scaled_features, n_clusters=2):
    """
    Applies K-Means clustering on the scaled features.
    """
    logger.info(f"Applying K-Means clustering (K={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(scaled_features)
    logger.info("K-Means clustering complete.")
    return kmeans_labels
