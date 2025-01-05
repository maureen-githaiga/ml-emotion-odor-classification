'''
Author: Maureen Githaiga

Description: This script performs clustering analysis on preprocessed data using the KMeans algorithm, with PCA applied for dimensionality reduction. 
It incorporates parameter tuning through GridSearchCV and evaluates clustering performance using metrics such as silhouette score,
adjusted Rand index, and confusion matrices.

'''

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering,DBSCAN
from sklearn.metrics import accuracy_score,make_scorer, classification_report, confusion_matrix,silhouette_score,adjusted_rand_score

sys.path.append(r'C:\Users\githa\Documents\thesis\Analysis\scripts')


# Parameter grid
kmeans_param_grid = {
    'init': ['k-means++', 'random'],
    'n_init': [10, 20, 30],
    'max_iter': [300, 500, 1000]
    
}


def combine_data(filtered_df):
    combined_data = []
    labels = []
    for index, row in filtered_df.iterrows():
        #smoothed_binary_map = np.array(ast.literal_eval(row['smoothed_Intensity'])).flatten()#on smoothed intensity
       
        smoothed_binary_map = np.fromstring(row['smoothed_binary_map'].replace("\n", "").replace("[", "").replace("]", "").replace(".", ""),sep=" ").flatten()
        #smoothed_binary_map = np.fromstring(row['non_smoothed_binary_map'].replace("\n", "").replace("[", "").replace("]", "").replace(".", ""),sep=" ").flatten()
        
        #smoothed_binary_map = np.array(ast.literal_eval(row['smoothed_binary_map'])).flatten()
        #smoothed_binary_map = np.array(ast.literal_eval(row['normalized_Intensity'])).flatten()#on normalised intensity

        combined_data.append(smoothed_binary_map)
        labels.append(row['Label'])
    combined_data = np.vstack(combined_data)
    labels = np.array(labels)
    return combined_data, labels

# Custom scorer for silhouette score
def silhouette_scorer(estimator, X):
    cluster_labels = estimator.fit_predict(X)
    return silhouette_score(X, cluster_labels)

def apply_kmeans_clustering(data, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_grid_search = GridSearchCV(kmeans, kmeans_param_grid, cv=3, scoring=silhouette_scorer)
    #kmeans.fit(data)
    kmeans_grid_search.fit(data)
    print(f"Best Parameters: {kmeans_grid_search.best_params_}")
    print(f"Best Silhouette Score: {kmeans_grid_search.best_score_}")
    #labels = kmeans.labels_
    best_model = kmeans_grid_search.best_estimator_
    labels = best_model.labels_
    return labels

def visualize_clusters(data, labels,title = 'clusters'):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', marker='o')
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar()
    plt.show()

def visualize_clusters(data, cluster_labels, actual_labels,title):
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data)
    pcadf = pd.DataFrame(data=principal_components, columns=['PCA Component 1', 'PCA Component 2'])
    pcadf['predicted_cluster'] = cluster_labels
    pcadf['true_label_label'] = actual_labels

     # Define labels for the clusters and actual labels
    cluster_labels_map = {0: 'Neutral', 1: 'Fear'}
    true_label_map = {0: 'Neutral', 1: 'Fear'}
    
    # Map the numerical labels to descriptive labels
    pcadf['predicted_cluster_label'] = pcadf['predicted_cluster'].map(cluster_labels_map)
    pcadf['true_label'] = pcadf['true_label_label'].map(true_label_map)
    palette = sns.color_palette("bright",2)
    #palette = sns.color_palette("husl", 2)
    plt.style.use("fivethirtyeight")
    plt.figure(figsize=(8, 8))
    
    scat = sns.scatterplot(
        x='PCA Component 1',
        y='PCA Component 2',
        s=50,
        data=pcadf,
        hue='predicted_cluster_label',
        style='true_label',
        palette=palette
    )
    plt.title(title)
    plt.show()

def evaluate_clustering(combined_data,actual_labels, cluster_labels):
    # Map the actual labels to binary values (0 for neutral, 1 for fear)
    #actual_labels = np.vstack(filtered_df['Label'].values)

    # Calculate the confusion matrix and accuracy
    conf_matrix = confusion_matrix(actual_labels, cluster_labels)
    accuracy = accuracy_score(actual_labels, cluster_labels)
    ari = adjusted_rand_score(actual_labels, cluster_labels)
    silhouette_avg = silhouette_score(combined_data, cluster_labels)
    
    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',cbar=False
                , xticklabels=['Neutral', 'Fear'], yticklabels=['Neutral', 'Fear'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('k-Means Confusion Matrix')
    plt.show()
    
    print(f'Clustering Accuracy: {int(accuracy * 100)}')
    print(f'Adjusted Rand Index (ARI): {ari:.1f}')
    print(f'Silhouette Score: {silhouette_avg:.1f}')  

def main():
    # Load the preprocessed data
    preprocessed_data = pd.read_csv(r"C:\Users\githa\Documents\thesis\Analysis\Data\preprocessed_data_1.csv",index_col = False)
   
    #without the sample 0 in fear and neutral
    #preprocessed_df = pd.read_csv(r'C:\Users\githa\Documents\thesis\Analysis\Data\preprocessed_data_2.csv')
  
    #print(preprocessed_data['smoothed_Intensity'])
    preprocessed_df = preprocessed_data[['smoothed_binary_map','smoothed_Intensity', 'Label']]
    combined_data, actual_labels = combine_data(preprocessed_df)
    #print(combined_data)

    # Apply KMeans clustering to the data
    cluster_labels = apply_kmeans_clustering(combined_data, n_clusters=2)
    # Apply Gaussian Mixture Model clustering
    """gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(combined_data)
    cluster_labels = gmm.predict(combined_data)

    agg_clustering = AgglomerativeClustering(n_clusters=2)
    cluster_labels = agg_clustering.fit_predict(combined_data)"""
    
    # Visualize the clusters
    visualize_clusters(combined_data, cluster_labels,actual_labels,title = 'k-Means Clustering')
    #visualize_clusters(combined_data, actual_labels,title = 'clusters')
    evaluate_clustering(combined_data,actual_labels, cluster_labels)

if __name__ == '__main__':
    main()


