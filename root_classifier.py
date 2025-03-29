import os
import argparse
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import hamming
#from sklearn.cluster import HDBSCAN, DBSCAN
import hdbscan
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid
from sklearn.metrics.pairwise import pairwise_distances
import pandas as pd
from sklearn.manifold import MDS
import warnings
import os
import logging



warnings.filterwarnings("ignore")
DETECTION_THRESHOLD = 2

#TODO: remove this
def switch_elements(sketch1, sketch2):
    # Ensure both sketches have at least 5 elements
    elements = 5
    if len(sketch1) < elements or len(sketch2) < elements:
        raise ValueError("Both sketches must have at least 5 elements.")

    # Convert tuples to lists for modification
    sketch1_list = list(sketch1)
    sketch2_list = list(sketch2)

    # Randomly select `elements` indices from each sketch
    indices1 = random.sample(range(len(sketch1)), elements)
    indices2 = random.sample(range(len(sketch2)), elements)

    # Switch the elements between the sketches
    for i in range(elements):
        sketch1_list[indices1[i]], sketch2_list[indices2[i]] = sketch2_list[indices2[i]], sketch1_list[indices1[i]]

    # Convert back to tuples to maintain tuple structure
    return tuple(sketch1_list), tuple(sketch2_list)


def read_sketches_from_file(filename, directory):
    """Reads sketches from text files in a directory. Each file is a sketch (list of labels)."""
    sketches = set()  # Use a set to store unique sketches

    first_sketch_length = None
    file_path = os.path.join(directory, filename)
    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                sketch = tuple(map(int, line.strip().split()))
                
                # If first_sketch_length is None, this is the first sketch
                if first_sketch_length is None:
                    first_sketch_length = len(sketch)
                
                # Only add the sketch if its length matches the first sketch's length
                if len(sketch) == first_sketch_length:
                    sketches.add(sketch)
                #else: print(f"Skipping sketch from {filename} due to length mismatch.")
                    
    return sketches


def compute_distance_matrix(sketches):
    """Compute a distance matrix using Hamming distance between sketches."""
    num_sketches = len(sketches)
    distance_matrix = np.zeros((num_sketches, num_sketches))
    
    for i in range(num_sketches):
        for j in range(i + 1, num_sketches):
            dist = hamming_distance(sketches[i], sketches[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist  # Symmetric matrix
    
    return distance_matrix

def hamming_distance(a, b):
    return hamming(a, b)

def evaluate_model(model, X):
    # Count the number of noise points (HDBSCAN labels noise as -1)
    num_noise_points = np.sum(model.labels_ == -1)
    unique_labels = np.unique(model.labels_)
    
    # Ensure we have more than one cluster before calculating silhouette score
    if len(set(model.labels_)) > 1:
        score = silhouette_score(X, model.labels_, metric='hamming')  # Use the correct metric
    else:
        score = -1  # Return -1 if only one cluster is found
    
    return score, num_noise_points, len(unique_labels)  # Return both values

def plot_tree(clusterer):
    plt.figure(figsize=(10, 6))
    clusterer.condensed_tree_.plot(
    select_clusters=True, 
    colorbar=True, 
)  # Highlights selected clusters
    plt.show()

def hyperparam_testing(data):
    best_score = -1  # Initialize a variable to track the best score
    best_score_params = None  # Variable to store best parameters

    best_noice = 10000  # Initialize a variable to track the best score
    best_noice_params = None  # Variable to store best parameters

    param_grid = {
    'min_samples': [3, 5, 10, 15, 20],
    'min_cluster_size': [3, 5, 10, 15, 20],
}

    r_value = build_directory.rsplit('_', 1)[-1]
    output_dir = r_value
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_filename = os.path.join(output_dir, f"{r_value}_log.txt")
    logging.basicConfig(filename=log_filename, level=logging.INFO,filemode='w', format='%(asctime)s - %(message)s')
    # Perform grid search
    for params in ParameterGrid(param_grid):
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=params['min_cluster_size'],
            min_samples=params['min_samples'],
            metric='hamming'
        )
        
        # Fit the model
        clusterer.fit(data)
        
        plt.figure(figsize=(10, 6))
        clusterer.condensed_tree_.plot(
            select_clusters=True, 
            colorbar=True
        )  # Highlights selected clusters
        #plt.show()
        # Construct the filename with the full path to the new directory
        filename = os.path.join(output_dir, f"{r_value}_tree_size{params['min_cluster_size']}_samples{params['min_samples']}.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")  # Save as high-quality PNG
        plt.close()

        # Evaluate the model using silhouette score
        score, noice_points, clusters = evaluate_model(clusterer, data)
        if noice_points > 0:
            clusters -=1 # to account for the noice label

        logging.info("Params: %s", params)
        logging.info("Noise points: %f", (noice_points / len(data)))
        logging.info("Clusters: %d", clusters)
        logging.info("Silhouette score: %f", score)
        logging.info("--------------")
        
        # Update best score and parameters if necessary
        if clusters > 3:
            if noice_points < best_noice:
                best_noice = noice_points
                best_noice_params = params

            if score > best_score:
                best_score = score
                best_score_params = params

    logging.info("Best score hyperparameters: %s", best_score_params)
    logging.info("Best noise hyperparameters: %s", best_noice_params)
    logging.info("Best Sillhouette score with more than 3 clusters score: %f", best_score)

def save_distance(data, size, sample):
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    distance_matrix = pairwise_distances(data, metric=hamming_distance, n_jobs=-1)
    mds_data = mds.fit_transform(distance_matrix)
    # Visualize 2D
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=mds_data[:, 0], y=mds_data[:, 1], 
        hue=clusterer.labels_, palette="tab10", alpha=0.7
    )
    plt.xlabel("")
    plt.ylabel("")
    plt.title("")
    plt.legend().remove()
    #plt.show()
    r_value = build_directory.rsplit('_', 1)[-1]
    filename = "MDS_" + r_value +"_"+"Size" + str(size) + "_Sample" +str(sample)
    plt.savefig(filename, dpi=300, bbox_inches="tight")  # Save as high-quality PNG
    plt.close()

def compareToModel(sketches, clusterer):
    # Predict cluster for the new datapoint
    p_labels, probabilities = hdbscan.approximate_predict(clusterer, sketches)
    #max_outliers = 0
    outlier_rows = 0
    #print(f"Predicted Cluster: {label[0]}, Probability: {probability[0]}")
    for i, (label, probability) in enumerate(zip(p_labels, probabilities)):

        #print(f"Point {i+1}: Predicted Cluster: {label}, True Cluster: {labels[i]}, Probability: {probability}")

        if label == -1:
            outlier_rows += 1
            if(outlier_rows  >= DETECTION_THRESHOLD):
                return True
        else:
            #max_outliers = max(outlier_rows, max_outliers)
            outlier_rows = 0
    #print(max_outliers)
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Root Classification.")
    parser.add_argument("-b", "--build", required=True, help="Path to the directory containing training sketches")
    parser.add_argument("-t", "--test", required=False, help="Path to the directory containing test sketches")
    parser.add_argument("-v", "--verify", action="store_true", help="Verify data format")

    #TODO: add arg to save model. To make testing with streamspot easier

    args = parser.parse_args()
    build_directory = args.build
    test_directory = args.test
    verify = args.verify

    if not os.path.isdir(build_directory):
        print(f"Error: Directory '{build_directory}' not found.")
        exit(1)

    #TODO: enable this
    """
    if not os.path.isdir(test_directory):
        print(f"Error: Directory '{test_directory}' not found.")
        exit(1)
    """

    filenames = sorted(os.listdir(build_directory))
    unique_sketches = set() 
    for filename in filenames:
        build_sketches = read_sketches_from_file(filename, build_directory)

        for sketch in build_sketches:
            sketch_tuple = tuple(sketch)  # Convert to tuple so it can be added to a set
            unique_sketches.add(sketch_tuple)

    # Convert the unique set of sketches to a numpy array
    build_sketches = np.array(list(unique_sketches), dtype=int)

    if len(build_sketches) == 0:
        print("No sketches found in directory.")
        exit(1)

    print(f"Loaded {len(build_sketches)} sketches.")
    if(verify):
    # Ensure all sketches have the same length
        sketch_length = len(build_sketches[0])
        print("sketch_len: " + str(sketch_length))
        for s in build_sketches:
            if len(s) != sketch_length:
                print("Error: Found a sketch with incorrect length!")
                print("Faulty sketch:", s)
                print(f"Expected length: {sketch_length}, but got: {len(s)}")
                exit(1)

    #TODO: 
    # Alternatives:
    # 4: re-run tests with sketches that actually are 200 labels long for more variance and thus more unique points
    
    #hyperparam_testing(build_sketches)
    #quit()
    #TODO: print spanning tree, and path once htn eremove parameter
    size = 15
    sample = 5
    clusterer = hdbscan.HDBSCAN(min_cluster_size=size, min_samples=sample, metric='hamming', 
                                gen_min_span_tree=True, prediction_data=True).fit(build_sketches)

    labels = clusterer.labels_

    #plot_tree(clusterer)
    #save_distance(build_sketches, size, sample)
    #quit()
    filenames = sorted(os.listdir(build_directory))
    unique_sketches = set() 
    for filename in filenames:
        test_sketches = read_sketches_from_file(filename, build_directory)

        for sketch in test_sketches:
            sketch_tuple = tuple(sketch)  # Convert to tuple so it can be added to a set
            unique_sketches.add(sketch_tuple)

        test_sketches = np.array(list(unique_sketches), dtype=int)
        #TODO: save to output file, or print entire log to file via makefile command
        malicious = compareToModel(test_sketches, clusterer)
        print(f"File: {filename}, is Malicious: {malicious}")