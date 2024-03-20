# Part 2: Cluster Analysis
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'wholesale_customers.csv'.
def read_csv_2(data_file):
	data = pd.read_csv(data_file)
	data = data.drop(['Channel', 'Region'], axis=1)
	return data

# Return a pandas dataframe with summary statistics of the data.
# Namely, 'mean', 'std' (standard deviation), 'min', and 'max' for each attribute.
# These strings index the new dataframe columns. 
# Each row should correspond to an attribute in the original data and be indexed with the attribute name.
def summary_statistics(df):
    statistics = {
		'min': df.min().round(0),
		'max': df.max().round(0),
		'std': df.std().round(0),
		'mean': df.mean().round(0)
	}

    # print("Info:", type(statistics))
    # Create a DataFrame to display the statistics
    statistics_df = pd.DataFrame(statistics)
    return statistics_df

# Given a dataframe df with numeric values, return a dataframe (new copy)
# where each attribute value is subtracted by the mean and then divided by the
# standard deviation for that attribute.
def standardize(df):
	# Subtract the mean and divide by the std dev for the columns except min and max

    # columns_to_standardize = [col for col in df.columns if col not in ['min', 'max']]
    # df[columns_to_standardize] = (df[columns_to_standardize] - df[columns_to_standardize].mean()) / df[columns_to_standardize].std()

    standardized_df = df.copy()
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    standardized_df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()

    # standardized_df = df
    # standardized_df = (df - df.mean()) / df.std()
    
    return standardized_df

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans.
# y should contain values in the set {0,1,...,k-1}.
# To see the impact of the random initialization,
# using only one set of initial centroids in the kmeans run.
def kmeans(df, k_values = [3, 5, 10]):

    best_score = -1
    best_k = None
    best_labels = None
    
    for k in k_values:
        for init in range(10):
            kmeans = KMeans(n_clusters=k, n_init=1, init='random', random_state=init)
            labels = kmeans.fit_predict(df)
            score = silhouette_score(df, labels)
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
                best_run = init
                
    y = pd.Series(best_labels, index=df.index, name=f'Cluster_Assignments_K={best_k}_Run={best_run}')

    return y

# specifying an assignment of instances to clusters, using kmeans++.
# y should contain values from the set {0,1,...,k-1}.
def kmeans_plus(df, k_values=[3, 5, 10]):
    best_score = -1
    best_k = None
    best_labels = None
    
    for k in k_values:
        # Note: KMeans++ is the default setting for 'init', so it's technically not required to specify it here
        kmeans = KMeans(n_clusters=k, n_init=10, init='k-means++', random_state=42)  # n_init=10 to try 10 different centroid initializations
        labels = kmeans.fit_predict(df)
        score = silhouette_score(df, labels)
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
                
    return best_k, best_score, best_labels

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using agglomerative hierarchical clustering.
# y should contain values from the set {0,1,...,k-1}.
def agglomerative(df, k_values = [3, 5, 10], linkage='single'):
    best_score = -1
    best_k = None
    best_labels = None
    
    for k in k_values:
        agglo = AgglomerativeClustering(n_clusters=k, linkage=linkage)
        labels = agglo.fit_predict(df)
        score = silhouette_score(df, labels)
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    y = pd.Series(best_labels, index=df.index, name=f'Cluster_Assignments_K={best_k}')
                
    return y

start = time()
df = read_csv_2('wholesale_customers.csv')
print(df)
print("Time to read csv:", time() - start, "\n")
start = time()
stats_df = summary_statistics(df)
print(stats_df)
print("Time to read csv:", time() - start, "\n")

# Run k-means clustering
start = time()
y = kmeans(df)
print("k-means 1:\n", repr(y))
print("Time:", time() - start, "\n")

# best_k_plus, best_score_plus, best_labels_plus = kmeans_plus(df)
# print("k-means++ 1:", "\nbest_k:", best_k_plus, "\nbest_score:", best_score_plus, "\nbest_labels:", best_labels_plus)

start = time()
y_2 = agglomerative(df)
print("Agglomerative 1:\n", repr(y_2))
print("Time:", time() - start, "\n")

# Standardise
start = time()
sdf = standardize(df)
print("standardised dataframe:\n", repr(df))
print("sum stats of standardised df:\n", summary_statistics(df))
print("Time:", time() - start, "\n")

# Run it again
start = time()
y_3 = kmeans(sdf)
print("k-means 2:\n", y_3)
print("Time:", time() - start, "\n")

# best_k_plus, best_score_plus, best_labels_plus = kmeans_plus(sdf)
# print("k-means++ 2:", "\nbest_k:", best_k_plus, "\nbest_score:", best_score_plus, "\nbest_labels:", best_labels_plus)

start = time()
y_4 = agglomerative(sdf)
print("Agglomerative 2:\n", y_4)
print("Time:", time() - start, "\n")













# !! Must be for 3, 5, 10 - add for k in 3, 5, 10 to function. 
# Don't use ward default, agglomerative should be the best by far
# Should have 66 rows

# Given a data set X and an assignment to clusters y
# return the Silhouette score of this set of clusters.
def clustering_score(X,y):
	silhouette_score_2 = silhouette_score(X, y)
	return silhouette_score_2

clustering_score_ = clustering_score(sdf, y_3)
print("clustering_score_2 kmeans:", clustering_score_, "\n")
clustering_score_agglo = clustering_score(sdf, y_4)
print("clustering_score_2 agglo:", clustering_score_agglo, "\n")

# Perform the cluster evaluation described in the coursework description.
# Given the dataframe df with the data to be clustered,
# return a pandas dataframe with an entry for each clustering algorithm execution.
# Each entry should contain the: 
# 'Algorithm' name: either 'Kmeans' or 'Agglomerative', 
# 'data' type: either 'Original' or 'Standardized',
# 'k': the number of clusters produced,
# 'Silhouette Score': for evaluating the resulting set of clusters.
def cluster_evaluation(sdf, k_values=[3, 5, 10], linkage_methods=['ward', 'complete', 'average', 'single']):
    results_list = []

    # KMeans
    for k in k_values:
        # Wrap k in a list since kmeans expects a list for k_values
        y = kmeans(df, [k])  
        score = clustering_score(sdf, y)
        results_list.append({
            'Algorithm': 'KMeans',
            'data': 'Standardized',  
            'k': k,
            'Silhouette Score': score
        })
        
    # Agglomerative Clustering
    for k in k_values:
        for linkage in linkage_methods:
            y = agglomerative(sdf, [k], linkage)  
            score = clustering_score(sdf, y)
            results_list.append({
                'Algorithm': 'Agglomerative',
                'data': 'Standardized',  
                'k': k,
                'Silhouette Score': score,
            })

    results_df = pd.DataFrame(results_list)
    return results_df

print("DEBUG:", sdf)

start = time()
results_df = cluster_evaluation(sdf)
print("Cluster evaluation:\n", results_df, "\n")
print("Time:", time() - start, "\n")






# Given the performance evaluation dataframe produced by the cluster_evaluation function,
# return the best computed Silhouette score.
def best_clustering_score(rdf):
	# Find the maximum Silhouette Score in the DataFrame
    best_score = rdf['Silhouette Score'].max()
    return best_score

start = time()
best_score = best_clustering_score(results_df)
print("Best score:", best_score)
print("Time:", time() - start, "\n")








# Generate a scatter plot for each pair of attributes.
# Data points in different clusters should appear with different colors.
def scatter_plots(df):

    results_df = cluster_evaluation(df)
    best_config = results_df.loc[results_df['Silhouette Score'].idxmax()]
    
    # Identify the best clustering config details
    algorithm = best_config['Algorithm']
    data_type = best_config['data']
    k = best_config['k']
    
    if algorithm == 'KMeans':
        y = kmeans(sdf, [k])
    else:
        y = agglomerative(sdf, [k])
    
    num_attributes = sdf.shape[1]
    attributes = sdf.columns
    fig, axes = plt.subplots(num_attributes, num_attributes, figsize=(16, 10))
    
    for i in range(num_attributes):
        for j in range(num_attributes):
            if i == j:
                sns.kdeplot(data=sdf, x=attributes[i], ax=axes[i, j], fill=True)
                axes[i, j].set_title(f'Distribution of {attributes[i]}')
            else:
                sns.scatterplot(data=sdf, x=attributes[i], y=attributes[j], hue=y, ax=axes[i, j], palette='viridis', alpha=0.6)
                axes[i, j].set_title(f'Scatter plot of {attributes[i]} vs {attributes[j]}')
                axes[i, j].set_xlabel(attributes[i])
                axes[i, j].set_ylabel(attributes[j])
                axes[i, j].legend(title='Cluster')
    
    plt.tight_layout()
    plt.show()

# !! Different, make sure this is correct
# start = time()
scattered = scatter_plots(sdf)
print(scattered)
# print("Time:", time() - start, "\n")