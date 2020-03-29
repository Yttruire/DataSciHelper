# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 21:38:04 2020

@author: Zhi Hao
"""
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.decomposition import PCA

from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from matplotlib import cm


def kendall_pval(x,y):
    return kendalltau(x,y)[1]

def pearsonr_pval(x,y):
    return pearsonr(x,y)[1]

def spearmanr_pval(x,y):
    return spearmanr(x,y)[1]

def corr_finder(df:pd.DataFrame, threshold=0.3, get_list=False, p_value=False, method=None):
    if method is None:
        method = 'pearson'
    
    df = df.corr(method=method)

    # Calculate p-values if requested
    if p_value:
        if method == 'pearson':
            df_pv = df.corr(method=pearsonr_pval)
        elif method == 'spearman':
            df_pv = df.corr(method=spearmanr_pval)
        elif method == 'kendall':
            df_pv = df.corr(method=kendall_pval)
            
    corr_list = list()
    
    # Combination of for statements add up to iterate through all matrices of the 
    # bottom-left triangular half of the correlation matrix
    for y in range(1, df.shape[1]):
        for x in range(0, y):
            
            # If correlation is above given threshold
            if abs(df.iloc[x, y]) > threshold:
                
                # If p-value is desired, print it together with correlation and coordinates
                if p_value:
                    print('({}, {})'.format(x, y), '{} has a correlation of'.format(df.columns[x]), 
                          round(df.iloc[x, y], 4), 'with {}'.format(df.columns[y]),
                         'with p-value of', round(df_pv.iloc[x, y], 4))
                else:
                    print('({}, {})'.format(x, y), '{} has a correlation of'.format(df.columns[x]), 
                          round(df.iloc[x, y], 4), 'with {}'.format(df.columns[y]))
                    
                # If a list was requested to be returned
                if get_list:
                    # Add p-value into list if it is desired
                    if p_value:
                        corr_list.append([x, y, round(df.iloc[x, y], 4), round(df_pv.iloc[x, y], 4)])
                    else:
                        corr_list.append([x, y, round(df.iloc[x, y], 4)])
                    
                    
    return corr_list

def plot_heatmap(df, cmap=None, figsize=None, ax=None, **kwargs):
    if figsize is None:
        figsize = (16, 12)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    if cmap is None:
        cmap = 'coolwarm'
        
    ax = sns.heatmap(df, ax=ax, **kwargs)
    
    b, t = ax.get_ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    ax.set_ylim(b, t) # update the ylim(bottom, top) values
    
    
def plot_scree_plot(pca, plot_cumul=True):
    '''For visualising percentage of variance explained by each PC
    '''
    fig, ax = plt.subplots(figsize=(12,8))
    pca_var = pca.explained_variance_ratio_
    ax.set_xlim(1, len(pca_var))
    ax.set_ylim(0, 1)
    ax.plot(list(range(1, len(pca_var) + 1)), pca_var, '-o')
    if plot_cumul:
        cumulative_pca_var = [sum(pca_var[:i]) for i in range(1, len(pca_var) + 1)]
        plt.plot(list(range(1, len(pca_var) + 1)), cumulative_pca_var, '-o')
        return pca_var, cumulative_pca_var
    else:
        return pca_var
    

def plot_pca3d(df, labels=None):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['PC1'], df['PC2'], zs=df['PC3'], c=labels, s=20, marker='x')
    
    
def get_percentage_range(df, column, lower_percent=0.0, upper_percent=1.0):
    return df.sort_values(by=column).iloc[math.floor(df.shape[0] * lower_percent):math.floor(df.shape[0] * upper_percent + 1)]

def get_bounded_range(df, column, lower_bound=None, upper_bound=None):
    if lower_bound is None:
        lower_bound = min(df[column])
    if upper_bound is None:
        upper_bound = max(df[column])
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def count_percentage_range(df, column, lower_percent=0.0, upper_percent=1.0):
    sub_sample = get_percentage_range(df, column, lower_percent=lower_percent, upper_percent=upper_percent)
    print('Percentage of scores_sample of total sample size:', len(sub_sample) / df.shape[0])
    print('Number of observations in scores_sample:', len(sub_sample))
    return sub_sample


def count_bounded_range(df, column, lower_bound=None, upper_bound=None):
    if lower_bound is None:
        lower_bound = min(df[column])
    if upper_bound is None:
        upper_bound = max(df[column])
        
    sub_sample = get_bounded_range(df, column, lower_bound, upper_bound)
    print('Percentage of scores_sample of total sample size:', len(sub_sample) / df.shape[0])
    print('Number of observations in scores_sample:', len(sub_sample))
    return sub_sample


def plot_percentage_range_hist(df, column, lower_percent=0.0, upper_percent=0.2, bins=30, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    sub_sample = count_percentage_range(df, column, lower_percent=lower_percent, upper_percent=upper_percent)[column]
    
    if bins != 0:
        bins = np.linspace(min(sub_sample), max(sub_sample), bins)
        ax.hist(sub_sample, bins=bins)
    else:
        ax.scatter(list(range(1, len(sub_sample) + 1)), sub_sample)
        
        
def plot_bounded_range_scatter(df, column, lower_bound=None, upper_bound=None, ax=None):
    if lower_bound is None:
        lower_bound = min(df[column])
    if upper_bound is None:
        upper_bound = max(df[column])
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
    sub_sample = count_bounded_range(df, column, lower_bound=lower_bound, upper_bound=upper_bound)[column]

    ax.scatter(list(range(1, len(sub_sample) + 1)), sub_sample.sort_values())
    
def plot_gradient_plot(s: pd.Series, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
    ax.plot(s.diff())
    
    
def plot_multiple_plots(n_rows: int, n_columns: int, plot_func, args_dict_list: list, figsize=None):

    assert len(args_dict_list) == n_rows * n_columns, 'Number of sets of arguments do not match number of plots'
    
    if figsize is None:
        figsize = (12, 8)
        
    fig, axs = plt.subplots(n_rows, n_columns, figsize=figsize)
    
    n_iter = 0
    for i in range(0, n_rows):
        # Plot two dimension plots only if there are more than one column
        if n_columns > 1:
            for j in range(0, n_columns):
                plot_func(**(args_dict_list[n_iter]), ax=axs[i, j])
                n_iter += 1
        else:
            plot_func(**(args_dict_list[n_iter]), ax=axs[i])
            n_iter += 1
            
            


def plot_silhouette_score(X, cluster_labels, n_clusters, ax=None):
    if ax is None:
        # If not given an axis to plot on, create own one
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
    # The silhouette coefficient can range from -1, 1, but we check [-0.2, 1]
    ax.set_xlim([-0.2, 1])
    
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax.set_title("The silhouette plot for the various clusters.")
    ax.set_xlabel("The silhouette coefficient values")
    ax.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.suptitle(("Silhouette analysis for clustering on data with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

    plt.show()