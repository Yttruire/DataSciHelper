'''
def kendall_pval(x,y)
- Returns the p value of a kendall rank correlation

def pearsonr_pval(x,y)
- Returns the p value of a pearson correlation

def spearmanr_pval(x,y)
- Returns the p value of a spearman rank correlation

def corr_finder(df:pd.DataFrame, threshold=0.3, print_corr=True, get_list=False, p_value=False, method=None) -> list
def get_color_map(threshold, inverse=False)
def get_zoutlier(df: pd.DataFrame, column_to_test: str, threshold=4, style=True)
def get_multicolumn_zoutlier(df: pd.DataFrame, columns_to_test: list, threshold=4, style=True)

pandas_manip
def get_df_missing_data(df: pd.DataFrame) -> pd.DataFrame
def find_diff_pairs_rows(df: pd.DataFrame, index=None, column_index=None, step=1, pair_step=2, print_max=100) -> dict

plot_funcs
def plot_scree_plot(pca, plot_cumul=True, figsize=None, ax=None)
def plot_2d(df_pca, labels, columns_index: list=None, figsize=None, ax=None, **kwargs)
def plot_3d(df_pca, labels, columns_index: list=None, figsize=None, ax=None, **kwargs)
def plot_kmeans_inertia(X, range_n_clusters, figsize=None, dpi=80, get_list=False)
def plot_kmeans_silhouette_scores(X, list_n_clusters)

def get_percentage_range(df, column, lower_percent=0.0, upper_percent=1.0)
def get_bounded_range(df, column, lower_bound=None, upper_bound=None)
def count_percentage_range(df, column, lower_percent=0.0, upper_percent=1.0)
def count_bounded_range(df, column, lower_bound=None, upper_bound=None)
def plot_percentage_range_hist(df, column, lower_percent=0.0, upper_percent=0.2, bins=30, figsize=None, ax=None)
def plot_bounded_range_scatter(df, column, lower_bound=None, upper_bound=None, figsize=None, ax=None)
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# For kendall, pearson and spearman pvals
from scipy.stats import kendalltau, pearsonr, spearmanr

# For get_zoutlier and get_multicolumn_zoutlier
from sklearn.preprocessing import StandardScaler
# For plot_kmeans_inertia
from sklearn.cluster import KMeans

# For plot_kmeans_silhouette_scores
from sklearn.metrics import silhouette_score

# For get_percentage_range
import math


def kendall_pval(x,y):
    '''Returns the p value of a kendall rank correlation
    '''
    return kendalltau(x,y)[1]


def pearsonr_pval(x,y):
    '''Returns the p value of a pearson correlation
    '''
    return pearsonr(x,y)[1]


def spearmanr_pval(x,y):
    '''Returns the p value of a spearman rank correlation
    '''
    return spearmanr(x,y)[1]


def corr_finder(df:pd.DataFrame, threshold=0.3, print_corr=True, get_list=False, p_value=False, method='pearson') -> list:
    '''Returns a list of [corr_value, [i, j]] where [i, j] are the coordinates of the value on the correlation matrix
    
    Parameters
    ---------
    df : pd.DataFrame

    threshold : float, default=0.3
        The abs(threshold) at which to flag a relationship between two features as being correlated
        
    print_corr : bool, default=True
        Prints the results of the correlation finder if True
    
    get_list : bool, default=False
        Returns the list of correlations in the format [x, y, corr_value, p_value]
        p_value is only returned if p_value=True

    p_value : bool, default=True
        Prints the corresponding p value of a correlation if print_corr=True, and adds it to the list returned if get_list=True

    method : {'pearson', 'spearman', 'kendall'} or function, default='pearson'
        The method with which to calculate the correlation matrix

    Returns
    -------
    corr_list: list or None, shape [n_correlations, 3 or 4]
        List is returned only upon request via get_list.
        n_correlations = number of correlations found above given threshold
        3 or 4 columns returned. 3 if p_value=False, 4 if otherwise
        
    Notes
    -----
    Only accepts DataFrames without categorical features (Or has been OneHotEncoded properly)
    Checks through correlations of the bottom left triangle of the correlation matrix
    If categorical features are present, coordinates i and j will no longer reflect the correct column coordinates as in df.columns
    '''
    # If the shape of df.corr() is not equal to a square matrix with the len/width equal to df.shape[1], there are categorical features
    assert df.shape[1] == df.shape[1], \
        'Correlation matrix shape should equal ({0}, {1}), it is instead {2}. Are there categorical features inside?'\
        .format(df.shape[1], df.shape[1], df.shape)
    
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
    
    # Combination of for statements iterate through all matrices of the 
    # bottom-left triangular half of the correlation matrix
    for y in range(1, df.shape[1]):
        for x in range(0, y):
            
            # If correlation is above given threshold
            if abs(df.iloc[x, y]) > threshold:

                # Print anything only if requested (Default)
                # If p-value is desired, print it together with correlation and coordinates
                if print_corr and p_value:
                    print('({}, {})'.format(x, y), '{} has a correlation of'.format(df.columns[x]), 
                          round(df.iloc[x, y], 4), 'with {}'.format(df.columns[y]),
                         'with p-value of', round(df_pv.iloc[x, y], 4))
                elif print_corr:
                    print('({}, {})'.format(x, y), '{} has a correlation of'.format(df.columns[x]), 
                          round(df.iloc[x, y], 4), 'with {}'.format(df.columns[y]))
                    
                # If a list was requested to be returned
                if get_list:
                    # Add p-value into list if it is desired
                    if p_value:
                        corr_list.append([x, y, round(df.iloc[x, y], 4), round(df_pv.iloc[x, y], 4)])
                    else:
                        corr_list.append([x, y, round(df.iloc[x, y], 4)])
                    
    if get_list:            
        return corr_list


def get_color_map(threshold, inverse=False):
    ''' Returns a color map function for use in DataFrame styling. Colors are mapped based on a given threshold.
    
    Arguments
    ---------
    threshold : int or float
        When threshold is int or float, colors values >= threshold green, >= 1.5 threshold blue, >= 2 threshold red
    
    inverse : bool, default=False
        When True, colors values < threshold green, < 1.5 threshold blue, < 2 threshold red

    Notes
    -----
    Pass a list(Of shape (1, 3)) for the threshold to prevent above behaviour. 
    Threshold in index [2] (Color red) takes priority over threshold in index [1] (Color blue).
    E.g. Given threshold [2, 2, 2], color map will map all values above or equal to 2 to be red, and the rest black
    '''
    if type(threshold) is int or type(threshold) is float:
        threshold = [threshold, threshold + threshold / 2, threshold * 2]
    
    def color_map(val):
        if inverse:
            if abs(val) < threshold[0]:
                return 'color: red'
            elif abs(val) < threshold[1]:
                return 'color: blue'
            elif abs(val) < threshold[2]:
                return 'color: green'
            else:
                return 'color: black'
        else:
            if abs(val) >= threshold[2]:
                return 'color: red'
            elif abs(val) >= threshold[1]:
                return 'color: blue'
            elif abs(val) >= threshold[0]:
                return 'color: green'
            else:
                return 'color: black'
        
    return color_map


def get_zoutlier(df: pd.DataFrame, column_to_test: str, threshold=4, style=True):
    ''' Returns a dataframe of all outlier values in a single column according to given threshold (In sigma) using Z test
    
    Arguments:
        column_to_test: A string of the name of the column to be tested in the DataFrame df
        threshold: The threshold to flag out a value as an outlier (In sigma, the standard deviation of a Z-test)
        style: True to return a styled DataFrame. False for a normal DataFrame
    '''
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[column_to_test].values.reshape(-1, 1))
    outlier_list = [index for index, value in enumerate(scaled_values) if abs(value) > 4]
    df_outlier = pd.DataFrame(scaler.inverse_transform(scaled_values[outlier_list]), columns=[column_to_test])
    df_outlier[column_to_test + '_scaled'] = scaled_values[outlier_list]
    df_outlier.index = outlier_list
    
    if style:
        return df_outlier.style.applymap(get_color_map(threshold), subset=pd.IndexSlice[:, column_to_test + '_scaled'])
    else:
        return df_outlier
    
    
def get_multicolumn_zoutlier(df: pd.DataFrame, columns_to_test: list, threshold=4, style=True):
    ''' Returns a dataframe of all outlier values in multiple columns according to a given threshold (In sigma) using Z test
    
    Arguments:
        columns_to_test: A list of strings of the names of the columns to be tested in the DataFrame df
        threshold: The threshold to flag out a value as an outlier (In sigma, the standard deviation of a Z-test)
        style: True to return a styled DataFrame. False for a normal DataFrame
    '''
    temp_df = pd.DataFrame()
    
    for column in columns_to_test:
        temp_df = pd.concat([temp_df, get_zoutlier(df, column_to_test=column, threshold=threshold, style=False)], axis=1, sort=False)
    
    if style:
        return temp_df.style.applymap(get_color_map(threshold), subset=pd.IndexSlice[:, [column + '_scaled' for column in columns_to_test]])
    else:
        return temp_df



def get_df_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    '''Returns a DataFrame documenting all features with missing data (NaN)

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing the features with missing data to be counted
        and then returned by the function in a new DataFrame
        
    Returns
    -------
    DataFrame containing: 
        name of column with missing data
        index of column with missing data
        number of missing data in column
        percentage of missing data in column
    '''
    
    df_missing_data = pd.DataFrame(columns = ['name', 'index', 'number_of_missing_data', 'percentage_missing'])
    
    for index, column in enumerate(df.columns):
        # Calculate number and percentage of missing data
        no_of_missing_data = len(df[column]) - df[column].count()
        percentage_missing = no_of_missing_data / len(df) * 100
        # Append to DataFrame of missing data if missing data found in current feature
        if no_of_missing_data:
            df_missing_data = df_missing_data.append({'name': df.columns[index], 
                                                      'index': index, 
                                                      'number_of_missing_data': no_of_missing_data, 
                                                      'percentage_missing': percentage_missing}, ignore_index = True)
    return df_missing_data


def find_diff_pairs_rows(df: pd.DataFrame, index=None, column_index=None, step=1, pair_step=2, print_max=100) -> dict:
    '''Finds and prints the columns where each pair of rows of data are mismatched (Of different value, ignoring NaN)
    
    Arguments:
        index: [start, end] index in the DataFrame to start finding pairs of rows with different columns
        column_index: [start, end] index of the columns to find differences
        step: Distance between the first row and the second row in a pair of rows to compare
        pair_step: Distance between the first row in a pair of rows, and the first row in the next pair
        print_max: Number of differences this function will print before stopping (0 to disable print)
    
    Notes: 
        For the end index of (index) and (column_index), enter -1 to loop till last index. For (print_max), enter -1 for infinite
        Does not assert for any invalid values entered as arguments
    '''
    # If any end index was given negative 1, set it to the size of the respective axis
    if index is None:
        index = [0, df.shape[0]]
    if column_index is None:
        column_index = [0, df.shape[1]]

    printed = 0
    different_pairs_dict = dict() 
    # i and j, where they are the index of the first row and second row in a pair respectively
    for i, j in zip(range(*index, pair_step), range(step, index[1], pair_step)):
        
        # This list is reset every time we move on to a new pair
        different_column_list = list()
        # k is the index of the column that is currently being checked
        for k in range(*column_index):
            if pd.isnull(df.iloc[i, k]) and pd.isnull(df.iloc[j, k]):
                continue
                
            # If a mismatch has been found
            if df.iloc[i, k] != df.iloc[j, k]:
                
                # Track the names of the columns with mismatched values
                different_column_list.append(df.columns[k])
                
                if printed != print_max:
                    print('Rows {} and {} have a mismatched value'.format(i, j), 'at column', k, df.columns[k])
                    printed += 1
                    
        # If there were different column values detected earlier, log into different_pairs_dict
        if len(different_column_list) != 0:
            different_pairs_dict[i] = different_column_list
        
    if print_max == 0:
        pass
    elif printed == print_max:
        print('...')
        
    return different_pairs_dict


def plot_scree_plot(pca, plot_cumul=True, figsize=None, ax=None):
    '''For visualising percentage of variance explained by each PC
    '''
    
    if figsize is None:
        figsize = (12, 8)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    pca_var = pca.explained_variance_ratio_
    
    ax.set_xlim(1, len(pca_var))
    ax.set_ylim(0, 1)
    ax.plot(list(range(1, len(pca_var) + 1)), pca_var, '-o')
    
    if plot_cumul:
        cumulative_pca_var = [sum(pca_var[:i]) for i in range(1, len(pca_var) + 1)]
        ax.plot(list(range(1, len(pca_var) + 1)), cumulative_pca_var, '-o')
        return pca_var, cumulative_pca_var
    
    else:
        return pca_var


def plot_2d(df_pca, labels, columns_index: list=None, figsize=None, ax=None, **kwargs):
    '''Plots a 2D projection of given dataframe that has undergone PCA. Plots columns passed in
    
    Arguments:
        df_pca: A dataframe containing all the values, in PC axes, of each row of data
        labels: An iterable that contains the labels/groups that each data point falls under
        figsize: A tuple that decides size of the figure
        dpi: dpi of the figure to be drawn
    '''
    if figsize is None:
        figsize = (16, 12)
    
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    if columns_index is None:
        columns_index = [0, 1]
    
    ax.scatter(df_pca.iloc[:, columns_index[0]], df_pca.iloc[:, columns_index[1]], c=labels, **kwargs)
    ax.set_xlabel('PC' + str(columns_index[0]+1))
    ax.set_ylabel('PC' + str(columns_index[1]+1))


def plot_3d(df_pca, labels, columns_index: list=None, figsize=None, ax=None, **kwargs):
    '''Plots a 3D projection of given dataframe that has undergone PCA. Plots columns passed in
    
    Arguments:
        df_pca: A dataframe containing all the values, in PC axes, of each row of data
        labels: An iterable that contains the labels/groups that each data point falls under
        figsize: A tuple that decides size of the figure
        dpi: dpi of the figure to be drawn
    '''
    if figsize is None:
        figsize = (16, 12)
    
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

    if columns_index is None:
        columns_index = [0, 1, 2]
    
    ax.scatter(df_pca.iloc[:, columns_index[0]], df_pca.iloc[:, columns_index[1]], df_pca.iloc[:, columns_index[2]], c=labels, **kwargs)
    ax.set_xlabel('Component' + str(columns_index[0]+1))
    ax.set_ylabel('Component' + str(columns_index[1]+1))
    ax.set_zlabel('Component' + str(columns_index[2]+1))


def plot_kmeans_inertia(X, range_n_clusters, figsize=None, dpi=80, get_list=False):
    '''Plots a graph of inertia for KMeans, for elbow method for choosing of n_cluster k parameter
    
    Returns:
        prediction_label_list: List of all prediction labels generated by each n_cluster
        centroids_list: List of all centroids generated
        inertia_list: List of all inertia values generated
    '''
    prediction_label_list = [None] * len(range_n_clusters)
    centroids_list = [None] * len(range_n_clusters)
    inertia_list = [None] * len(range_n_clusters)
    for index, n_clusters in enumerate(range_n_clusters):
        model = KMeans(n_clusters=n_clusters)
        prediction_label_list[index] = model.fit_predict(X)
        centroids_list[index] = model.cluster_centers_
        inertia_list[index] = model.inertia_
    
    
    if figsize is None:
        figsize = (8, 4)
        
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range_n_clusters, inertia_list)
    ax.set_xlabel(r'Number of clusters (k)')
    ax.set_ylabel(r'Inertia')
    
    if get_list:
        return prediction_label_list, centroids_list, inertia_list


def plot_kmeans_silhouette_scores(X, list_n_clusters):
    '''Plots multiple silhouette scores for the range of n_clusters to try for kmeans
    '''
    fig = plt.figure(figsize=(12, 8), dpi=80)
    ax = fig.add_subplot(111)
    silhouette_avg_list = list()
    
    for n_clusters in list_n_clusters:
        kmeans = KMeans(n_clusters)
    
        cluster_labels = kmeans.fit_predict(X)
        
        silhouette_avg_list.append(silhouette_score(X, cluster_labels))
        
    ax.plot(list_n_clusters, silhouette_avg_list)

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
    

def plot_percentage_range_hist(df, column, lower_percent=0.0, upper_percent=0.2, bins=30, figsize=None, ax=None):
    if figsize is None:
        figsize = (12, 8)
        
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    sub_sample = count_percentage_range(df, column, lower_percent=lower_percent, upper_percent=upper_percent)[column]
    
    if bins != 0:
        bins = np.linspace(min(sub_sample), max(sub_sample), bins)
        ax.hist(sub_sample, bins=bins)
    else:
        ax.scatter(list(range(1, len(sub_sample) + 1)), sub_sample)
        
        
def plot_bounded_range_scatter(df, column, lower_bound=None, upper_bound=None, figsize=None, ax=None):
    if lower_bound is None:
        lower_bound = min(df[column])
    if upper_bound is None:
        upper_bound = max(df[column])
    if figsize is None:
        figsize = (12, 8)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
    sub_sample = count_bounded_range(df, column, lower_bound=lower_bound, upper_bound=upper_bound)[column]

    ax.scatter(list(range(1, len(sub_sample) + 1)), sub_sample.sort_values())
