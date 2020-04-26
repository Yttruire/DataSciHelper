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
    '''Returns a list of [x, y, corr_value[, p_value]] where [x, y] are the coordinates of the value on the correlation matrix
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to find correlations in

    threshold : float, default=0.3
        The abs(threshold) at which to flag a relationship between two features as being correlated
        
    print_corr : bool, default=True
        Prints the results of the correlation finder if True
    
    get_list : bool, default=False
        Returns the list of correlations in the format [x, y, corr_value, p_value]
        p_value is only returned if p_value=True

    p_value : bool, default=True
        Prints the corresponding p value of a correlation if print_corr=True, and adds it to the list returned if get_list=True

    method : {'pearson', 'spearman', 'kendall'} or callable, default='pearson'
        The method with which to calculate the correlation matrix

    Returns
    -------
    corr_list: list or None, shape (n_correlations, 3 or 4)
        List is returned only upon request via get_list in the format [x, y, corr_value[, p_value]]
        n_correlations = number of correlations found with the given threshold
        p_value is optional. Returned only if given argument p_value is True
        
    Notes
    -----
    Only accepts DataFrames without categorical features (Or has been OneHotEncoded properly)
    Checks through correlations of the bottom left triangle of the correlation matrix
    If categorical features are present, coordinates x and y will no longer reflect the correct column coordinates as in df.columns
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
    
    Parameters
    ----------
    threshold : int, float or list
        When threshold is int or float, colors values >= threshold green, >= 1.5 threshold blue, >= 2 threshold red
        When a list, it will be taken as it is for green, blue then red respectively
    
    inverse : bool, default=False
        When True, colors values < threshold green, < 1.5 threshold blue, < 2 threshold red

    Notes
    -----
    Pass a list of shape (1, 3) for the threshold to prevent above behaviour. 
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


def get_zoutlier(df, column_to_test, threshold=4, style=True):
    ''' Returns a dataframe of all outlier values in a single column according to given threshold (In sigma) using Z test
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe from which values that exceed the given threshold are found

    column_to_test : str
        The name of the column to be tested in df

    threshold : int or float, default=4
        The threshold to flag out a value as an outlier 
        The units are in sigma, the standard deviation of a Z-test
        E.g. threshold=2 will flag all values within 2 standard deviations of the mean

    style : bool
        True to return a styled DataFrame. False for a normal DataFrame
    '''

    # Apply standard scaling to the column
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[column_to_test].values.reshape(-1, 1))

    # Get list of indices where its value exceeds the reshold
    outlier_list = [index for index, value in enumerate(scaled_values) if abs(value) > threshold]

    # Construct a dataframe containing both the original and scaled values that were flagged as outliers
    df_outlier = pd.DataFrame(scaler.inverse_transform(scaled_values[outlier_list]), columns=[column_to_test])
    df_outlier[column_to_test + '_scaled'] = scaled_values[outlier_list]
    df_outlier.index = outlier_list
    
    # Return styled dataframe if requested, normal dataframe if not
    if style:
        return df_outlier.style.applymap(get_color_map(threshold), subset=pd.IndexSlice[:, column_to_test + '_scaled'])
    else:
        return df_outlier
    
    
def get_multicolumn_zoutlier(df, columns_to_test=None, threshold=4, style=True):
    ''' Returns a dataframe of all outlier values in multiple columns according to a given threshold (In sigma) using Z test
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe from which values that exceed the given threshold are found

    columns_to_test : list or None, default=None
        A list of the names of the columns to be tested in df
        If None, it will go through all of the columns

    threshold : int or float, default=4
        The threshold to flag out a value as an outlier 
        The units are in sigma, the standard deviation of a Z-test
        E.g. threshold=2 will flag all values within 2 standard deviations of the mean

    style : bool, default=True
        True to return a styled DataFrame. False for a normal DataFrame
    '''

    # If no column specified, go through all columns
    if columns_to_test is None:
        columns_to_test = df.columns

    df_outliers = pd.DataFrame()
    
    # Go through all columns and concatenate the results
    for column in columns_to_test:
        df_outliers = pd.concat([df_outliers, get_zoutlier(df, column_to_test=column, threshold=threshold, style=False)], axis=1, sort=False)
    
    # Return styled dataframe is requested, normal dataframe if not
    if style:
        return df_outliers.style.applymap(get_color_map(threshold), subset=pd.IndexSlice[:, [column + '_scaled' for column in columns_to_test]])
    else:
        return df_outliers



def get_df_missing_data(df):
    '''Returns a DataFrame documenting all features with missing data (NaN)

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing the features with missing data to be counted and then returned by the function in a new DataFrame
        
    Returns
    -------
    df_missing_data : pd.DataFrame of shape (n_features_with_missing, 4)
        Contains 4 columns with:
        - name of column with missing data
        - index of column with missing data
        - number of missing data in column
        - percentage of missing data in column
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


def find_diff_pairs_rows(df, index_range=None, column_index_range=None, step=1, pair_step=2, print_max=100, get_dict=False):
    '''Finds and prints the columns where each pair of rows of data are mismatched (Of different value, ignoring NaN)
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe from which to find columns where a pair of rows of data are mismatched

    index_range : None or list, of shape (2), default=None
        [start, end] index in the dataframe to start finding pairs of rows with different columns (axis=0)
        Inclusive of start, non-inclusive of end
        If None, go through all rows in the dataframe
        Total number of rows to compare must be even

    column_index_range : None or list, of shape (2), default=None
        [stard, end] index of the columns to compare and find differences
        Inclusive of start, non-inclusive of end
        If None, go through all columns in the dataframe

    step : int, default=1
        Distance between the first row and the second row in a pair of rows to compare

    pair_step : int, default=2
        Distance between the first row in a pair of rows, and the first row in the next pair
    
    print_max : int, default=100
        Number of differences this function will print before stopping
        0 (or any negative number) will disable all prints

    get_dict : bool, default=False
        If True, construct and return the dictionary of mismatched rows and the corresponding columns

    Returns
    -------
    different_pairs_dict : dict
        The key is the index of the first row of a pair that had mismatched columns
        The value is a list of column indices 
    '''


    # If given range is None, go through all available rows/columns respectively
    if index_range is None:
        index_range = [0, df.shape[0]]
    if column_index_range is None:
        column_index_range = [0, df.shape[1]]

    # Check that the total number of rows is even so that the rows can be divided into pairs to compare columns.
    assert (index_range[1] - index_range[0] + 1) % 2 == 0, 'Number of rows must be even in order to be compared in pairs'

    different_pairs_count = 0
    different_pairs_dict = dict() 

    # first_row and second_row, are the index of the first row and second row in a pair respectively
    for first_row, second_row in zip(range(*index_range, pair_step), range(step, index_range[1], pair_step)):
        
        # This list is reset every time we move on to a new pair
        different_column_list = list()

        # col_index is the index of the column that is currently being checked
        for col_index in range(*column_index_range):
            if pd.isnull(df.iloc[first_row, col_index]) and pd.isnull(df.iloc[second_row, col_index]):
                continue
                
            # If a mismatch has been found
            if df.iloc[first_row, col_index] != df.iloc[second_row, col_index]:
                
                # Track the names of the columns with mismatched values
                different_column_list.append(col_index)
                
        # Print all the mismatched columns for this pair of rows as long as we have not yet reached the limit to print     
        if different_pairs_count < print_max:
            print('Rows {} and {} have a mismatched value'.format(first_row, second_row), 'at columns', different_column_list, df.columns[different_column_list])

        # If a pair of rows with different columns was found, increment the counter
        if len(different_column_list) != 0:
            different_pairs_count += 1
                    
        # If there were different column values detected earlier, log into different_pairs_dict. Log only if the dict is requested
        if len(different_column_list) != 0 and get_dict == True:
            different_pairs_dict[first_row] = different_column_list
    
    # If there were more different pairs found than printed, and printing was enabled (By having print_max > 0) 
    # print '...' to indicate more pairs were found than printed
    if different_pairs_count > print_max and print_max > 0:
        print('...')
        
    # Return the dictionary only if requested
    if get_dict == True:
        return different_pairs_dict


def plot_scree_plot(pca, plot_cumul=True, figsize=None, ax=None):
    '''For visualising percentage of variance explained by each PC

    Parameters
    ----------
    pca : PCA
        Principal Component Analysis object form Sci-kit Learn

    plot_cumul : bool, default=True
        Whether to also plot the cumulative line of explained variance

    figsize : tuple, default=None
        Defaults to (12, 8)
        As per matplotlib

    ax : axes
        The Axes object on which to plot the scree plot

    Returns
    -------
    pca_var : array, shape (n_components)
        An array of the explained variance ratio

    cumulative_pca_var : array, shape (n_components)
        The cumulated explained variance ratio
    '''
    
    if figsize is None:
        figsize = (12, 8)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    pca_var = pca.explained_variance_ratio_
    
    # Set limits and plot explained variance ratio
    ax.set_xlim(1, len(pca_var))
    ax.set_ylim(0, 1)
    ax.plot(list(range(1, len(pca_var) + 1)), pca_var, '-o')
    
    # If requested, plot the cumulative explained variance ratio
    if plot_cumul:
        cumulative_pca_var = [sum(pca_var[:i]) for i in range(1, len(pca_var) + 1)]
        ax.plot(list(range(1, len(pca_var) + 1)), cumulative_pca_var, '-o')
        return pca_var, cumulative_pca_var
    # If cumulative plot was not requested, return only the explained variance ratio
    else:
        return pca_var


def plot_2d(df_pca, labels, columns_index=None, figsize=None, ax=None, **kwargs):
    '''Plots a 2D projection of given dataframe. Plots the columns passed in
    
    Can take dataframes of any shape, but will always plot a 2D scatter of the indicated columns
    Originally intended for dataframes that have undergone dimentionality reduction

    Parameters
    ----------
    df_pca : pd.DataFrame
        A dataframe containing all the values, in PC axes, of each row of data
    
    labels : array-like
        An iterable that contains the labels/groups that each data point falls under
        Used to color different points

    columns_index : list, shape (3), default=None
        Defaults to [0, 1]
    
    figsize : tuple, shape (2), default=None
        Defaults to (12, 8)
        As per matplotlib

    ax : Axes, default=None
        The axes on which to plot the scatter plot
        If None, it will create its own plot and axis
        
    **kwargs :
        Any keyword argument passed to axes.scatter
    '''
    if figsize is None:
        figsize = (16, 12)
    
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    if columns_index is None:
        columns_index = [0, 1]
    
    ax.scatter(df_pca.iloc[:, columns_index[0]], df_pca.iloc[:, columns_index[1]], c=labels, **kwargs)
    ax.set_xlabel('Component' + str(columns_index[0]+1))
    ax.set_ylabel('Component' + str(columns_index[1]+1))


def plot_3d(df_pca, labels, columns_index=None, figsize=None, ax=None, **kwargs):
    '''Plots a 3D projection of given dataframe. Plots the columns passed in

    Can take dataframes of any shape, but will always plot a 3D scatter of the indicated columns
    Originally intended for dataframes that have undergone dimentionality reduction

    Parameters
    ----------
    df_pca : pd.DataFrame
        A dataframe containing all the values, in PC axes, of each row of data
    
    labels : array-like
        An iterable that contains the labels/groups that each data point falls under
        Used to color different points

    columns_index : list, shape (3), default=None
        Defaults to [0, 1, 2]
    
    figsize : tuple, shape (2), default=None
        Defaults to (12, 8)
        As per matplotlib

    ax : Axes, default=None
        The axes on which to plot the scatter plot
        If None, it will create its own plot and axis
        
    **kwargs :
        Any keyword argument passed to axes.scatter
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
