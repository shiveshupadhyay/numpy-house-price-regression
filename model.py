"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
import numpy as np
def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """
    col_mean = np.nanmean(X, axis = 0)
    col_mean = np.nan_to_num(col_mean, nan = 0.0)
    X = np.nan_to_num(X, nan = col_mean)

    return X

# Step 2 - compute_iqr_bounds
import numpy as np
def compute_iqr_bounds(X, k=1.5):
    col_q25 = np.quantile(X, 0.25, axis = 0)
    col_q75 = np.quantile(X, 0.75, axis = 0)

    iqr = col_q75 - col_q25

    lower_bound = col_q25 - k * iqr
    upper_bound = col_q75 + k * iqr 

    return (lower_bound, upper_bound)

# Step 3 - clip_columns
def clip_columns(X, lower, upper):
    lower = lower.reshape(1,-1)
    upper = upper.reshape(1,-1)
    lower_mask = (X < lower).astype(int)
    upper_mask = (X > upper).astype(int)
    range_mask = ((X >= lower) & (X <= upper)).astype(int)

    masked_X = range_mask * X + lower_mask * lower + upper_mask * upper
    return masked_X

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    ratio_feature = numerator/(denominator + eps)

    return ratio_feature

# Step 5 - append_column
def append_column(X, col):
    col = col.reshape(-1, 1)
    extended_X = np.hstack((X, col))
    return extended_X

# Step 6 - one_hot_encode
def one_hot_encode(labels):
    N = len(labels)

    unique_label, counts = np.unique(labels, return_counts=True)
    unique_label = unique_label.reshape(1, -1)

    pre_one_hot_matrix = np.tile(unique_label, (N, 1))
    labels = labels.reshape(N, -1)

    one_hot_matrix = (pre_one_hot_matrix == labels).astype(float)

    return one_hot_matrix

# Step 7 - fit_standardizer
def fit_standardizer(X):
    col_mean = np.mean(X, axis = 0)
    col_std = np.std(X, axis = 0)
    col_std[col_std == 0] = 1.0
    
    return (col_mean, col_std)

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    mean = mean.reshape(1, -1)
    std = std.reshape(1, -1)
    standard_X = (X - mean) / std 

    return standard_X

# Step 9 - add_bias_column
def add_bias_column(X):
    N, F = X.shape 
    bias = np.ones(N).reshape(-1,1)

    return np.hstack((bias, X))

# Step 10 - make_shuffled_indices
import numpy as np
def make_shuffled_indices(n_samples, seed):
    np.random.seed(seed)
    return np.random.permutation(n_samples)

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):
    n = len(indices)
    train_end = int(np.floor(train_ratio * n))
    val_end = int(np.floor((train_ratio + val_ratio) * n))

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    return train_indices, val_indices, test_indices

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    X_sub = X[indices, : ]
    y_sub = y[indices]

    return (X_sub, y_sub)

# Step 13 - ols_fit
def ols_fit(X, y):
    weights = np.linalg.pinv(X.T @ X) @ X.T @ y
    return weights

# Step 14 - ols_predict
def ols_predict(X, theta):
    
    return X @ theta

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    mae = np.mean(np.abs(y_pred - y_true))
    return mae

# Step 16 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    """Compute root mean squared error between targets and predictions.

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (N,).
        y_pred (np.ndarray): Predicted targets, shape (N,).

    Returns:
        float: RMSE value.
    """
    rmse = np.sqrt(np.mean((y_pred- y_true)**2))
    return rmse

# Step 17 - r_squared
def r_squared(y_true, y_pred):
    mean_pred = np.mean(y_true)

    SST = np.sum((mean_pred - y_true)**2)
    SSR = np.sum((y_pred - y_true)**2)
    r2 = 0 if SST == 0 else (1 - SSR/SST) 
    return r2

# Step 18 - residual_summary
def residual_summary(y_true, y_pred):
    error = y_true - y_pred
    abs_error = np.abs(error)
    mean = np.mean(error)
    std = np.std(error)
    median_abs = np.quantile(abs_error,0.50)

    return {
        'mean' : mean, 
        'std' : std, 
        'median_abs' : median_abs
    }

# Step 19 - prepare_cleaned_features
def prepare_cleaned_features(X, iqr_k=1.5):
    """Impute NaNs then IQR-clip columns to produce a clean numeric matrix.

    Args:
        X: (N, F) array-like of floats, may contain NaN.
        iqr_k: IQR multiplier passed to compute_iqr_bounds (default 1.5).

    Returns:
        (N, F) float ndarray with no NaNs, columns clipped to IQR bounds.
    """
    X = impute_nan_with_mean(X)
    lower_bound,upper_bound = compute_iqr_bounds(X, iqr_k) 

    X = clip_columns(X, lower_bound, upper_bound)

    return X

# Step 20 - assemble_feature_matrix
def assemble_feature_matrix(X_num, ratio_num_idx, ratio_den_idx, cat_labels=None):
    num_ratio_feature = X_num[:, ratio_num_idx]
    den_ratio_feature = X_num[:, ratio_den_idx]

    ratio_feature = make_ratio_feature(num_ratio_feature, den_ratio_feature, eps=1e-8)
    X = append_column(X_num, ratio_feature)

    if cat_labels is not None:
        cat_mat = one_hot_encode(cat_labels)
        X = np.hstack((X, cat_mat))
    return X

# Step 21 - make_train_val_test
def make_train_val_test(X, y, train_ratio, val_ratio, seed):
    N, F = X.shape
    indices = make_shuffled_indices(N, seed)
    train_indices, val_indices, test_indices = partition_indices(indices, train_ratio, val_ratio)
    X_train, y_train = subset_xy(X, y, train_indices)
    X_val, y_val = subset_xy(X, y, val_indices)
    X_test, y_test = subset_xy(X, y, test_indices)
    return {
        'X_train' : X_train,
        'y_train' : y_train,
        'X_val' : X_val,
        'y_val' : y_val,
        'X_test' : X_test,
        'y_test' : y_test
        }

# Step 22 - standardize_and_add_bias
def standardize_and_add_bias(splits):
    mean, std = fit_standardizer(splits['X_train'])
    std_splits = splits.copy()
    std_splits['X_train'] = apply_standardizer(splits['X_train'], mean, std)
    std_splits['X_val'] = apply_standardizer(splits['X_val'], mean, std)
    std_splits['X_test'] = apply_standardizer(splits['X_test'], mean, std)
    
    std_splits['X_train'] = add_bias_column(std_splits['X_train'])
    std_splits['X_val'] = add_bias_column(std_splits['X_val'])
    std_splits['X_test'] = add_bias_column(std_splits['X_test'])

    return (std_splits, mean, std)

# Step 23 - evaluate_predictions
def evaluate_predictions(y_true, y_pred):
    # TODO: Bundle MAE, RMSE, R^2, and residual summary into one metrics dict.
    mae = mean_absolute_error(y_true, y_pred)
    rmse =  root_mean_squared_error(y_true, y_pred)
    r2 = r_squared(y_true, y_pred)
    res_summary = residual_summary(y_true, y_pred)

    return {
        'mae' : mae, 
        "rmse" : rmse, 
        "r2" : r2, 
        "residual_summary" : res_summary
    }

# Step 24 - house_price_pipeline
def house_price_pipeline(X, y, ratio_num_idx, 
        ratio_den_idx, cat_labels=None, 
        train_ratio=0.7, val_ratio=0.15, 
        seed=42, iqr_k=1.5
        ):
    X = prepare_cleaned_features(X, iqr_k)
    X = assemble_feature_matrix(X, ratio_num_idx, 
            ratio_den_idx, cat_labels)
    splits = make_train_val_test(X, y, train_ratio, val_ratio, seed)

    std_splits, mean, std = standardize_and_add_bias(splits)
    weights = ols_fit(std_splits['X_train'], std_splits['y_train'])
    
    y_pred_test = ols_predict(std_splits['X_test'], weights)
    test_metrics = evaluate_predictions(std_splits['y_test'],y_pred_test)

    y_pred_val = ols_predict(std_splits['X_val'], weights)
    val_metrics = evaluate_predictions(std_splits['y_val'],y_pred_val)

    return {
       "theta" : weights, 
       "y_test" : std_splits['y_test'], 
       'y_test_pred' : y_pred_test, 
       'test_metrics' : test_metrics,
       'val_metrics' : val_metrics 
    }

