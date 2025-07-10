import pandas as pd
import numpy as np

def calculate_rolling_correlation(df, columns, window_size, use_r_squared=False):
    """Calculate rolling correlation between variables"""
    if len(columns) < 2:
        return None
    
    correlations = {}
    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns[i+1:], i+1):
            if col1 in df.columns and col2 in df.columns:
                # Calculate rolling correlation
                rolling_corr = df[col1].rolling(window=window_size).corr(df[col2])
                if use_r_squared:
                    # Square the correlation to get R²
                    rolling_corr = rolling_corr ** 2
                correlations[f"{col1} vs {col2}"] = rolling_corr
    
    return correlations

def calculate_weighted_correlation(df, columns, window_size=20, weight_type='magnitude'):
    """Calculate weighted correlation between variables, giving more weight to larger movements."""
    if len(columns) < 2:
        return None
    
    correlations = {}
    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns[i+1:], i+1):
            if col1 in df.columns and col2 in df.columns:
                # Calculate rolling weighted correlation
                rolling_weighted_corr = df[col1].rolling(window=window_size).apply(
                    lambda x: calculate_weighted_corr_pair(x, df[col2].loc[x.index], weight_type)
                )
                correlations[f"{col1} vs {col2}"] = rolling_weighted_corr
    
    return correlations

def calculate_weighted_corr_pair(x, y, weight_type='magnitude'):
    """Calculate weighted correlation for a pair of series."""
    if len(x) < 2 or len(y) < 2:
        return np.nan
    
    # Remove NaN values
    valid_mask = ~(x.isna() | y.isna())
    x_clean = x[valid_mask]
    y_clean = y[valid_mask]
    
    if len(x_clean) < 2:
        return np.nan
    
    # Calculate weights based on the specified type
    if weight_type == 'magnitude':
        # Weight by sum of absolute values
        weights = np.abs(x_clean) + np.abs(y_clean)
    elif weight_type == 'volatility':
        # Weight by rolling volatility (standard deviation)
        weights = np.sqrt(x_clean.var() + y_clean.var())
        weights = np.full_like(x_clean, weights)
    elif weight_type == 'change_magnitude':
        # Weight by magnitude of changes (first differences)
        x_diff = np.abs(x_clean.diff().fillna(0))
        y_diff = np.abs(y_clean.diff().fillna(0))
        weights = x_diff + y_diff
    else:
        # Default to equal weights
        weights = np.ones_like(x_clean)
    
    # Normalize weights to sum to 1
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones_like(x_clean) / len(x_clean)
    
    # Calculate weighted means
    x_mean = np.average(x_clean, weights=weights)
    y_mean = np.average(y_clean, weights=weights)
    
    # Calculate weighted correlation
    numerator = np.sum(weights * (x_clean - x_mean) * (y_clean - y_mean))
    x_var = np.sum(weights * (x_clean - x_mean) ** 2)
    y_var = np.sum(weights * (y_clean - y_mean) ** 2)
    
    denominator = np.sqrt(x_var * y_var)
    
    if denominator == 0:
        return np.nan
    
    return numerator / denominator

def calculate_bucket_correlation_metrics(df, variable_cols, bucket_labels, bucket_col, corr_window_size, use_r_squared=False):
    """Calculate correlation metrics (min, mean, std, max, avg abs, count) for each variable pair and custom bucket using rolling correlation."""
    # Ensure bucket_labels is a list
    if not isinstance(bucket_labels, list):
        bucket_labels = list(bucket_labels)
    
    corr_metrics = {}
    if use_r_squared:
        corr_metrics_list = ['Min', 'Mean', 'Std', 'Max', 'Avg Abs', 'Count']
    else:
        corr_metrics_list = ['Min', 'Mean', 'Std', 'Max', 'Avg Abs', 'Count']
    var_pairs = [(variable_cols[i], variable_cols[j]) for i in range(len(variable_cols)) for j in range(i+1, len(variable_cols))]
    debug_hour_counts = []
    for (var1, var2) in var_pairs:
        for bucket in bucket_labels:
            bucket_data = df[df[bucket_col] == bucket]
            debug_hour_counts.append((bucket, var1, var2, len(bucket_data)))
            if len(bucket_data) >= corr_window_size:
                rolling_corr = bucket_data[var1].rolling(window=corr_window_size).corr(bucket_data[var2])
                rolling_corr = rolling_corr.dropna()
                if len(rolling_corr) > 0:
                    if use_r_squared:
                        # Square the correlation values to get R²
                        rolling_corr = rolling_corr ** 2
                    corr_metrics[((var1, var2), bucket)] = [
                        rolling_corr.min(),
                        rolling_corr.mean(),
                        rolling_corr.std(),
                        rolling_corr.max(),
                        rolling_corr.abs().mean(),
                        len(rolling_corr)
                    ]
                else:
                    corr_metrics[((var1, var2), bucket)] = [None]*5 + [0]
            else:
                corr_metrics[((var1, var2), bucket)] = [None]*5 + [0]
    # Create MultiIndex DataFrame for correlations
    corr_index = pd.MultiIndex.from_product([var_pairs, corr_metrics_list], names=["Variable Pair", "Metric"])
    corr_columns = bucket_labels
    corr_data = []
    for pair in var_pairs:
        for m_idx, metric in enumerate(corr_metrics_list):
            row = []
            for bucket in bucket_labels:
                val = corr_metrics[(pair, bucket)][m_idx]
                if metric == 'Count':
                    row.append(0 if val is None else int(val))
                else:
                    row.append(None if val is None else round(val, 4))
            corr_data.append(row)
    corr_metrics_df = pd.DataFrame(corr_data, index=corr_index, columns=corr_columns)
    return corr_metrics_df, debug_hour_counts