import pandas as pd
import numpy as np

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

def calculate_spread_metrics(df, variable_cols):
    """Calculate spread metrics between variable pairs at each timestamp."""
    spread_metrics = {}
    
    # Create pairs of variables to compare
    var_pairs = [(variable_cols[i], variable_cols[j]) for i in range(len(variable_cols)) for j in range(i+1, len(variable_cols))]
    
    for (var1, var2) in var_pairs:
        if var1 in df.columns and var2 in df.columns:
            # Calculate spread between the two variables at each timestamp
            var1_values = df[var1].dropna()
            var2_values = df[var2].dropna()
            
            if len(var1_values) > 0 and len(var2_values) > 0:
                # Get common timestamps where both variables have data
                common_index = var1_values.index.intersection(var2_values.index)
                if len(common_index) > 0:
                    var1_common = var1_values.loc[common_index]
                    var2_common = var2_values.loc[common_index]
                    
                    # Calculate spread metrics between the two variables
                    # Absolute difference
                    abs_diff = abs(var1_common - var2_common)
                    
                    # Relative difference (as percentage of mean)
                    mean_val = (var1_common + var2_common) / 2
                    rel_diff = abs_diff / mean_val.where(mean_val != 0, 1) * 100
                    
                    # Ratio (larger value / smaller value)
                    ratio = np.maximum(var1_common, var2_common) / np.minimum(var1_common, var2_common).where(np.minimum(var1_common, var2_common) != 0, 1)
                    
                    # Standardized difference (z-score difference)
                    overall_std = np.sqrt(var1_common.var() + var2_common.var())
                    std_diff = abs_diff / overall_std if overall_std != 0 else 0
                    
                    spread_metrics[f"{var1} vs {var2}"] = {
                        'Absolute Difference': abs_diff,
                        'Relative Difference (%)': rel_diff,
                        'Ratio': ratio,
                        'Standardized Difference': std_diff
                    }
    
    return spread_metrics

def calculate_bucket_spread_metrics(df, variable_cols, bucket_labels, bucket_col):
    """Calculate spread metrics statistics (min, mean, max, std) for variable pairs in each bucket."""
    # Ensure bucket_labels is a list
    if not isinstance(bucket_labels, list):
        bucket_labels = list(bucket_labels)
    
    # Create pairs of variables to compare
    var_pairs = [(variable_cols[i], variable_cols[j]) for i in range(len(variable_cols)) for j in range(i+1, len(variable_cols))]
    
    spread_metrics = {}
    metrics = ['Min', 'Mean', 'Max', 'Std']
    
    for (var1, var2) in var_pairs:
        for bucket in bucket_labels:
            bucket_data = df[df[bucket_col] == bucket]
            
            if len(bucket_data) > 0:
                # Get values for both variables in this bucket
                var1_values = bucket_data[var1].dropna()
                var2_values = bucket_data[var2].dropna()
                
                # Find common timestamps where both variables have data
                common_index = var1_values.index.intersection(var2_values.index)
                
                if len(common_index) > 0:
                    var1_common = var1_values.loc[common_index]
                    var2_common = var2_values.loc[common_index]
                    
                    # Calculate absolute difference between variables
                    abs_diff = abs(var1_common - var2_common)
                    
                    if len(abs_diff) > 0:
                        spread_metrics[((var1, var2), bucket)] = [
                            abs_diff.min(),
                            abs_diff.mean(),
                            abs_diff.max(),
                            abs_diff.std()
                        ]
                    else:
                        spread_metrics[((var1, var2), bucket)] = [None, None, None, None]
                else:
                    spread_metrics[((var1, var2), bucket)] = [None, None, None, None]
            else:
                spread_metrics[((var1, var2), bucket)] = [None, None, None, None]
    
    # Create MultiIndex DataFrame for spread metrics
    spread_index = pd.MultiIndex.from_product([var_pairs, metrics], names=["Variable Pair", "Metric"])
    spread_columns = bucket_labels
    spread_data = []
    for pair in var_pairs:
        for m_idx, metric in enumerate(metrics):
            row = []
            for bucket in bucket_labels:
                val = spread_metrics[(pair, bucket)][m_idx]
                row.append(None if val is None else round(val, 4))
            spread_data.append(row)
    spread_metrics_df = pd.DataFrame(spread_data, index=spread_index, columns=spread_columns)
    return spread_metrics_df

def calculate_bucket_raw_metrics(df, variable_cols, bucket_labels, bucket_col, use_spread_analysis=False):
    """Calculate raw metrics (min, mean, std, max, count) and spread metrics for each variable and custom bucket."""
    # Ensure bucket_labels is a list
    if not isinstance(bucket_labels, list):
        bucket_labels = list(bucket_labels)
    
    raw_metrics = {}
    if use_spread_analysis:
        metrics = ['Min', 'Mean', 'Std', 'Max', 'Count', 'CV', 'Range', 'IQR', 'Variance']
    else:
        metrics = ['Min', 'Mean', 'Std', 'Max', 'Count']
    
    for col in variable_cols:
        for bucket in bucket_labels:
            bucket_data = df[df[bucket_col] == bucket][col].dropna()
            if len(bucket_data) > 0:
                # Basic metrics
                basic_metrics = [
                    bucket_data.min(),
                    bucket_data.mean(),
                    bucket_data.std(),
                    bucket_data.max(),
                    len(bucket_data)
                ]
                
                if use_spread_analysis:
                    # Calculate spread metrics
                    mean_val = bucket_data.mean()
                    std_val = bucket_data.std()
                    q25 = bucket_data.quantile(0.25)
                    q75 = bucket_data.quantile(0.75)
                    
                    cv = std_val / mean_val if mean_val != 0 else None
                    range_val = bucket_data.max() - bucket_data.min()
                    iqr = q75 - q25
                    variance = std_val ** 2
                    
                    spread_metrics = [cv, range_val, iqr, variance]
                    raw_metrics[(col, bucket)] = basic_metrics + spread_metrics
                else:
                    raw_metrics[(col, bucket)] = basic_metrics
            else:
                if use_spread_analysis:
                    raw_metrics[(col, bucket)] = [None, None, None, None, 0, None, None, None, None]
                else:
                    raw_metrics[(col, bucket)] = [None, None, None, None, 0]
    
    # Create MultiIndex DataFrame
    raw_index = pd.MultiIndex.from_product([variable_cols, metrics], names=["Variable", "Metric"])
    raw_columns = bucket_labels
    raw_data = []
    for var in variable_cols:
        for m_idx, metric in enumerate(metrics):
            row = []
            for bucket in bucket_labels:
                val = raw_metrics[(var, bucket)][m_idx]
                if metric == 'Count':
                    row.append(0 if val is None else int(val))
                else:
                    row.append(None if val is None else round(val, 4))
            raw_data.append(row)
    raw_metrics_df = pd.DataFrame(raw_data, index=raw_index, columns=raw_columns)
    return raw_metrics_df

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