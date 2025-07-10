import pandas as pd
import numpy as np

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