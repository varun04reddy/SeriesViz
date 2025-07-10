import pandas as pd
import numpy as np

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
