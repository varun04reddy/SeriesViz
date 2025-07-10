import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import plotly.io as pio

# 1) Read CSV and build DataFrame, setting timestamp as the index
sanity_check = pd.read_csv('data/sanity_check_data.csv')
sanity_check['timestamp'] = pd.to_datetime(sanity_check['SCHEDULED_TS'])
returnsDF = pd.DataFrame({
    'nifty_returns': sanity_check['normalized_returns_UNDERLYING_NIFTY'],
    'es_returns':    sanity_check['normalized_returns_UNDERLYING_ESMINI']
}, index=sanity_check['timestamp'])

# 2) rolling_beta_from_df with NaN handling
def rolling_beta_from_df(df, nifty_col='nifty_returns', es_col='es_returns', window=20):
    betas = []
    for i in range(window, len(df)):
        window_df = df.iloc[i-window:i]
        if window_df[nifty_col].isna().any() or window_df[es_col].isna().any():
            betas.append(np.nan)
        else:
            X = window_df[es_col].values.reshape(-1, 1)
            y = window_df[nifty_col].values
            betas.append(LinearRegression().fit(X, y).coef_[0])
    return pd.Series([np.nan]*window + betas, index=df.index)

# 3) compute_divergence_from_df
def compute_divergence_from_df(df, beta_window=20, corr_window=20,
                               nifty_col='nifty_returns', es_col='es_returns'):
    df = df.copy()
    df['beta']         = rolling_beta_from_df(df, nifty_col, es_col, beta_window)
    df['expected_nifty'] = df['beta'] * df[es_col]
    df['spread']       = df[nifty_col] - df['expected_nifty']
    df['rolling_corr'] = df[nifty_col].rolling(corr_window).corr(df[es_col])
    df['corr_past']    = df['rolling_corr'].shift(corr_window)
    df['es_reversal']  = df[es_col] * df[es_col].shift(1) < 0
    df['signal']       = (
        (df['corr_past'] > 0.6) &
        (df['rolling_corr'] < 0.2) &
        (df['spread'].abs() > df['spread'].std(skipna=True)) &
        df['es_reversal'] &
        (df[nifty_col] * df[es_col] < 0)
    )
    return df

# 4) Compute metrics
beta_window = 20
corr_window = 20
df_sig = compute_divergence_from_df(returnsDF, beta_window, corr_window)

# 5) Slice off the all-NaN head (OPTIONAL but recommended)
start_idx = max(beta_window, corr_window)
df_plot = df_sig.iloc[start_idx:]

# 6) Plot with Plotly and open in browser
pio.renderers.default = "browser"
fig = go.Figure()

# Lines with connectgaps=True so isolated NaNs don’t break the line
fig.add_trace(go.Scatter(
    x=df_plot.index, y=df_plot['beta'],
    mode='lines', name=f'β ({beta_window})',
    connectgaps=True
))
fig.add_trace(go.Scatter(
    x=df_plot.index, y=df_plot['spread'],
    mode='lines', name='Spread',
    connectgaps=True
))
fig.add_trace(go.Scatter(
    x=df_plot.index, y=df_plot['rolling_corr'],
    mode='lines', name=f'Corr ({corr_window})',
    connectgaps=True
))
# Signal markers
fig.add_trace(go.Scatter(
    x=df_plot.index[df_plot['signal']],
    y=[0] * df_plot['signal'].sum(),
    mode='markers', name='Divergence Signal',
    marker=dict(symbol='x', size=10)
))

fig.update_layout(
    title="Rolling β, Spread, Correlation & Divergence Signal",
    xaxis_title="Timestamp",
    yaxis_title="Value",
    template="plotly_white",
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
)

fig.show()
