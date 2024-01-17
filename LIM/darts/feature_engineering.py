import pandas as pd
import numpy as np
import datetime
from sktime.transformations.series.date import DateTimeFeatures

import warnings
warnings.filterwarnings('ignore')


def create_shifted_columns(df, column, lags):
    shifted_df = df.copy()

    for lag in lags:
        shifted_df[column + "_lag_" + str(lag)] = df[column].shift(lag)

    return shifted_df


def create_rolling_mean_columns(df, columns, windows=[14, 28]):
    modified_df = df.copy()

    if type(columns) != list:
        columns = [columns]

    for window in windows:
        for column in columns:
            rolling_mean = df[column].rolling(window=window).mean()
            modified_df[column + "_rolling_mean_" + str(window)] = rolling_mean

    modified_df = modified_df.dropna()

    return modified_df


def create_time_based_features(df, target_col, freq=["M", "W"]):

    modified_df = df.copy()
    modified_df["date"] = df.index

    if "Y" in freq:
        modified_df["Year"] = modified_df["date"].dt.year
    if "M" in freq:
        modified_df["Month"] = modified_df["date"].dt.month
    if "W" in freq:
        modified_df["Week"] = modified_df["date"].dt.week
    if "D" in freq:
        modified_df["Day"] = modified_df["date"].dt.day

    '''
    series = modified_df[target_col]
    series.name = target_col
    series = series.resample('H').mean()

    hourly_feats = DateTimeFeatures(ts_freq='H',
                                    keep_original_columns=False,
                                    feature_scope='efficient')

    dtime_train = hourly_feats.fit_transform(series)
    df = pd.concat([modified_df, dtime_train],
                            axis=1)'''
    modified_df = modified_df.drop(columns=["date"])

    return modified_df


def create_expanding_window_features(df, columns):
    modified_df = df.copy()

    if type(columns) != list:
        columns = [columns]

    for column in columns:
        exp_window = modified_df[column].expanding()
        exp_mean = exp_window.mean()
        exp_std = exp_window.std()
        exp_min = exp_window.min()
        exp_max = exp_window.max()
        modified_df[column + "_expanding_mean"] = exp_mean
        modified_df[column + "_expanding_std"] = exp_std
        modified_df[column + "_expanding_min"] = exp_min
        modified_df[column + "_expanding_max"] = exp_max

    return modified_df


def create_rm_last_year(df, column):
    modified_df = df.copy()
    modified_df["date"] = df.index

    if "Year" not in modified_df.columns:
        modified_df["Year"] = modified_df["date"].dt.year
    if "Week" not in modified_df.columns:
        modified_df["Week"] = modified_df["date"].dt.week

    modified_df["last_year"] = modified_df["Year"] - 1

    # Compute average value for each week number and year combination
    weekly_avg_value = modified_df.groupby(['Week', 'Year'])[
        column].mean().reset_index().rename(columns={column: f'avg_{column}_last_year_week'})

    # Compute week number and year for same week last year
    weekly_avg_value['join_year'] = weekly_avg_value['Year']
    weekly_avg_value['last_week'] = weekly_avg_value['Week']

    # Merge average sales for same week last year to original data
    modified_df = pd.merge(modified_df, weekly_avg_value[
        ['join_year', 'last_week', f'avg_{column}_last_year_week']], how='left',
                           left_on=['Week', 'last_year'],
                           right_on=['last_week', 'join_year'])

    modified_df = modified_df.drop(columns=["date", "last_week", "join_year", "last_year"])

    return modified_df


def create_shifted_df(df, history, horizon, drop_nan=True, future_shift=False):

    for col in df:
        for i in range(1, history + 1):
            df[col + '_t-' + str(i)] = df[col].shift(-i)
        if future_shift:
            for i in range(1, horizon + 1):
                df[col + '_t+' + str(i)] = df[col].shift(i)

    if drop_nan:
        df = df.dropna()

    return df

def create_fourier_series_features(df, date_col, column):
    modified_df = df.copy()
    modified_df["date"] = pd.to_datetime(df[date_col])
    modified_df = modified_df.set_index("date")

    fourier_daily = FourierTerms(n_terms=2, period=24, prefix='D_')
    fourier_monthly = FourierTerms(n_terms=2, period=24 * 30.5, prefix='M_')
    fourier_yearly = FourierTerms(n_terms=2, period=24 * 365, prefix='Y_')

    dfourier_train = fourier_daily.transform(modified_df.index)
    mfourier_train = fourier_monthly.transform(modified_df.index)
    yfourier_train = fourier_yearly.transform(modified_df.index)

    modified_df = pd.concat([modified_df, mfourier_train, dfourier_train,
                             mfourier_train, yfourier_train],
                            axis=1)

    return modified_df


def correlation_filter(data: pd.DataFrame, target_col, corr_threshold: float = .9):

    data_cp = data.copy()
    # Absolute correlation matrix
    corr_matrix = data_cp.corr().abs()

    # Create a True/False mask and apply it
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    tri_df = corr_matrix.mask(mask)

    # List column names of highly correlated features (r > 0.95)
    corr_features = \
        [c for c in tri_df.columns
         if any(tri_df[c] > corr_threshold)]

    if target_col in corr_features:
        corr_features.remove(target_col)

    # Drop the features in the to_drop list
    data_subset = data.drop(corr_features, axis=1)

    return data_subset, corr_features


class FourierTerms:

    def __init__(self, period: float, n_terms: int, prefix=''):
        self.period = period
        self.n_terms = n_terms
        self.prefix = prefix

    def transform(self, index: pd.DatetimeIndex, use_as_index: bool = True):
        t = np.array(
            (index - datetime(1970, 1, 1)).total_seconds().astype(float)
        ) / (3600 * 24.)

        fourier_x = np.column_stack([
            fun((2.0 * (i + 1) * np.pi * t / self.period))
            for i in range(self.n_terms)
            for fun in (np.sin, np.cos)
        ])

        col_names = [
            f'{self.prefix}{fun.__name__[0].upper()}{i}'
            for i in range(self.n_terms)
            for fun in (np.sin, np.cos)
        ]

        fourier_df = pd.DataFrame(fourier_x, columns=col_names)

        if use_as_index:
            fourier_df.index = index

        return fourier_df
