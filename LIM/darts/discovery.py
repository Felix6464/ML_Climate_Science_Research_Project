import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.ensemble import IsolationForest
import statsmodels.api as sm

import warnings
warnings.filterwarnings('ignore')



def plot_resampled(df, start, end, date_col, column, freq=["D", "W"]):
    """
    Plots resampled time series data with different frequencies and rolling means.

    This function takes a DataFrame and resamples the time series data based on specified frequencies
    and a rolling window. It generates a plot showing the original data resampled daily, weekly,
    and as a rolling mean, allowing for visual comparison of trends at different time scales.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the time series data.
    - start (str): The start date for the plot.
    - end (str): The end date for the plot.
    - date_col (str): The name of the column in df that contains the date information.
    - column (str): The name of the column containing the time series data to be plotted.
    - freq (list): A list containing two strings representing the resampling frequencies.
                   Default is ["D", "W"] for daily and weekly.

    Returns:
    - matplotlib.pyplot: A plot object representing the generated time series plot.

    Note:
    - The function first converts the date column to a DateTime index, then resamples the data
      using the specified frequencies and a rolling window.
    - The resulting plots are saved to a file named after the time series column, under a directory 'discovery_plots'.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df[date_col])
    df = df.set_index(["date"])

    df_freq1 = df[column].resample(freq[0]).mean()
    df_freq2 = df[column].resample(freq[1]).mean()
    df_freq3 = df[column].rolling(24 * 7 * 12, center=True).mean()

    # Plot daily and weekly resampled time series together
    fig, ax = plt.subplots()
    ax.plot(df_freq1.loc[start:end],
            marker='.', linestyle='-', linewidth=0.5, label=freq[0] + " mean resample")
    ax.plot(df_freq2.loc[start:end],
            marker='o', markersize=8, linestyle='-', label=freq[1] + " mean resample")
    ax.plot(df_freq3.loc[start:end],
            marker='.', linestyle='-', label='7-D Rolling Mean')
    ax.set_ylabel(column)
    ax.legend()
    plt.show()
    fig.savefig("./discovery_plots/" + column + "_resampled.png")


def plot_columns(df, columns):
    """
    Creates a subplot for each specified column in a DataFrame and saves the resulting plot.

    This function generates a series of subplots, each representing the time series data of a specified column from the DataFrame. The plots are saved to a file for further analysis and visualization.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data to be plotted.
    - columns (list): A list of column names from df that are to be plotted as subplots.
    """
    axes = df[columns].plot(marker='.', alpha=0.5, linestyle='None', figsize=(15, 15), subplots=True, legend=False)
    for i, ax in enumerate(axes):
        ax.set_ylabel(columns[i])
    plt.savefig("./discovery_plots/plot_columns.png")


def check_trend(df, date, col_name, interval="Year"):
    """
    Analyzes and plots the trend in a specified column of a DataFrame over time.

    This function visualizes the trend in a time series by plotting rolling means and standard deviations over different intervals. It also generates a boxplot to visualize the distribution of values over the specified interval.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the time series data.
    - date (str): The name of the column in df that contains date information.
    - col_name (str): The name of the column for which the trend is to be analyzed.
    - interval (str, optional): The time interval for the boxplot (e.g., "Year", "Month"). Default is "Year".

    Returns:
    - None: The function does not return anything but saves generated plots to files.

    Note:
    - The function creates rolling means and standard deviations for 1 day, 7 days, 30 days, and 365 days.
    - Boxplots are generated to analyze how the selected column's values are distributed across different time intervals.
    - The rolling statistics are plotted to visualize the trend and variability in the data over time.
    - The generated plots are saved as image files in the 'discovery_plots' directory.
    """

    df = df.copy()
    df["date"] = pd.to_datetime(df[date])
    df = df.set_index(["date"])

    df["Year"] = df.index.year
    df["Month"] = df.index.month
    df["Week"] = df.index.week
    df["Day"] = df.index.day

    df_365d = df[col_name].rolling(window=365, center=True, min_periods=360).mean()
    df_30d = df[col_name].rolling(window=30, center=True, min_periods=25).mean()
    df_30d_std = df[col_name].rolling(window=30, center=True, min_periods=25).std()
    df_7d = df[col_name].rolling(window=7, center=True, min_periods=3).mean()
    df_1d = df[col_name].rolling(window=1, center=True, min_periods=1).mean()

    # Boxplot of data by Interval
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df[[date, col_name, interval]], x=interval, y=col_name, ax=ax)
    ax.set_title(f'{col_name} by {interval}')
    plt.show()
    fig.savefig("./discovery_plots/boxplot_trend.png")

    # Plot daily, 7-day rolling mean, and 365-day rolling mean time series
    fig, ax = plt.subplots()
    # ax.plot(df_1d, marker='.', markersize=2, color='#c4841d',
    #        linestyle='None', label='Daily Mean')
    ax.plot(df_7d, linewidth=2, label='7-d Rolling Mean', color='#687fa3', alpha=0.3)
    ax.plot(df_30d, linewidth=2, label='30-d Rolling Mean', color='#c4b486', alpha=0.7)
    ax.plot(df_30d_std, linewidth=2, label='30-d Rolling Std', color='#32a852', alpha=0.7)
    ax.plot(df_365d, color='blue', linewidth=3,
            label='Trend (365-d Rolling Mean)')
    # Set x-ticks to yearly interval and add legend and labels
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend()
    ax.set_xlabel('Year')
    ax.set_ylabel(col_name)
    ax.set_title(f'Trend in {col_name}')
    plt.show()
    fig.savefig('./discovery_plots/check_trend.png')


def compare_trends(df, date, data_columns):
    """
    Compares the trends in multiple data columns of a DataFrame over time.

    This function visualizes the trends by plotting rolling means over a 365-day window for each specified data column. The trends are displayed on a single plot for easy comparison.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the time series data.
    - date (str): The name of the column in df that contains date information.
    - data_columns (list): A list of column names in df for which trends are to be compared.

    Returns:
    - None: The function does not return anything but saves the generated plot to a file.

    Note:
    - The function first converts the date column to a DateTime index.
    - Rolling means are computed for each data column over a 365-day window.
    - Each column's trend is plotted on the same axes for direct comparison.
    - The plot is saved as an image file in the 'discovery_plots' directory.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df[date])
    df = df.set_index(["date"])

    df_365d = df[data_columns].rolling(window=365, center=True, min_periods=360).mean()

    fig, ax = plt.subplots()

    for col in data_columns:
        ax.plot(df_365d[col], label=col)
        # Set x-ticks to yearly interval, adjust y-axis limits, add legend and labels
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.set_ylim(0, 400)
        ax.legend()
        ax.set_ylabel(col)
        ax.set_title('Trends in Solar Irradiance')

    plt.show()
    fig.savefig('./discovery_plots/compare_trends.png')


def autocorrelation_plot(df, date_col, column, interval):
    """
    Generates an autocorrelation plot for a specified column in a DataFrame.

    This function plots the autocorrelation of a time series data column over a specified interval.
    The autocorrelation plot helps in understanding the correlation of the time series with its lagged values.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the time series data.
    - date_col (str): The name of the column in df that contains date information.
    - column (str): The name of the column for which the autocorrelation is to be analyzed.
    - interval (str): The resampling interval (e.g., 'D' for daily).

    Returns:
    - None: The function does not return anything but saves the generated plot to a file.

    Note:
    - The function first converts the date column to a DateTime index.
    - The specified column is resampled according to the given interval, and the median is computed for each resampled period.
    - The autocorrelation plot is generated using pandas' built-in plotting functionality.
    - The plot is saved as an image file in the 'discovery_plots' directory.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df[date_col])
    df = df.set_index(["date"])

    pd.plotting.autocorrelation_plot(df[column].resample(interval).median())
    plt.show()
    plt.savefig('./discovery_plots/autocorrelation_plot.png')



def run_isolation_forest(df: pd.DataFrame, date_col, contamination=0.005, n_estimators=200,
                         max_samples=0.7) -> pd.DataFrame:
    """
    Applies an Isolation Forest algorithm to detect outliers in a time series dataset.

    This function uses the Isolation Forest algorithm to identify outliers in the dataset. It adds two new columns to the DataFrame indicating whether each point is an outlier and the anomaly score for each point.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the time series data.
    - date_col (str): The name of the column in df that contains date information.
    - contamination (float, optional): The proportion of outliers in the data set. Default is 0.005.
    - n_estimators (int, optional): The number of base estimators in the ensemble. Default is 200.
    - max_samples (float, optional): The proportion of samples to draw from X to train each base estimator. Default is 0.7.

    Returns:
    - tuple: A tuple containing two elements:
        1. A pandas Series indicating outlier status (1 for outliers, 0 otherwise).
        2. A pandas Series containing the anomaly scores for each data point.

    Note:
    - The function visualizes the time series data, highlighting the outliers identified by the model.
    - The results, including the outlier status and anomaly scores, are saved in the 'Outlier' and 'Score' columns of the input DataFrame.
    - A plot showing the identified outliers is saved to a file in the 'discovery_plots' directory.
    """
    df = df.copy()
    df_ = df.set_index([date_col])
    model_data = df_.values.reshape(-1, 1)

    iso_forest = (IsolationForest(random_state=42,
                                  contamination=contamination,
                                  n_estimators=n_estimators,
                                  max_samples=max_samples)
                  )

    iso_forest.fit(model_data)
    output = pd.Series(iso_forest.predict(model_data)).apply(lambda x: 1 if x == -1 else 0)

    score = iso_forest.decision_function(model_data)

    df["Outlier"] = output
    df["Score"] = score

    print(f'Number of Outliers below Anomaly Score Threshold {contamination}:')
    print(len(df.query(f"Outlier == 1 & Score <= {contamination}")))

    df[df.columns[1]].plot(marker='.', alpha=0.5, linestyle='None', figsize=(15, 15), subplots=True, legend=False)
    df.loc[df['Outlier'] == 1, df.columns[1]].plot(marker='o', linestyle='None', color='r')
    plt.show()
    plt.savefig("./discovery_plots/outliers.png")

    return output, score


def seasonal_decomposition(df, date_col, columns, resample_freq, seasonal_period=[1, 12]):
    """
    Performs seasonal decomposition on specified columns of a DataFrame.

    This function applies time series seasonal decomposition to each specified column after resampling the data. It visualizes the trend, seasonality, and residuals for each time series.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the time series data.
    - date_col (str): The name of the column in df that contains date information.
    - columns (list or str): The names of columns to be decomposed. If a string is provided, it's converted to a list.
    - resample_freq (list): A list of strings representing the frequencies for resampling.
    - seasonal_period (list, optional): A list of integers indicating the seasonal periods for decomposition. Default is [1, 12].

    Returns:
    - None: The function does not return anything but saves the decomposition plots to files.

    Note:
    - The function resamples the data according to the provided frequencies and then applies seasonal decomposition.
    - The decomposition results are plotted and saved to the 'discovery_plots' directory.
    - The function handles multiple columns and resampling frequencies, making it versatile for various time series analyses.
    """
    modified_df = df.copy()
    modified_df["date"] = pd.to_datetime(df[date_col])
    modified_df = modified_df.set_index("date")

    if type(columns) != list:
        columns = [columns]

    for i, freq in enumerate(resample_freq):
        for column in columns:
            df_temp = modified_df.resample(freq).mean()

            result = sm.tsa.seasonal_decompose(df_temp[column].values, period=seasonal_period[i])
            trend = result.trend
            seasonal = result.seasonal
            residual = result.resid
            result.plot()
            plt.show()
            plt.savefig("./discovery_plots/seasonal_decomposition.png")

        '''
        models = [
            MSTL(
                season_length=[12*24],  # seasonalities of the time series
                trend_forecaster=AutoARIMA(),  # model used to forecast trend
            )
        ]

        sf = StatsForecast(
            models=models,  # model used to fit each time series
            freq="H",  # frequency of the data
        )
        temp_df = pd.DataFrame(modified_df[[column, date_col]]).set_index(date_col)
        print(temp_df.tail(10))
        sf = sf.fit(temp_df)
        test = sf.fitted_[0, 0].model_
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        test.plot(ax=ax, subplots=True, grid=True)
        plt.tight_layout()
        plt.show()'''



