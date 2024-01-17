import pandas as pd
import csv
from sklearn.metrics import r2_score, mean_squared_error
import torch
import numpy as np
import os

def getParentDir(levels=1) -> str:
    """
    @param path: starts without /
    @return: Parent path at the specified levels above.
    """
    current_directory = os.path.dirname(__file__)

    parent_directory = current_directory
    for i in range(0, levels):
        parent_directory = os.path.split(parent_directory)[0]

    #file_path = os.path.join(parent_directory, path)
    return parent_directory


def preprocess_csv(csv_path):

    data = pd.read_csv(f'data/{csv_path}.csv', quoting=csv.QUOTE_NONE)
    print("data.head():\n", data.head())

    # Parse the datetime column using the specified format
    data['"DateTime"'] = pd.to_datetime(data['"DateTime"'], format='"%d/%m/%Y %H:%M"')

    # Format the datetime column as 'YYYY-MM-DD HH:mm:ss'
    data['"DateTime"'] = data['"DateTime"'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Replace 'your_output.csv' with the desired output file path
    output_file = 'data/WindForecast_2022_2023.csv'

    # Save the modified DataFrame to a new CSV file
    data.to_csv(output_file, index=False)

    with open(f'data/WindForecast_2022_2023.csv', "r+", encoding="utf-8") as csv_file:
        content = csv_file.read()

    with open(f'data/WindForecast_2022_2023.csv', "w+", encoding="utf-8") as csv_file:
        csv_file.write(content.replace('"', ''))

# Define a custom scoring function for MSE
def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def normalize_data(data):
    """
    Normalize the raw_data using mean and standard deviation.

    Args:
        data (torch.Tensor): Input raw_data to be normalized.

    Returns:
        torch.Tensor: Normalized raw_data.

    """
    # Calculate the mean and standard deviation along the feature dimension
    mean = torch.mean(data, dim=1, keepdim=True)
    std = torch.std(data, dim=1, keepdim=True)

    # Apply normalization using the mean and standard deviation
    normalized_data = torch.zeros_like(data)

    for i in range(len(mean)):
        normalized_data[i, :] = (data[i, :] - mean[i]) / std[i]

    return normalized_data


def min_max_normalization(data):
    """
    Normalize the raw_data using min-max normalization.

    Args:
        data (torch.Tensor): Input raw_data to be normalized.

    Returns:
        torch.Tensor: Normalized raw_data.

    """
    # Calculate the minimum and maximum values along the feature dimension
    min_vals, _ = torch.min(data, dim=1, keepdim=True)
    max_vals, _ = torch.max(data, dim=1, keepdim=True)

    # Apply min-max normalization
    normalized_data = (data - min_vals) / (max_vals - min_vals)

    return normalized_data


# determine the supported device
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu') # don't have GPU
    return device

# convert a df to tensor to be used in pytorch
def df_to_tensor(df):
    device = get_device()
    return torch.from_numpy(df.values).float().to(device)


def mase(training_series, testing_series, prediction_series):
    """
    Computes the MEAN-ABSOLUTE SCALED ERROR forcast error for univariate time series prediction.

    See "Another look at measures of forecast accuracy", Rob J Hyndman

    parameters:
        training_series: the series used to train the model, 1d numpy array
        testing_series: the test series to predict, 1d numpy array or float
        prediction_series: the prediction of testing_series, 1d numpy array (same size as testing_series) or float
        absolute: "squares" to use sum of squares and root the result, "absolute" to use absolute values.

    """
    n = training_series.shape[0]
    d = np.abs(np.diff(training_series)).sum() / (n - 1)

    errors = np.abs(testing_series - prediction_series)
    return errors.mean() / d


class LogTransformation:

    @staticmethod
    def transform(x):
        xt = np.sign(x) * np.log(np.abs(x) + 1)

        return xt

    @staticmethod
    def inverse_transform(xt):
        x = np.sign(xt) * (np.exp(np.abs(xt)) - 1)

        return x
