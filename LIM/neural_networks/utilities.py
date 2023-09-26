import numpy as np
import xarray as xr
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import json
import torch
import os

from LIM.neural_networks.models.LSTM_enc_dec_input import *
from LIM.neural_networks.models.LSTM_enc_dec import *
from LIM.neural_networks.models.FNN_model import *
from LIM.neural_networks.models.LSTM import *
from LIM.neural_networks.models.GRU_enc_dec import *


def reshape_xarray(input_data):
    """
    Reshape an xarray object to a target latitude and longitude grid using linear interpolation.

    Parameters
    ----------
    input_data : xarray.DataArray
        The input xarray object to be reshaped.

    Returns
    -------
    xarray.DataArray
        The reshaped xarray object with the target latitude and longitude dimensions.

    Notes
    -----
    This function assumes that the input xarray object has 'lat' and 'lon' dimensions.

    The target latitude and longitude dimensions are defined as follows:
    - target_lat: a DataArray with 192 points linearly spaced between -90 and 90 degrees, with dimension 'lat'.
    - target_lon: a DataArray with 360 points linearly spaced between -180 and 180 degrees, with dimension 'lon'.

    The input xarray object is reshaped using the xr.interp() method with the 'nearest' interpolation method.
    """
    # Define the target latitude and longitude dimensions
    target_lat = xr.DataArray(np.linspace(-90, 90, 192), dims='lat')
    target_lon = xr.DataArray(np.linspace(-180, 180, 360), dims='lon')

    # Reshape the input raw_data using xr.interp()
    reshaped_data = input_data.interp(lat=target_lat, lon=target_lon, method='nearest')

    return reshaped_data

def apply_mask(mask, array):
    """
    Apply a mask to an xarray object using linear interpolation.

    Parameters
    ----------
    mask : xarray.DataArray
        The mask to be applied to the input array.
    array : xarray.DataArray
        The input xarray object to be masked.

    Returns
    -------
    xarray.DataArray
        The masked xarray object with NaN values where the mask is 100.
    """
    # Create a masked array using the where function
    masked_array = xr.where(mask == 100, np.nan, array)

    return masked_array


def calculate_monthly_anomalies(data):
    """
    Calculate monthly anomalies of an xarray object.

    Parameters
    ----------
    data : xarray.DataArray
        The input xarray object to calculate anomalies for.

    Returns
    -------
    xarray.DataArray
        The xarray object with monthly anomalies calculated.
    """
    # Calculate the climatological mean for each month
    climatological_mean = data.groupby('time.month').mean(dim='time', keep_attrs=True)

    # Calculate the anomalies by subtracting the climatological mean for each month
    anomalies = data.groupby('time.month') - climatological_mean

    return anomalies




def crop_xarray(lon_start, lon_end, input_data):
    if lon_start > lon_end:
        cropped_dataset_left = input_data.sel(lat=slice(-30, 30), lon=slice(lon_start - 2, 180))
        new_scale_left = np.linspace(-180, -119, 52)
        cropped_dataset_left["lon"] = new_scale_left

        cropped_dataset_right = input_data.sel(lat=slice(-30, 30), lon=slice(-180, lon_end + 2))
        new_scale_right = np.linspace(-121, -10, 112)
        cropped_dataset_right["lon"] = new_scale_right

        cropped_dataset = xr.concat(
            [
                cropped_dataset_left,
                cropped_dataset_right

            ],
            dim='lon'
        ).sortby('lon')
    else:
        cropped_dataset = input_data.sel(lon=slice(lon_start, lon_end))

    return cropped_dataset


def concatenate_and_save_data(pc_ts, pc_zos, data_type, filename):
    if data_type == "xr":
        # Concatenate along a specified dimension
        concatenated_xarray = xr.concat([pc_ts, pc_zos], dim='eof')

        # Save the xarray to a NetCDF file
        concatenated_xarray.to_netcdf(filename + '.nc')
    else:
        ts_20 = torch.from_numpy(pc_ts.data)
        zos_10 = torch.from_numpy(pc_zos.data)
        print(ts_20.shape)
        print(zos_10.shape)
        data = torch.cat((ts_20, zos_10), dim=0)
        print(data.shape)
        torch.save(data, filename + '.pt')


def map2flatten(x_map: xr.Dataset) -> list:
    """Flatten dataset/dataarray and remove NaNs.
    Args:
        x_map (xr.Dataset/ xr.DataArray): Dataset or DataArray to flatten.
    Returns:
        x_flat (xr.DataArray): Flattened dataarray without NaNs
        ids_notNaN (xr.DataArray): Boolean array where values are on the grid.
    """
    if type(x_map) == xr.core.dataset.Dataset:
        x_stack_vars = [x_map[var] for var in list(x_map.data_vars)]
        x_stack_vars = xr.concat(x_stack_vars, dim='var')
        x_stack_vars = x_stack_vars.assign_coords({'var': list(x_map.data_vars)})
        x_flatten = x_stack_vars.stack(z=('var', 'lat', 'lon'))
    else:
        x_flatten = x_map.stack(z=('lat', 'lon'))

    # Flatten and remove NaNs
    if 'time' in x_flatten.dims:
        idx_notNaN = ~np.isnan(x_flatten.isel(time=0))
    else:
        idx_notNaN = ~np.isnan(x_flatten)
    x_proc = x_flatten.isel(z=idx_notNaN.data)

    return x_proc, idx_notNaN


def flattened2map(x_flat: np.ndarray, ids_notNaN: xr.DataArray, times: np.ndarray = None) -> xr.Dataset:
    """Transform flattened array without NaNs to gridded raw_data with NaNs.
    Args:
        x_flat (np.ndarray): Flattened array of size (n_times, n_points) or (n_points).
        ids_notNaN (xr.DataArray): Boolean dataarray of size (n_points).
        times (np.ndarray): Time coordinate of xarray if x_flat has time dimension.
    Returns:
        xr.Dataset: Gridded raw_data.
    """
    if len(x_flat.shape) == 1:
        x_map = xr.full_like(ids_notNaN, np.nan, dtype=float)
        x_map[ids_notNaN.data] = x_flat
    else:
        temp = np.ones((x_flat.shape[0], ids_notNaN.shape[0])) * np.nan
        temp[:, ids_notNaN.data] = x_flat
        if times is None:
            times = np.arange(x_flat.shape[0])
        x_map = xr.DataArray(data=temp, coords={'time': times, 'z': ids_notNaN['z']})

    if 'var' in list(x_map.get_index('z').names):
        x_map = x_map.unstack()

        if 'var' in list(x_map.dims):  # For xr.Datasset only
            da_list = [xr.DataArray(x_map.isel(var=i), name=var)
                       for i, var in enumerate(x_map['var'].data)]
            x_map = xr.merge(da_list, compat='override')
            x_map = x_map.drop(('var'))
    else:
        x_map = x_map.unstack()

    return x_map


class SpatioTemporalPCA:
    """PCA of spatio-temporal raw_data.
    Wrapper for sklearn.decomposition.PCA with xarray.DataArray input.

    See EOF tutorial.
    Args:
        ds (xr.DataArray or xr.Dataset): Input dataarray to perform PCA on.
            Array dimensions (time, 'lat', 'lon')
        n_components (int): Number of components for PCA
    """

    def __init__(self, ds, n_components, **kwargs):
        self.ds = ds

        self.X, self.ids_notNaN = map2flatten(self.ds)

        # PCA
        self.pca = PCA(n_components=n_components)
        self.pca.fit(self.X.data)

        self.n_components = self.pca.n_components

    def eofs(self):
        """Return components of PCA.
        Return:
            components (xr.dataarray): Size (n_components, N_x, N_y)
        """
        # EOF maps
        eof_map = []
        for i, comp in enumerate(self.pca.components_):
            eof = flattened2map(comp, self.ids_notNaN)
            try:
                eof = eof.drop(['time', 'month'])
            except:
                eof = eof.drop(['time'])
            eof_map.append(eof)

        return xr.concat(eof_map, dim='eof')

    def principal_components(self):
        """Returns time evolution of components.
        Return:
            time_evolution (xr.Dataarray): Principal components of shape (n_components, time)
        """
        time_evolution = []
        for i, comp in enumerate(self.pca.components_):
            ts = self.X.data @ comp

            da_ts = xr.DataArray(
                data=ts,
                dims=['time'],
                coords=dict(time=self.X['time']),
            )
            time_evolution.append(da_ts)
        return xr.concat(time_evolution, dim='eof')

    def explained_variance(self):
        return self.pca.explained_variance_ratio_

    def reconstruction(self, z, newdim='x'):
        """Reconstruct the dataset from components and time-evolution.
        Args:
            z (np.ndarray): Low dimensional vector of size (time, n_components)
            newdim (str, optional): Name of dimension. Defaults to 'x'.
        Returns:
            _type_: _description_
        """
        reconstruction = z.T @ self.pca.components_

        rec_map = []
        for rec in reconstruction:
            x = flattened2map(rec, self.ids_notNaN)
            rec_map.append(x)

        rec_map = xr.concat(rec_map, dim=newdim)

        return rec_map


def matrix_decomposition(A):
    """Decompose square matrix:

        A = U D V.T
    Args:
        A (np.ndarray): Square matrix
    Returns:
        w (np.ndarray): Sorted eigenvalues
        U (np.ndarray): Matrix of sorted eigenvectors of A
        V (np.ndarray): Matrix of sorted eigenvectors of A.T
    """
    w, U = np.linalg.eig(A)
    idx_sort = np.argsort(w)[::-1]
    w = w[idx_sort]
    U = U[:, idx_sort]

    w_transpose, V = np.linalg.eig(A.T)
    idx_sort = np.argsort(w_transpose)[::-1]
    V = V[:, idx_sort]

    return w, U, V


def calculate_percentage(value, percentage):
    """
    Calculate the value of a percentage from a given value.

    :param value: The original value.
    :param percentage: The percentage to calculate.
    :return: The calculated value of the percentage.
    """
    return (value * percentage) / 100


def plot_loss(loss_values, loss_type, identifier):
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, 'b', label='Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'temp_models/{loss_type}_loss_{identifier}.png')
    plt.show()


def load_text_as_json(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
        json_data = json.loads(text)
    return json_data


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


def dataloader_seq2seq_feat(y, input_window, output_window, num_features):
    '''
    Create a windowed dataset

    :param y:                Time series feature (array)
    :param input_window:     Number of y samples to give the model
    :param output_window:    Number of future y samples to predict
    :param stride:           Spacing between windows
    :param num_features:     Number of features
    :return X, Y:            Arrays with correct dimensions for LSTM
                             (i.e., [input/output window size # examples, # features])
    '''
    data_len = y.shape[1]
    num_samples = data_len - input_window - output_window * 2

    # Initialize X and Y arrays with zeros
    X = np.zeros([num_samples, input_window, num_features])
    Y = np.zeros([num_samples, output_window, num_features])

    for feature_idx in np.arange(num_features):
        for sample_idx in np.arange(num_samples):
            # Create input window
            start_x = sample_idx
            end_x = start_x + input_window
            X[sample_idx, :, feature_idx] = y[feature_idx, start_x:end_x]

            # Create output window
            start_y = sample_idx + input_window
            end_y = start_y + output_window
            Y[sample_idx, :, feature_idx] = y[feature_idx, start_y:end_y]

    return X, Y


def numpy_to_torch(Xtrain, Ytrain, Xtest, Ytest):
    '''
    convert numpy array to PyTorch tensor
    : param Xtrain:                    windowed training input raw_data (input window size, # examples, # features)
    : param Ytrain:                    windowed training target raw_data (output window size, # examples, # features)
    : param Xtest:                     windowed test input raw_data (input window size, # examples, # features)
    : param Ytest:                     windowed test target raw_data (output window size, # examples, # features)
    : return X_train_torch, Y_train_torch,
    :        X_test_torch, Y_test_torch:      all input np.arrays converted to PyTorch tensors

    '''

    X_train = torch.from_numpy(Xtrain).type(torch.Tensor)
    Y_train = torch.from_numpy(Ytrain).type(torch.Tensor)

    X_test = torch.from_numpy(Xtest).type(torch.Tensor)
    Y_test = torch.from_numpy(Ytest).type(torch.Tensor)

    return X_train, Y_train, X_test, Y_test


def save_dict(file_path, dictionary):
    if os.path.exists(file_path):
        # Load existing dictionary from file
        with open(file_path, 'r') as file:
            existing_dict = json.load(file)

        # Merge dictionaries
        merged_dict = dict_merge([existing_dict, dictionary])
        data_to_write = merged_dict
    else:
        # Create a new dictionary with the given dictionary
        data_to_write = dictionary

    # Save dictionary to file
    with open(file_path, 'w') as file:
        json.dump(data_to_write, file)


def dict_merge(dicts_list):
    d = {**dicts_list[0]}
    for entry in dicts_list[1:]:
        for k, v in entry.items():
            d[k] = ([d[k], v] if k in d and type(d[k]) != list
                    else [*d[k], v] if k in d
            else v)
    return d


def normalize_tensor_individual(tensor):
    normalized_tensor = torch.zeros(tensor.size())

    for feature_idx in range(len(tensor[:, 0])):
        # Calculate the minimum and maximum values of the tensor
        min_value = torch.min(tensor[feature_idx, :])
        max_value = torch.max(tensor[feature_idx, :])

        # Check if the tensor has a single value (min and max are the same) to avoid division by zero
        if min_value == max_value:
            return tensor.new_ones(tensor.size())

        # Normalize the tensor using min-max normalization formula
        normalized_tensor[feature_idx, :] = (tensor[feature_idx, :] - min_value) / (max_value - min_value)

    return normalized_tensor


def load_models_testing(num_lstm_base, num_lstm, num_lstm_input, num_gru, num_ffn, num_lstm_input_tf):
    # Load the saved models
    saved_model_lstm_base = torch.load(f"./final_models/model_{num_lstm_base}.pt")
    saved_model_lstm = torch.load(f"./final_models/model_{num_lstm}.pt")
    saved_model_lstm_input = torch.load(f"./final_models/model_{num_lstm_input}.pt")
    saved_model_lstm_input_tf = torch.load(f"./final_models/model_{num_lstm_input_tf}.pt")
    saved_model_fnn = torch.load(f"./final_models/model_{num_ffn}.pt")
    saved_model_gru = torch.load(f"./final_models/model_{num_gru}.pt")

    # Load the hyperparameters of the lstm_model base
    params_lb = saved_model_lstm_base["hyperparameters"]

    # Load the hyperparameters of the lstm_model_enc_dec
    params_l = saved_model_lstm["hyperparameters"]

    # Load the hyperparameters of the lstm_input_model_enc_dec
    params_li = saved_model_lstm_input["hyperparameters"]

    # Load the hyperparameters of the lstm_input_model_enc_dec with teacher_forcing
    params_li_tf = saved_model_lstm_input_tf["hyperparameters"]

    # Load the hyperparameters of the fnn_model
    params_f = saved_model_fnn["hyperparameters"]

    # Load the hyperparameters of the fnn_model
    params_g = saved_model_gru["hyperparameters"]

    # Specify the device to be used for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_lstm_base = LSTM_Sequence_Prediction_Base(input_size=params_lb["num_features"],
                                                    hidden_size=params_lb["hidden_size"],
                                                    num_layers=params_lb["num_layers"])

    model_lstm = LSTM_Sequence_Prediction(input_size=params_lb["num_features"],
                                          hidden_size=params_l["hidden_size"],
                                          num_layers=params_l["num_layers"])

    model_lstm_inp = LSTM_Sequence_Prediction_Input(input_size=params_lb["num_features"],
                                                    hidden_size=params_li["hidden_size"],
                                                    num_layers=params_li["num_layers"])

    model_lstm_inp_tf = LSTM_Sequence_Prediction_Input(input_size=params_lb["num_features"],
                                                       hidden_size=params_li_tf["hidden_size"],
                                                       num_layers=params_li_tf["num_layers"])

    model_ffn = FeedforwardNetwork(input_size=params_lb["num_features"],
                                   hidden_size=params_f["hidden_size"],
                                   output_size=params_lb["num_features"],
                                   input_window=params_f["input_window"])

    model_gru = GRU_Sequence_Prediction(input_size=params_lb["num_features"],
                                        hidden_size=params_g["hidden_size"],
                                        num_layers=params_g["num_layers"])

    # Load the saved models
    model_gru.load_state_dict(saved_model_gru["model_state_dict"])
    model_gru = model_gru.to(device)
    model_lstm_base.load_state_dict(saved_model_lstm_base["model_state_dict"])
    model_lstm_base = model_lstm_base.to(device)
    model_lstm.load_state_dict(saved_model_lstm["model_state_dict"])
    model_lstm = model_lstm.to(device)
    model_lstm_inp.load_state_dict(saved_model_lstm_input["model_state_dict"])
    model_lstm_inp = model_lstm_inp.to(device)
    model_lstm_inp_tf.load_state_dict(saved_model_lstm_input["model_state_dict"])
    model_lstm_inp_tf = model_lstm_inp.to(device)
    model_ffn.load_state_dict(saved_model_fnn["model_state_dict"])
    model_ffn = model_ffn.to(device)

    return model_lstm_base, model_lstm, model_lstm_inp, model_ffn, model_gru, model_lstm_inp_tf


def min_max_values_per_slice(tensor):
    """
    Calculate the minimum and maximum values for each slice along the first dimension of the input tensor.

    Args:
    tensor (torch.Tensor): Input tensor.

    Returns:
    dict: A dictionary containing the minimum and maximum values for each slice.
    """
    result_dict = {}
    num_slices = tensor.size(0)

    for i in range(num_slices):
        slice_min = tensor[i].min().item()
        slice_max = tensor[i].max().item()
        result_dict[f'slice_{i}'] = {'min': slice_min, 'max': slice_max}

    return result_dict