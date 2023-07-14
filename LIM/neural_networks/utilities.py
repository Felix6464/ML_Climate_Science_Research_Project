import numpy as np
import xarray as xr
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import json
import torch
import pickle


def reshape_xarray(input_data):
    # Define the target latitude and longitude dimensions
    target_lat = xr.DataArray(np.linspace(-90, 90, 192), dims='lat')
    target_lon = xr.DataArray(np.linspace(-180, 180, 360), dims='lon')

    # Reshape the input data using xr.interp()
    reshaped_data = input_data.interp(lat=target_lat, lon=target_lon, method='nearest')

    return reshaped_data

def apply_mask(mask, array):
    # Create a masked array using the where function
    masked_array = xr.where(mask == 100, np.nan, array)

    return masked_array


def calculate_monthly_anomalies(data):

    # Calculate the climatological mean for each month
    climatological_mean = data.groupby('time.month').mean(dim='time', keep_attrs=True)
    # Calculate the anomalies by subtracting the climatological mean for each month
    anomalies = data.groupby('time.month') - climatological_mean

    return anomalies

def crop_xarray_lat(input_data):

    cropped_ds = input_data.sel(lat=slice(-30, 30))

    return cropped_ds


def crop_xarray(lon_start, lon_end, input_data):
    if lon_start > lon_end:
        cropped_dataset_left = input_data.sel(lat=slice(-30, 30), lon=slice(lon_start-2, 180))
        new_scale_left = np.linspace(-180, -119, 52)
        cropped_dataset_left["lon"] = new_scale_left

        cropped_dataset_right = input_data.sel(lat=slice(-30, 30), lon=slice(-180, lon_end+2))
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
    """Transform flattened array without NaNs to gridded data with NaNs.
    Args:
        x_flat (np.ndarray): Flattened array of size (n_times, n_points) or (n_points).
        ids_notNaN (xr.DataArray): Boolean dataarray of size (n_points).
        times (np.ndarray): Time coordinate of xarray if x_flat has time dimension.
    Returns:
        xr.Dataset: Gridded data.
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
    """PCA of spatio-temporal data.
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
    Normalize the data using mean and standard deviation.

    Args:
        data (torch.Tensor): Input data to be normalized.

    Returns:
        torch.Tensor: Normalized data.

    """

    # Calculate the mean and standard deviation along the feature dimension
    mean = torch.mean(data, dim=1, keepdim=True)
    std = torch.std(data, dim=1, keepdim=True)

    # Apply normalization using the mean and standard deviation
    normalized_data = (data - mean) / std

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
    : param Xtrain:                    windowed training input data (input window size, # examples, # features)
    : param Ytrain:                    windowed training target data (output window size, # examples, # features)
    : param Xtest:                     windowed test input data (input window size, # examples, # features)
    : param Ytest:                     windowed test target data (output window size, # examples, # features)
    : return X_train_torch, Y_train_torch,
    :        X_test_torch, Y_test_torch:      all input np.arrays converted to PyTorch tensors

    '''

    X_train = torch.from_numpy(Xtrain).type(torch.Tensor)
    Y_train = torch.from_numpy(Ytrain).type(torch.Tensor)

    X_test = torch.from_numpy(Xtest).type(torch.Tensor)
    Y_test = torch.from_numpy(Ytest).type(torch.Tensor)

    return X_train, Y_train, X_test, Y_test



def save_dictionary(dictionary, filename):
    with open(filename, 'w') as file:
        json.dump(dictionary, file)

def load_dictionary(filename):
    with open(filename, 'r') as file:
        return eval(json.load(file))