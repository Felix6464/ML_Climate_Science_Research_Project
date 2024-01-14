from LIM.LIM_class import *
from LIM.utilities.utilities import *

# Load control data_generated
data = torch.load("../data_piControl.pt")



def generate_lim_data(timesteps, tau_list, num_models, data, save_name, time=False):
    """
    Generate LIM (Linear Inverse Modeling) data_generated integration and save it to a file.

    Parameters:
        timesteps (int): The number of time steps for data_generated integration.
        tau_list (list): List of time constants for LIM modeling.
        num_models (int): Number of LIM models to generate data_generated.
        data (torch.Tensor): Input data_generated for LIM modeling.
        save_name (str): Name of the file to save the generated data_generated.
        time (bool, optional): Whether to include time information in the generated data_generated.
                              Defaults to False.

    Returns:
        None: The function saves the generated data_generated and does not return any values.
    """
    # Initialize a zero tensor for LIM data_generated integration
    lim_integration_ = torch.zeros((30, timesteps))

    # Determine the amount of data_generated available
    amount_data = len(data[0, :])

    # Loop through each LIM model
    for m in range(num_models):
        # Set the time constant tau for the current model
        tau = 1

        # Create a LIM model with the specified tau
        model = LIM(tau)

        # Fit the LIM model to the input data_generated
        model.fit(data[:, :amount_data].detach().cpu().numpy())

        # Perform noise integration with the LIM model
        lim_integration, times_ = model.noise_integration(data[:, amount_data-1],
                                                          timesteps=timesteps,
                                                          num_comp=30,
                                                          t_delta_=tau_list[m])

        # Convert the integration result to a PyTorch tensor
        lim_integration = torch.from_numpy(lim_integration.T)

        # If 'time' is True, add time information to the data_generated
        if time is True:
            lim_integration = torch.zeros(lim_integration.shape[0]+1, lim_integration.shape[1])
            count = 0
            for t in range(int(timesteps / 12)):
                for m in range(12):
                    month = np.array(m+1)
                    month = np.expand_dims(month, axis=0)
                    lim = np.append(lim_integration[:, count + m], month, axis=0)
                    lim_integration[:, count+m] = torch.from_numpy(lim)
                count += 12

        # Update the 'lim_integration_' tensor with the current results
        if amount_data == len(data[0, :]):
            lim_integration_ = lim_integration
        else:
            lim_integration_ = torch.cat((lim_integration_, lim_integration), dim=1)

        # Reduce the amount of available data_generated for the next iteration
        amount_data -= 1000

    # Save the final 'lim_integration_' tensor to a file
    torch.save(lim_integration_, save_name)
    print("saved model")


def generate_lim_data_euler_method(timesteps, num_models, data, save_name, time=False):
    """
    Generate LIM (Linear Inverse Modeling) data_generated integration using Euler's method and save it to a file.

    Parameters:
        timesteps (int): The number of time steps for data_generated integration.
        num_models (int): Number of LIM models to generate data_generated.
        data (torch.Tensor): Input data_generated for LIM modeling.
        save_name (str): Name of the file to save the generated data_generated.
        time (bool, optional): Whether to include time information in the generated data_generated.
                              Defaults to False.

    Returns:
        None: The function saves the generated data_generated and does not return any values.
    """
    # Initialize a zero tensor for LIM data_generated integration
    lim_integration_ = torch.zeros((30, timesteps))

    # Determine the amount of data_generated available
    amount_data = len(data[0, :])

    # Loop through each LIM model
    for m in range(num_models):
        # Set the time constant tau for the current model
        tau = 1

        # Create a LIM model with the specified tau
        model = LIM(tau)

        # Fit the LIM model to the input data_generated
        model.fit(data[:, :amount_data].detach().cpu().numpy())

        # Perform data_generated integration using Euler's method
        lim_integration = model.euler_method(L=model.logarithmic_matrix,
                                             Q=model.noise_covariance,
                                             x0=data[:, amount_data-1],
                                             dt=1,
                                             T=timesteps
                                             )

        # Convert the integration result to a PyTorch tensor
        lim_integration = torch.from_numpy(lim_integration)

        # If 'time' is True, add time information to the data_generated
        if time is True:
            lim_integration = torch.zeros(lim_integration.shape[0]+1, lim_integration.shape[1])
            count = 0
            for t in range(int(timesteps / 12)):
                for m in range(12):
                    month = np.array(m+1)
                    month = np.expand_dims(month, axis=0)
                    lim = np.append(lim_integration[:, count + m], month, axis=0)
                    lim_integration[:, count+m] = torch.from_numpy(lim)
                count += 12

        # Update the 'lim_integration_' tensor with the current results
        if amount_data == len(data[0, :]):
            lim_integration_ = lim_integration
        else:
            lim_integration_ = torch.cat((lim_integration_, lim_integration), dim=1)

        # Reduce the amount of available data_generated for the next iteration
        amount_data -= 1000

    # Save the final 'lim_integration_' tensor to a file
    torch.save(lim_integration_, save_name)
    print("saved model")

num_models = 1
tau_list = [1]
timesteps = 200000

generate_lim_data_euler_method(timesteps, len(tau_list), data, "lim_integration_200k_new_.pt", time=False)



# eofs = np.arange(0, 30)
#
# date_end = (timesteps / 12) + 12
# # Set the start and end dates
# start_date = cftime.DatetimeNoLeap(1, 1, 15, 12, 0, 0, 0, has_year_zero=True)
# end_date = cftime.DatetimeNoLeap(date_end, 10, 15, 12, 0, 0, 0, has_year_zero=True)
#
# # Create a range of monthly timestamps
# time = xr.cftime_range(start=start_date, end=end_date, freq='M')
#
# print("Lim interation : {} {}".format(type(lim_integration), lim_integration.shape))
# # Create a DataArray from the numpy array with coordinates
# data_xr = xr.DataArray(lim_integration, coords=[np.arange(30), time], dims=['eof', 'time'])
#
# # Print the created xarray
# print(type(data_xr))
#
# #data_xr.to_netcdf('./neural_networks/synthetic_data/lim_integration_xarray_20k[-1]p.nc')
# print("Save raw_data")