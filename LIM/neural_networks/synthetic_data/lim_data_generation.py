from LIM.neural_networks.models.LIM_class import *
from LIM.neural_networks.utilities import *
import cftime

# Load control data
data = torch.load("./data/data_piControl.pt")
#print("Data shape : {}".format(data.shape))
#print(min_max_values_per_slice(data))



def generate_lim_data(timesteps, tau_list, num_models, data, save_name, time=False):


    lim_integration_ = torch.zeros((30, timesteps))

    amount_data = len(data[0, :])

    for m in range(num_models):

        tau = 1
        model = LIM(tau)
        model.fit(data[:, :amount_data].detach().cpu().numpy())

        lim_integration, times_ = model.noise_integration(data[:, amount_data-1],
                                                          timesteps=timesteps,
                                                          num_comp=30,
                                                          t_delta_=tau_list[m])

        lim_integration = torch.from_numpy(lim_integration.T)


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


        if amount_data == len(data[0, :]):
            lim_integration_ = lim_integration
        else:
            lim_integration_ = torch.cat((lim_integration_, lim_integration), dim=1)

        amount_data -= 1000

    torch.save(lim_integration_, save_name)
    print("saved model")



num_models = 20
tau_list = [0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3]
timesteps = 40000

generate_lim_data(timesteps, tau_list, len(tau_list), data, "lim_integration_200kXLimXTau.pt", time=False)



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