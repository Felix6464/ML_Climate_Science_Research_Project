from LIM.neural_networks.models.LIM_class import *
from LIM.neural_networks.utilities import *
import cftime


# Create the DataLoader for first principal component
data = torch.load("data_piControl.pt")
print("Data shape : {}".format(data.shape))
print(min_max_values_per_slice(data))

# timesteps = 60000
# time = True
# # original fit
# tau = 1
# model_org = LIM(tau)
# model_org.fit(data.detach().cpu().numpy())
# #131999 = 11000 years
# lim_integration, times_ = model_org.noise_integration(data[:, -1], timesteps=timesteps, num_comp=30)
# lim_integration = lim_integration.T
#
# lim_data = torch.zeros(lim_integration.shape[0]+1, lim_integration.shape[1])
#
# if time is True:
#     count = 0
#     for t in range(int(timesteps / 12)):
#         for m in range(12):
#             month = np.array(m+1)
#             month = np.expand_dims(month, axis=0)
#             lim = np.append(lim_integration[:, count + m], month, axis=0)
#             lim_data[:, count+m] = torch.from_numpy(lim)
#         count += 12
#     lim_integration = lim_data
#
# if time is False:
#     lim_integration = torch.from_numpy(lim_integration)
# torch.save(lim_integration, "lim_integration_TIME_60k_tau[-1].pt")
# print("Saved data")


timesteps = 20000
time = True
# original fit
tau = 1
model1 = LIM(tau)
model1.fit(data[:, :12000].detach().cpu().numpy())

model2 = LIM(tau)
model2.fit(data[:, :11000].detach().cpu().numpy())

model3 = LIM(tau)
model3.fit(data[:, :10000].detach().cpu().numpy())

model4 = LIM(tau)
model4.fit(data[:, :9000].detach().cpu().numpy())

model5 = LIM(tau)
model5.fit(data[:, :8000].detach().cpu().numpy())

model6 = LIM(tau)
model6.fit(data[:, :7000].detach().cpu().numpy())

model7 = LIM(tau)
model7.fit(data[:, :6000].detach().cpu().numpy())

model8 = LIM(tau)
model8.fit(data[:, :5000].detach().cpu().numpy())

#131999 = 11000 years
lim_integration1, times_ = model1.noise_integration(data[:, 12000], timesteps=timesteps, num_comp=30)
lim_integration1 = torch.from_numpy(lim_integration1.T)

lim_integration2, times_ = model2.noise_integration(data[:, 11000], timesteps=timesteps, num_comp=30)
lim_integration2 = torch.from_numpy(lim_integration2.T)

lim_integration3, times_ = model3.noise_integration(data[:, 10000], timesteps=timesteps, num_comp=30)
lim_integration3 = torch.from_numpy(lim_integration3.T)

lim_integration4, times_ = model4.noise_integration(data[:, 9000], timesteps=timesteps, num_comp=30)
lim_integration4 = torch.from_numpy(lim_integration4.T)

lim_integration5, times_ = model5.noise_integration(data[:, 8000], timesteps=timesteps, num_comp=30)
lim_integration5 = torch.from_numpy(lim_integration5.T)

lim_integration6, times_ = model6.noise_integration(data[:, 7000], timesteps=timesteps, num_comp=30)
lim_integration6 = torch.from_numpy(lim_integration6.T)

lim_integration7, times_ = model7.noise_integration(data[:, 6000], timesteps=timesteps, num_comp=30)
lim_integration7 = torch.from_numpy(lim_integration7.T)

lim_integration8, times_ = model8.noise_integration(data[:, 5000], timesteps=timesteps, num_comp=30)
lim_integration8 = torch.from_numpy(lim_integration8.T)

lim_integration = torch.cat((lim_integration1, lim_integration2, lim_integration3, lim_integration4,lim_integration5, lim_integration6, lim_integration7, lim_integration8), dim=1)
#lim_integration = torch.cat((lim_integration1, lim_integration2, lim_integration3, lim_integration4), dim=1)
print(lim_integration.shape)

print(min_max_values_per_slice(lim_integration1))
print(min_max_values_per_slice(lim_integration2))
print(min_max_values_per_slice(lim_integration3))
print(min_max_values_per_slice(lim_integration4))
print(min_max_values_per_slice(lim_integration))
torch.save(lim_integration, "lim_integration_multipleLim_XL.pt")



eofs = np.arange(0, 30)

date_end = (timesteps / 12) + 12
# Set the start and end dates
start_date = cftime.DatetimeNoLeap(1, 1, 15, 12, 0, 0, 0, has_year_zero=True)
end_date = cftime.DatetimeNoLeap(date_end, 10, 15, 12, 0, 0, 0, has_year_zero=True)

# Create a range of monthly timestamps
time = xr.cftime_range(start=start_date, end=end_date, freq='M')

print("Lim interation : {} {}".format(type(lim_integration), lim_integration.shape))
# Create a DataArray from the numpy array with coordinates
data_xr = xr.DataArray(lim_integration, coords=[np.arange(30), time], dims=['eof', 'time'])

# Print the created xarray
print(type(data_xr))

#data_xr.to_netcdf('./neural_networks/synthetic_data/lim_integration_xarray_20k[-1]p.nc')
print("Save data")