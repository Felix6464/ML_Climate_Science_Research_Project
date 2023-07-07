from LIM_class import *
import utilities as ut
from LSTM_enc_dec import *
from plots import *


def main():

    data = torch.load("./data/data_piControl.pt")
    data_lim = torch.load("./synthetic_data/lim_integration_100k[-1].pt")


    # Reshape the data if necessary (assuming a 2D tensor)
    if len(data.shape) == 1:
        data = data.unsqueeze(1)

    # Calculate the mean and standard deviation along the feature dimension
    data = ut.normalize_data(data)

    # original fit
    tau = 1
    model_org = LIM(tau)
    model_org.fit(data_lim, eps=0.01)

    criterion = nn.L1Loss()


    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    loss_list = []

    data = data.numpy()

    for output_window in x:

        for datapoint in range(len(data[1])):
            print("datapoint", datapoint.shape)

            # Forecast mean using LIM model
            lim_integration, times_ = model_org.noise_integration(data[:, datapoint], timesteps=output_window, num_comp=30)
            lim_integration = lim_integration.T

            loss = criterion(lim_integration, data[:, datapoint:datapoint+output_window])






    print(f"Test loss: {loss_list}")
    plot_loss_horizon(loss_list, model_num, loss_type)




if __name__ == "__main__":
    main()
