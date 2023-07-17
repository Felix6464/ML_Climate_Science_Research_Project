from LIM_class import *
import utilities as ut
from LIM.neural_networks.old.LSTM_enc_dec_old import *
from plots import *


def main():

    #data = torch.load("./data/data_piControl.pt")
    data = torch.load("./synthetic_data/lim_integration_130k[-1].pt")


    # Calculate the mean and standard deviation along the feature dimension
    data = ut.normalize_data(data)

    data_len = 3000
    data = data[:, :data_len]
    data = data.numpy()

    # original fit
    tau = 1
    model_org = LIM(tau)
    model_org.fit(data, eps=0.01)
    model_num = "LIM"

    criterion = nn.L1Loss()


    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    loss_list = []

    for output_window in x:

        loss = 0
        sample_size = len(data[1]) - output_window - 50
        print("sample_size: {}".format(sample_size))
        print("output_window: {}".format(output_window))
        for datapoint in range(sample_size):

            # Forecast mean using LIM model
            lim_integration, times_ = model_org.noise_integration(data[:, datapoint ], timesteps=output_window, num_comp=30)
            lim_integration = lim_integration.T[:, 1:]
            #print("lim_integration: {} + type : {}".format(lim_integration, type(lim_integration)))
            #print("datapoint: {} + type : {}".format(data[:, datapoint:datapoint+output_window], type(data[:, datapoint:datapoint+output_window])))

            lim_integration = torch.from_numpy(lim_integration)
            target = torch.from_numpy(data[:, datapoint:datapoint+output_window])
            loss += criterion(lim_integration, target).item()



        loss /= sample_size
        loss_list.append((loss, f"{model_num}_out={output_window}"))
        print("Loss for output window {} : {}".format(output_window, loss))




    print(f"Test loss: {loss_list}")
    plot_loss_horizon(loss_list, model_num, "L1")




if __name__ == "__main__":
    main()
