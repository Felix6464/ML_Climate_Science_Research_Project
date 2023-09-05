from models.LSTM_enc_dec import *
import models.GRU_enc_dec as gru
from plots import *
from utilities import *
from torch.utils.data import DataLoader


def main():

    data = torch.load("data_piControl.pt")
    data = normalize_data(data)

    horizon = True

    # Specify the model number of the model to be tested


    model_num = [("517928np", "2-1"),
                 ("4716746np", "2-2"),
                 ("2482928np", "2-4"),
                 ("7125364np", "2-6"),
                 ("4908365np", "2-10"),
                 ("8049569np", "2-12"),
                 ("791884np", "4-1"),
                 ("5740785np", "4-2"),
                 ("9099984np", "4-4"),
                 ("4151419np", "4-6"),
                 ("5460966np", "4-8"),
                 ("65094np", "4-10"),
                 ("2316936np", "4-12")]
    model_num = [                 ("4062133np", "6-1"),
                                  ("9279148np", "6-2"),
                                  ("275478np", "6-4"),
                                  ("7009385np", "6-6"),
                                  ("2147163np", "6-8"),
                                  ("2301582np", "6-10"),
                                  ("5286230np", "6-12"),
                                  ("3436058np", "12-1"),
                                  ("9656267np", "12-2"),
                                  ("7693311np", "12-6"),
                                  ("593430np", "12-8"),
                                  ("7921374np", "12-10"),
                                  ("254581np", "12-12")]
####250EPOCHS
    # model_num = [("3699959np", "2-1"),
    #              ("9889937np", "2-2"),
    #              ("9647375np", "2-4"),
    #              ("9933773np", "2-6"),
    #              ("6597868np", "2-10"),
    #              ("567106np", "2-12"),
    #              ("9488902np", "4-1"),
    #              ("2848141np", "4-2"),
    #              ("1291107np", "4-4"),
    #              ("6941530np", "4-6"),
    #              ("6902494np", "4-8"),
    #              ("7139471np", "4-10"),
    #              ("9457686np", "4-12")]
    # model_num = [                 ("366124np", "6-1"),
    #                               ("6436104np", "6-2"),
    #                               ("5766124np", "6-4"),
    #                               ("1716158np", "6-6"),
    #                               ("4227347np", "6-8"),
    #                               ("6916466np", "6-10"),
    #                               ("6031523np", "6-12"),
    #                               ("1607761np", "12-1"),
    #                               ("3778713np", "12-2"),
    #                               ("5483003np", "12-4"),
    #                               ("3383730np", "12-6"),
    #                               ("36039np", "12-8"),
    #                               ("7361308np", "12-10"),
    #                               ("3031954np", "12-12")]
    model_num = [("8049569np", "2-12"),
                 ("3885361np", "XLimXTau_200k"),
                 ("5886861np", "XLimXTau"),
                 ("1042760np", "XLim_200k"),
                 ("3675721np", "XLim")]
    model_num = [("9877206np", "2-12_2l"),
                 ("1887542np", "2-12_2l_02d"),
                 ("773118np", "2-12_256h"),
                 ("9877206np", "2-12_2l"),
                 ("4493755np", "2-12_euler1"),
                 ("7531469np", "2-12_euler2")]
    model_num = [("5528071np", "vanilla")]
    model_num = [("7805350np", "input")]
    model_num = [("9949347np", "gru")]

    model_num = [("2097898np", "teacher_forcing")]

    ### FINAL PLOTS FOR REPORT

    #Model spread 2-12
    model_num = [("1902812np", "2-12"),
                 ("9133319np", "2-12"),
                 ("5298674np", "2-12"),
                 ("2690852np", "2-12"),
                 ("997732np", "2-12"),
                 ("7795178np", "2-12"),
                 ("8720532np", "2-12"),
                 ("6141024np", "2-12"),
                 ("2286937np", "2-12"),
                 ("6930550np", "2-12"),
                 ("4440213np", "2-12"),
                 ("2744386np", "2-12"),
                 ("2611751np", "2-12"),
                 ("3684351np", "2-12"),
                 ("6479918np", "2-12")]
    model_num = [("9197244np", "2-12"),
                 ("2989342np", "4-12"),
                 ("2574486np", "6-12"),
                 ("2299420np", "12-12")]
    
            #DATA
    model_num = [("7014435np", "60k"),
                ("9779810np", "50k"),
                ("4034580np", "40k"),
                ("94386np", "30k"),
                ("6153557np", "20k"),
                ("3880148np", "10k"),
                ("8217270np", "9k"),
                ("9251776np", "8k"),
                ("1281477np", "7k"),
                ("1578809np", "6k"),
                ("9939679np", "5k"),
                ("6043565np", "4k"),
                ("4567627np", "3k"),
                ("3918817np", "2k"),
                ("7829361np", "1k")]
    
    model_num = [("979173np", "2-1"),
                 ("10428np", "4-1"),
                 ("4832095np", "6-1"),
                 ("6818638np", "12-1")]
    model_num = [("5640620np", "2-2"),
                 ("8561347np", "4-2"),
                 ("9671189np", "6-2"),
                 ("6350493np", "12-2")]
    model_num = [("8893301np", "2-6"),
                 ("3028080np", "4-6"),
                 ("1706285np", "6-6"),
                 ("9624821np", "12-6")]
    model_num = [("527721np", "2-1"),
                 ("9384508np", "2-2"),
                 ("448978np", "2-4"),
                 ("8727459np", "2-6"),
                 ("2083560np", "2-8"),
                 ("5733615np", "2-10"),
                 ("4683225np", "2-12")]





    id = ["final-horizon"]

    loss_list = []
    loss_list_eval = []

    for m in range(len(model_num)):
        saved_model = torch.load(f"./final_models/model_{model_num[m][0]}.pt")

        # Load the hyperparameters of the model
        params = saved_model["hyperparameters"]
        #print("Hyperparameters of model {} : {}".format(model_num[m][0], params))
        #wandb.init(project=f"SST-{'SPREAD-Horizon'}", config=params, name=params['name'])

        hidden_size = params["hidden_size"]
        num_layers = params["num_layers"]
        input_window = params["input_window"]
        batch_size = params["batch_size"]
        loss_type = params["loss_type"]
        shuffle = params["shuffle"]
        loss_eval = params["loss_test"]


        if horizon is True:

            # Specify the number of features and the stride for generating timeseries raw_data
            num_features = 30
            x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
            losses = []

            for output_window in x:

                test_dataset = TimeSeriesLSTMnp(data.permute(1, 0),
                                               input_window,
                                               output_window)

                test_dataloader = DataLoader(test_dataset,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            drop_last=True)

                # Specify the device to be used for testing
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                # Initialize the model and load the saved state dict
                model = LSTM_Sequence_Prediction(input_size = num_features, hidden_size = hidden_size, num_layers=num_layers)
                model.load_state_dict(saved_model["model_state_dict"])
                model.to(device)

                loss = model.evaluate_model(test_dataloader, output_window, batch_size, loss_type)
                print("Output window: {}, Loss: {}".format(output_window, loss))
                losses.append(loss)
                #wandb.log({"Horizon": output_window, "Test Loss": loss})

            loss_list.append((losses, model_num[m][1]))
        loss_list_eval.append((loss_eval, model_num[m][1]))
        #wandb.finish()

    if horizon is True: plot_loss_horizon(loss_list, loss_type, id)
    #plot_loss_combined(loss_list_eval, id, loss_type)




if __name__ == "__main__":
    main()
