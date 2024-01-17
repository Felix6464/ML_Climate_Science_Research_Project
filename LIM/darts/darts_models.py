from darts.models import (
    LinearRegressionModel,
    RegressionModel,
    VARIMA,
    Theta,
    XGBModel,
    BlockRNNModel,
    RegressionEnsembleModel,
    TransformerModel,
    NBEATSModel,
    NHiTSModel,
    TCNModel,
    RandomForest,
)
from sklearn.linear_model import Ridge
from darts.metrics import mape, mase, mae, mse, ope, r2_score, rmse, rmsle
import time
import pandas as pd
import math
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts import concatenate
import seaborn as sns
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import numbers


import warnings
warnings.filterwarnings('ignore')




def fit_and_eval_model(model, train_data, val_data, covariates_train, covariates_val):
    """
    Fits a given time series model to the training data and evaluates it on validation data.

    This function is designed to handle various types of time series models, including ones that
    may require covariates. It fits the model, makes predictions, and computes various accuracy metrics.

    Parameters:
    - model (tuple): A tuple containing the model instance and its name as a string.
    - train_data (pandas.DataFrame or list): The training data for the model. Can be a DataFrame or a list of DataFrames.
    - val_data (pandas.DataFrame): The validation data used for evaluating the model.
    - covariates_train (pandas.DataFrame or list): Covariates corresponding to the training data.
    - covariates_val (pandas.DataFrame): Covariates corresponding to the validation data.

    Returns:
    - list: A list containing two elements:
        1. The first element is a list itself, containing the model's forecast and a dictionary of accuracy metrics (MAE, MSE, RMSE, MAPE, and computation time).
        2. The second element is a list of tuples, each containing a fitted model instance and its name.

    Note:
    - The function assumes the existence of specific functions for calculating MSE, MAE, RMSE, and MAPE.
    - It also uses `time.perf_counter()` to measure the processing time.
    - The function handles VARIMA models and others differently, particularly in how they are fitted and how predictions are made.
    """

    model_name = model[1]
    model = model[0]

    fitted_models = []

    t_start = time.perf_counter()
    # fit the model and compute predictions

    print("Start fitting Model: " + model_name + " ...")

    if model_name in ["VARIMA"]:
        _ = model.fit(train_data)
        fitted_models.append((model, model_name))
    else:
        if type(train_data) == list:

            covariates_train_lst = []
            for i in range(len(train_data)):
                covariates_train_lst.append(covariates_train)

            _ = model.fit(train_data)  # , past_covariates=covariates_train_lst)
            fitted_models.append((model, model_name))
        else:
            _ = model.fit(train_data, past_covariates=covariates_train)
            fitted_models.append((model, model_name))
        # print("Model: " + str(model) + " could not be fitted.")

    if model_name in ["VARIMA"]:
        forecast = model.predict(len(val_data))  # , series=val_data) #, future_covariates=None)
    else:
        if type(train_data) == list:
            forecast = model.predict(len(val_data), series=val_data)
        else:
            forecast = model.predict(len(val_data), past_covariates=covariates_val)  # ,  series=val_data, )
        # print("Model: " + str(model) + " could not be predicted.")

    print("Model: " + str(model_name) + " fitted and predicted.")
    # compute accuracy metrics and processing time
    res_mse = mse(val_data, forecast)
    res_mae = mae(val_data, forecast)
    res_rmse = rmse(val_data, forecast)
    res_mape = mape(val_data, forecast)
    res_time = time.perf_counter() - t_start

    res_accuracy = {"MAE": res_mae,
                    "MSE": res_mse,
                    "RMSE": res_rmse,
                    "MAPE": res_mape,
                    "time": res_time}

    results = [forecast, res_accuracy]
    return [results, fitted_models]


def eval_pretrained_model(model, val_data, covariates_val, horizon):
    """
    Evaluates a pretrained time series model on validation data.
    It makes predictions using the provided validation data and computes various accuracy metrics.

    Parameters:
    - model (tuple): A tuple containing the pretrained model instance and its name as a string.
    - val_data (pandas.DataFrame): The validation data used for evaluating the model's performance.
    - covariates_val (pandas.DataFrame): Covariates corresponding to the validation data, if applicable.
    - horizon (int): The forecast horizon.

    Returns:
    - list: A list containing two elements:
        1. The forecast made by the model on the validation data.
        2. A dictionary of accuracy metrics (MAE, MSE, RMSE, MAPE, R2, RMSLE, and computation time).
    """
    model_name = model[1]
    model = model[0]

    t_start = time.perf_counter()

    if model_name in ["VARIMA"]:
        forecast = model.predict(len(val_data), future_covariates=None)
    else:
        forecast = model.predict(len(val_data), past_covariates=covariates_val)

    print("Model: " + str(model_name) + " predicted val data.")
    # compute accuracy metrics and processing time
    res_mse = mse(val_data, forecast)
    res_mae = mae(val_data, forecast)
    res_r2 = r2_score(val_data, forecast)
    res_rmse = rmse(val_data, forecast)
    res_rmsle = rmsle(val_data, forecast)
    res_mape = mape(val_data, forecast)
    res_time = time.perf_counter() - t_start

    res_accuracy = {"MAE": res_mae,
                    "MSE": res_mse,
                    "RMSE": res_rmse,
                    "MAPE": res_mape,
                    "time": res_time}

    results = [forecast, res_accuracy]
    return results


def prepare_models(models_list, num_epochs, batch_size, history, horizon, lr):
    """
    Prepares a list of time series forecasting models based on specified parameters and model types.

    This function initializes various time series models with given hyperparameters. It supports a range
    of models, including NBEATS, NHiTS, TCN, Transformer, various RNN types (LSTM, GRU, RNN), RandomForest,
    XGBoost, VARIMA, LinearRegressionModel, and RidgeRegression.

    Parameters:
    - models_list (list): A list of strings representing the model names to be prepared.
    - num_epochs (int): The number of epochs for training the neural network models.
    - batch_size (int): The batch size for training the neural network models.
    - history (int): The length of the input time series history to consider for each model.
    - horizon (int): The prediction horizon for the forecasting models.
    - lr (float): Learning rate for the neural network models.

    Returns:
    - model_list (list): A list of tuples, where each tuple contains a model instance and its corresponding name.

    Note:
    - The function is designed to handle both univariate and multivariate time series models.
    - Early stopping is implemented for all neural network models to prevent overfitting.
    - The function can optionally be configured to use GPU acceleration.
    """
    model_list = []

    if "NBEATS" in models_list:
        m_nbeats = NBEATSModel(
            input_chunk_length=history,
            output_chunk_length=horizon,
            batch_size=batch_size,
            n_epochs=num_epochs,
            num_stacks=30,
            num_blocks=1,
            num_layers=4,
            layer_widths=256,
            expansion_coefficient_dim=5,
            trend_polynomial_degree=2,
            dropout=0.1,
            activation="ReLU",
            log_tensorboard=True,
            optimizer_kwargs={'lr': lr},
            pl_trainer_kwargs={
                "callbacks": [EarlyStopping(
                    monitor="train_loss",
                    patience=5,
                    min_delta=0.01,
                    mode='min',
                )],
                #"accelerator": "gpu",
                #"devices": [0]
            })
        model_list.append((m_nbeats, "NBeats"))

    if "NHiTS" in models_list:
        m_nhits = NHiTSModel(
            input_chunk_length=history,
            output_chunk_length=horizon,
            batch_size=batch_size,
            n_epochs=num_epochs,
            num_stacks=30,
            num_blocks=1,
            num_layers=4,
            layer_widths=256,
            dropout=0.1,
            activation="ReLU",
            log_tensorboard=True,
            optimizer_kwargs={'lr': lr},
            pl_trainer_kwargs={
                "callbacks": [EarlyStopping(
                    monitor="train_loss",
                    patience=5,
                    min_delta=0.01,
                    mode='min',
                )],
                #"accelerator": "gpu",
                #"devices": [0]
            })
        model_list.append((m_nhits, "NHiTS"))

    if "TCN" in models_list:
        m_tcn = TCNModel(
            input_chunk_length=history,
            output_chunk_length=horizon,
            batch_size=batch_size,
            n_epochs=num_epochs,
            num_filters=3,
            kernel_size=3,
            num_layers=4,
            dropout=0.1,
            dilation_base=2,
            log_tensorboard=True,
            optimizer_kwargs={'lr': lr},
            pl_trainer_kwargs={
                "callbacks": [EarlyStopping(
                    monitor="train_loss",
                    patience=5,
                    min_delta=0.01,
                    mode='min',
                )],
                #"accelerator": "gpu",
                #"devices": [0]
            })
        model_list.append((m_tcn, "TCN"))

    if "Transformer" in models_list:
        m_transformer = TransformerModel(
            input_chunk_length=history,
            output_chunk_length=horizon,
            batch_size=batch_size,
            n_epochs=num_epochs,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=256,
            dropout=0.1,
            activation="ReLU",
            log_tensorboard=True,
            optimizer_kwargs={'lr': lr},
            save_checkpoints=True,
            pl_trainer_kwargs={
                "callbacks": [EarlyStopping(
                    monitor="train_loss",
                    patience=5,
                    min_delta=0.01,
                    mode='min',
                )],
                #"accelerator": "gpu",
                #"devices": [0]
            })
        model_list.append((m_transformer, "Transformer"))
    for rnn_type in ["LSTM", "GRU", "RNN"]:
        if rnn_type in models_list:
            m_rnn = BlockRNNModel(
                input_chunk_length=history,
                output_chunk_length=horizon,  # RNNModel uses a fixed `output_chunk_length=1
                model=rnn_type,
                hidden_dim=256,
                n_rnn_layers=3,
                dropout=0.1,
                n_epochs=num_epochs,
                batch_size=batch_size,
                log_tensorboard=True,
                optimizer_kwargs={'lr': lr},
                pl_trainer_kwargs={
                    "callbacks": [EarlyStopping(
                        monitor="train_loss",
                        patience=5,
                        min_delta=0.01,
                        mode='min',
                    )],
                    #"accelerator": "gpu",
                    #"devices": [0]
                })
            model_list.append((m_rnn, rnn_type))

    if "RandomForest" in models_list:
        m_randf = RandomForest(
            lags=history,
            lags_past_covariates=12,
            lags_future_covariates=None,
            output_chunk_length=horizon,
            n_estimators=30,
            multi_models=False,  # might explore this
        )
        model_list.append((m_randf, "RandomForest"))

    if "XGBoost" in models_list:
        m_xgb = XGBModel(lags=history,
                         lags_past_covariates=history,
                         # lags_future_covariates=None,
                         output_chunk_length=horizon,
                         n_estimators=30,
                         multi_models=True)
        model_list.append((m_xgb, "XGBoost"))

    if "VARIMA" in models_list:
        m_varima = VARIMA(
            p=1,
            d=0,
            q=0,
            trend=None)
        model_list.append((m_varima, "VARIMA"))

    if "LinearRegressionModel" in models_list:
        m_linearreg = LinearRegressionModel(
            lags=history,
            lags_past_covariates=history,
            output_chunk_length=horizon)
        model_list.append((m_linearreg, "LinearRegressionModel"))

    if "RidgeRegression" in models_list:
        m_ridgereg = RegressionModel(
            model=Ridge(),
            lags=history,
            lags_past_covariates=history,
            output_chunk_length=horizon)
        model_list.append((m_ridgereg, "RidgeRegression"))

    return model_list


def create_covariates_from_df(df, train_df, test_df, history, target, eval=False, selected_series=[]):
    """
    Creates a set of covariates from a DataFrame for use in time series forecasting models.

    This function processes a given DataFrame and extracts time series covariates, excluding specified columns.
    The resulting covariates are concatenated and returned as a single TimeSeries object.

    Parameters:
    - df (pandas.DataFrame): The DataFrame from which to create covariates.
    - target (str): The name of the target variable column in df.
    - selected_series (list): A list of column names to be excluded from the covariates.

    Returns:
    - TimeSeries: A concatenated TimeSeries object containing all the covariates.

    Note:
    - The function excludes the date and target columns, as well as any columns listed in selected_series.
    - The TimeSeries.from_dataframe method is used to convert each column into a TimeSeries, ensuring consistent time indexing and handling missing dates.
    - The frequency is set to 'H' (hourly), but can be modified according to the time resolution of the data.
    """

    # Extend the val set for calculating the right amount of covariates
    new_df_tail = train_df.tail(history * 2)
    test_cov = pd.concat((new_df_tail, test_df), axis=0)
    new_df_tail = df[:10000].tail(history * 2)
    train_cov = pd.concat((new_df_tail, train_df), axis=0)

    if eval:
        df = test_cov
    else:
        df = train_cov

    covariates_df = None

    if len(df.columns) > 2:

        time_series_covariates = []

        for col in df.columns:
            if col == target:
                pass
            elif col in selected_series:
                pass
            else:
                covariate = TimeSeries.from_dataframe(pd.DataFrame(df[col]), fill_missing_dates=True,
                                                      freq=None)
                time_series_covariates.append(covariate)
                # print(f"Created covariate {col} with {len(covariate)} observations")

        covariates_df = concatenate(time_series_covariates, axis=1)

    return covariates_df


def create_predictions_dataframe(models, model_predictions):
    """
    Creates a DataFrame summarizing the prediction accuracy of multiple models.

    This function processes the predictions made by a list of models and consolidates their accuracy metrics into a single DataFrame for easy comparison and visualization.

    Parameters:
    - models (list): A list of tuples, where each tuple contains a model instance and its corresponding name.
    - model_predictions (list): A list of dictionaries, where each dictionary contains the accuracy metrics for a corresponding model.

    Returns:
    - pandas.DataFrame: A DataFrame where each column represents a model and rows represent different accuracy metrics.

    Note:
    - The function loops through each model and its predictions, creating a DataFrame from the prediction dictionary.
    - The accuracy metrics are then concatenated into a single DataFrame with model names as column headers.
    - The function also sets the display precision of the DataFrame and highlights the minimum and maximum values for each metric across models.
    """
    # Initialize an empty DataFrame to store prediction accuracy
    prediction_accuracy_df = pd.DataFrame()

    # Loop through models and their predictions
    for model_index, (model, model_name) in enumerate(models):
        # Create a DataFrame from the prediction dictionary

        predictions_df = pd.DataFrame.from_dict(model_predictions[model_index][1], orient="index")
        # Set column names to the model's name
        predictions_df.columns = [model_name]

        # Concatenate the predictions DataFrame with the accuracy DataFrame
        if model_index == 0:
            prediction_accuracy_df = predictions_df
        else:
            prediction_accuracy_df = pd.concat([prediction_accuracy_df, predictions_df], axis=1)

    # Set the display precision for the DataFrame
    pd.set_option("display.precision", 3)

    # Highlight minimum and maximum values in the DataFrame
    styled_df = prediction_accuracy_df.style.highlight_min(color="lightgreen", axis=1).highlight_max(color="yellow",
                                                                                                     axis=1)

    return prediction_accuracy_df


def create_eval_barplot(model_predictions_df, metric=["MSE"]):
    """
    Creates a bar plot to compare the performance of different models based on a specified metric.

    This function uses Seaborn to generate a bar plot where each bar represents the value of a chosen metric
    (e.g., MSE) for a different model. The plot provides a visual comparison of model performance.

    Parameters:
    - model_predictions_df (pandas.DataFrame): A DataFrame containing the prediction accuracy metrics of various models.
    - metric (list): A list containing the name of the metric to be plotted. Default is ["MSE"].

    Returns:
    - matplotlib.pyplot: A plot object representing the generated bar plot.

    Note:
    - The function extracts the specified metric's values from the DataFrame for each model and plots them as bars.
    - Model names are used as labels on the x-axis, and the metric value is displayed at the top of each bar.
    - The function supports customization of the metric to be plotted, allowing for flexibility in model evaluation.
    """
    dataframes = []

    for col in model_predictions_df.columns:
        data = {
            'Metric': metric,
            'Value': model_predictions_df[col].values[3]
        }
        dataframes.append((pd.DataFrame(data), col))

    # Create a barplot using Seaborn
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    width = 0.35  # Width of the bars
    x = range(len(dataframes[0][0]['Metric']))
    model_names = [df[1] for df in dataframes]

    for i, df in enumerate(dataframes):
        # Offset the x-positions to separate the bars
        offset = i * width
        bars = plt.bar([pos + offset for pos in x], df[0]['Value'], width=width, label=f'{df[1]}')

        # Add column names on top of each bar
        for bar, metric_value in zip(bars, df[0]['Value']):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0075, f'{metric_value:.3f}', ha='center',
                     va='bottom')

    plt.title(f"Comparison of {metric[0]}-Score")
    plt.xlabel(metric[0])
    plt.ylabel("Value")
    # Set x-ticks to model names
    # plt.xticks(range(len(model_names)), model_names)
    plt.xticks([width * (i + 0.05) for i in range(len(model_names))], model_names)
    plt.legend()
    plt.show()

    return plt


def create_predictions_plots(models, model_predictions, series, plt_whole_series=False):
    """
    Creates a series of plots comparing actual values to the predictions made by various models.

    This function generates a grid of subplots where each subplot represents the predictions of a specific model
    against the actual time series values. It also includes performance metrics in the subplot titles for reference.

    Parameters:
    - models (list): A list of tuples, where each tuple contains a model instance and its corresponding name.
    - model_predictions (list): A list of predictions made by each model.
    - series (TimeSeries): The actual time series data.
    - plt_whole_series (bool, optional): If True, plots the entire series; otherwise, plots only the forecasting range.

    Returns:
    - matplotlib.figure.Figure: A figure object containing the subplots.

    Note:
    - Each subplot shows the actual time series values and the corresponding model predictions.
    - The Mean Squared Error (MSE) and processing time of each model are displayed in the subplot titles.
    - The function dynamically adjusts the number of subplot rows based on the number of models.
    """
    # Calculate the number of rows of charts
    num_models = len(models)
    rows = math.ceil(num_models / 2)

    # Create a subplots grid for the charts
    fig, ax = plt.subplots(rows, 2, figsize=(20, 5 * rows))
    ax = ax.ravel()

    # Loop through the models to plot actual and predicted values
    for i, (model, model_name) in enumerate(models):
        # Plot the actual values

        forecasting_range = len(model_predictions[i][0])

        if plt_whole_series:
            series.plot(label="actual", ax=ax[i])
        else:
            series[len(series) - forecasting_range:].plot(label="actual", ax=ax[i])

        # Plot the predicted values and label
        model_predictions[i][0].plot(label="prediction: " + str(model_name), ax=ax[i])

        # Extract MAPE and processing time from model metrics
        mse_model = model_predictions[i][1]["MSE"]
        time_model = model_predictions[i][1]["time"]

        # Set the title for the subplot
        ax[i].set_title(
            "\n\n" + str(model_name) + ": MSE {:.4f}".format(mse_model) + " - time {:.2f} sec".format(time_model)
        )

        ax[i].set_xlabel("")
        ax[i].legend()

    fig.tight_layout()
    # Show the plot
    fig.show()

    return fig


def prepare_multiple_time_series(mv_series_, DATE, chosen_series):
    mv_series = []

    for var in chosen_series:
        series = TimeSeries.from_dataframe(pd.DataFrame(mv_series_[[DATE, var]]), time_col=DATE,
                                           fill_missing_dates=True, freq='H')

        mv_series.append(series)

    return mv_series


def create_split_at(train_index):
    # split position: if string, then interpret as Timestamp
    if isinstance(train_index, numbers.Number):
        split_at = train_index
    else:
        split_at = pd.Timestamp(train_index)

    return split_at


def plot_train_test_split(series, train_index, plot=True):
    # plot the split
    train_ts, val_ts = series.split_before(create_split_at(train_index))
    plt.figure(101, figsize=(20, 10))
    train_ts.plot(label='training')
    val_ts.plot(label='validation')
    plt.legend()
    plt.show()

    return train_ts, val_ts