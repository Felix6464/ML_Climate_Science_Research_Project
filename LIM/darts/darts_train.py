from darts_models import *
from darts import TimeSeries
from sklearn.model_selection import train_test_split
from discovery import *
from feature_engineering import *
from config import *
from utility_functions import getParentDir
from darts.utils.missing_values import fill_missing_values

from LIM.utilities.utilities import *
import pandas as pd


# Load data_generated from a file and store it in 'data_'
data = torch.load("../data/data_piControl_month_feature.pt")

mv_series = pd.DataFrame(data.transpose(0, 1)).astype("float")
mv_series.rename(columns={0: "PC1_Target"}, inplace=True)
mv_series.rename(columns={30: "date"}, inplace=True)
mv_series["date"] = mv_series["date"].astype(int)
mv_series.info()

# Generate the date range
start_date = '1950-01-01'
date_range = pd.date_range(start=start_date, periods=len(mv_series), freq='D')
mv_series["date"] = date_range



########################################################################################
# Using Darts library for modeling

LOAD_MODELS = False
SAVE_MODELS = False
PLOTS = False
TIME_DELAY_EMBEDDING = False

for j in range(1):
    for i in range(1):
        ########################################################################################
        # Using Darts library for modeling

        NUM_DATA = 14000
        DATE = "date"
        FREQUENCY = "D"
        TARGET = 'PC1_Target'  # target variable
        TRAIN = int(0.95 * len(mv_series[:NUM_DATA].index))  # training until index
        EPOCHS = 30  # number of epochs for NN models
        BATCH_SIZE = 64  # batch size for NN models
        HISTORY = (i + 1) * 24  # history for NN models
        HORIZON = (i + 1) * 24  # horizon for NN models
        LR = 1e-3

        # define the models to be used
        model_names = ["RidgeRegression", "LSTM", "RNN", "Transformer", "XGBoost", "NBEATS"]
        model_names = ["LSTM"]

        # Normalize the data and set date as index
        mv_series[DATE] = pd.to_datetime(mv_series[DATE])
        mv_series.set_index(DATE, inplace=True)
        mv_series = (mv_series - mv_series.mean()) / mv_series.std()


        print("TIME DELAY EMBEDDING:", TIME_DELAY_EMBEDDING)
        print("HISTORY:", HISTORY)
        print("HORIZON:", HORIZON)
        print("NUM DATA:", NUM_DATA)
        print("BATCH SIZE:", BATCH_SIZE)

        if PLOTS:

            # Start and end of the date range to extract
            start, end = '2012-01-01', '2013-01-01'
            series_cp = mv_series.copy()
            series_cp[DATE] = series_cp.index

            series_year = series_cp[(series_cp[DATE] >= start) & (series_cp[DATE] < end)]

            # Plot specified timeseries
            plot_columns(series_year, columns=series_cp.columns)

            # Plot the resampled data
            plot_resampled(series_cp, start, end, DATE, TARGET, ["W", "M"])

            # Plot the mean for specified time interval and column name to identify trend
            # ["Day", "Week", "Month", "Year"] as possible intervals
            check_trend(series_cp, DATE, TARGET, interval="Month")

            # Plot the mean for specified time interval and column name to identify trend
            compare_trends(series_cp, DATE, [TARGET])

            # Plot autocorrelation for specified time interval and column name
            autocorrelation_plot(series_cp, date_col=DATE, column=TARGET, interval="M")

            # Calculate anomalies using isolation forest and plot outliers
            output, score = run_isolation_forest(pd.DataFrame(series_cp[[DATE, TARGET]]), date_col=DATE,
                                                 contamination=0.005)

            # Seasonal decomposition
            seasonal_decomposition(series_cp, DATE, TARGET, resample_freq=["M"], seasonal_period=[12])


        if TIME_DELAY_EMBEDDING:

            # Create time delay embedding for specific column and lag
            mv_series = create_shifted_df(mv_series, history=int(HISTORY/2), horizon=int(HORIZON/2))

            # Create rolling mean columns for specific columns and windows
            mv_series = create_rolling_mean_columns(mv_series, [TARGET], windows=[6, 12, 24])

            # Create time based features columns
            mv_series = create_time_based_features(mv_series, TARGET, freq=["D", "W", "M"])

            # Create column with rolling mean from previous year same week
            if NUM_DATA > 18000:
                mv_series = create_rm_last_year(mv_series, TARGET)

            # Create column with expanding window statistics
            mv_series = create_expanding_window_features(mv_series, [TARGET])

            # Create fourier series features to account for seasonality (experimental)
            # mv_series = create_fourier_series_features(mv_series, DATE, TARGET)


        # Filter out columns with high correlation to minimize redundancy
        print("MV Series before corrleation filter: ", mv_series.info())
        mv_series, corr_features = correlation_filter(mv_series, TARGET, corr_threshold=0.9)

        # Specifiy the number of data points to be used
        mv_series_crop = mv_series[200:14200]
        print("MV Series after corrleation filter and crop: ", mv_series_crop.info())

        # Split the data into train and test
        forecast_size = (24 * 7) / len(mv_series_crop)
        train, test = train_test_split(mv_series_crop, test_size=forecast_size,
                                       shuffle=False)  # exactly 180 data points for test set
        #train = train[40000-NUM_DATA:]

        # create the covariates
        covariates_train = create_covariates_from_df(mv_series, train, test, HISTORY, TARGET)
        covariates_val = create_covariates_from_df(mv_series, train, test, HISTORY, TARGET)
        if covariates_train: print("Covariates used for Training: ", covariates_train.columns)

        # create the time series object
        series = TimeSeries.from_dataframe(pd.DataFrame(mv_series_crop[TARGET]), freq=FREQUENCY, fill_missing_dates=True)
        train = TimeSeries.from_dataframe(pd.DataFrame(train[TARGET]), freq=FREQUENCY, fill_missing_dates=True)
        val = TimeSeries.from_dataframe(pd.DataFrame(test[TARGET]), freq=FREQUENCY, fill_missing_dates=True)

        # fill missing values
        train = fill_missing_values(train)
        val = fill_missing_values(val)
        covariates_train = fill_missing_values(covariates_train)
        covariates_val = fill_missing_values(covariates_val)

        # Start training and evaluation
        if not LOAD_MODELS:

            # Function to initiate the models
            models = prepare_models(model_names, EPOCHS, BATCH_SIZE, HISTORY, HORIZON, LR)

            # call the forecasters one after the other
            results = [fit_and_eval_model(model, train, val, covariates_train, covariates_val) for model in
                       models]
            model_predictions = [result[0] for result in results]
            fitted_models = [result[1] for result in results]

            # Save the models if flag is True
            if SAVE_MODELS:
                for m in fitted_models:
                    m[0][0].save(f"trained_models/{m[0][1]}_{EPOCHS}_{BATCH_SIZE}_{HISTORY}_{HORIZON}_{LR}_{NUM_DATA}.pkl")

        else:

            # Function to initiate the models
            models = prepare_models(model_names, series, EPOCHS, BATCH_SIZE, HISTORY, HORIZON, LR)

            loaded_models = []

            # Load trained models
            for m in models:
                loaded_model = m[0].load(
                    f"trained_models/{m[1]}_{EPOCHS}_{BATCH_SIZE}_{HISTORY}_{HORIZON}_{LR}_{NUM_DATA}.pkl")
                loaded_models.append((loaded_model, m[1]))

            model_predictions = [eval_pretrained_model(model, val, covariates_val, HORIZON) for model in loaded_models]



        # Create a dataframe with the predictions
        prediction_accuracy_df = create_predictions_dataframe(models, model_predictions)
        eval_barplot = create_eval_barplot(prediction_accuracy_df)

        eval_barplot.savefig(
            f"./outputs/eval_barplot{EPOCHS}_{BATCH_SIZE}_{HISTORY}_{HORIZON}_{LR}_{NUM_DATA}_te_{TIME_DELAY_EMBEDDING}.png")
        print(prediction_accuracy_df.to_string())

        # Plot the predictions
        predictions_fig = create_predictions_plots(models, model_predictions, series, plt_whole_series=False)
        predictions_fig.savefig(
            f"./outputs/predictions{EPOCHS}_{BATCH_SIZE}_{HISTORY}_{HORIZON}_{LR}_{NUM_DATA}_te_{TIME_DELAY_EMBEDDING}.png")




