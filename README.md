# Machine Learning in Climate Science Research Project

## How much Data do S2S-Neural-Networks need? An ENSO Showcase

## Description
This Python machine learning repository focuses on predicting the time evolution of the El Niño-Southern Oscillation (ENSO) using various sequence-to-sequence (S2S) neural network architectures.
It also incorporates the "Linear Inverse Model" (LIM) as a baseline comparison and to enhance ENSO forecasts by integrating additional data points, aiming to examine whether additional data shows signfiicant improvement to the forecasting task. 
## Download Raw Data

### CESM2 piControl Data 



## How to run
First, install dependencies
```bash
# clone project   
git clone https://github.com/Felix6464/ML_Climate_Science_Research_Project.git

# install project   
cd ML_Climate_Science_Research_Project  
conda create --name <env> --file requirements.txt
conda activate <env>
 ```   
Next, navigate to any file and run it.

## Repository Structure

- **Linear Inverse Model**: In the directory `LIM`, you'll find:
  - **LIM_implementation_1994.pdf** - The base paper for the LIM implementation.
  - **LIM_class.py** - contains the implementation of the Liner Inverse Model (LIM).
  - **lim_integration_plots.ipynb** - A Jupyter Notebook that:
    - Loads raw data
    - Computes principal components
    - Plots Empirical Orthogonal Functions (EOFs)
    - Crops the data to the respective ENSO region
    - Fits the LIM model
    - Checks for stationary time series
    - Creates multiple plots to validate the implementation
  - **Plots** folder contains plots generated by **lim_integration_plots.ipynb** in both PNG and SVG formats.


- **Neural Networks**: This directory `LIM/neural_networks` holds the implementation for deep learning approaches to ENSO prediction. Inside the `models` folder, you'll find various neural network implementations:
  - `FNN_model.py` - FeedForwardNeuralNetwork
  - `GRU_enc_dec.py` - GatedRecurrentUnit
  - `LSTM.py` - LongShortTermModel
  - `LSTM_enc_dec.py` Encoder-Decoder-LSTM with only hidden state of encoder as input for prediction
  - `LSTM_enc_dec_input.py` Encoder-Decoder-LSTM with last state as input for prediction


- **Raw Data**: The `raw_data` folder contains the CESM2 piControl data for sea surface height and sea surface temperature, as well as the sea-land mask. Put the raw data here for consequently loading it


- **Final Models Trained**: The `final_models_trained` folder contains trained PyTorch models used for evaluation. Model names are represented by randomly generated integers for identification aswell as "np" (numpy) or "xr" (xarray) to identify on which type of data it was trained


- **Synthetic Data**: The `synthetic_data/data` folder contains files for generating synthetic data based on the CESM2 piControl data, as well as the final multidimensional NumPy data used for training.
  - `checking_timeseries.ipynb` - checks shape of timeseries of data and compares it to piControl data
  - `lim_data_generation.py` - generates synthetic data by integration of the LIM using the euler method and save the new data to the `data` folder
  - `testing_lim_integration.ipynb` - verifys the integration of the LIM for stationarity and plots the resulting time series


- **Training**: The following scripts contain scripts for training various models using synthetic data. These include:
  - `fnn_training.py`: Trains a feedforward neural network model with specified parameters using synthetically generated data on the train split of the data.
  - `rnn_training.py`: Trains a recurrent model with specified parameters on the synthetic data of the train split. The type of recurrent model used can be changed by importing the respective class from the `models` folder.


- **Testing**: The following scripts contain test scripts used during development to validate and experiment with the code. These include:
  - `fnn_testing.py`: Loads a pretrained feedforward neural network model from file and evaluates it for prediction horizons ranging from 1 to 24. It calculates the loss for each prediction horizon over the test set and plots the loss distribution curve.
  - `rnn_testing.py`: Evaluates a pretrained RNN model over prediction horizons of 1 to 24. The type of recurrent model used can be changed by importing the respective class from the `models` folder.
  - `plot_saved_model.py`: Loads a pretrained model from file and evaluates it on both the test and train sets for the chosen horizon during training. It also plots the timeseries forecast of the first principal component (can be varied) and the training loss curve of the model.
  - `testing_combined.py`: Loads multiple pretrained models, as well as the LIM, with different architectures and evaluates them simultaneously to create a plot that compares the performance of different architectures at once.


- **Utilities**: The `utilities` folder contains multiple utility functions for data preprocessing, cropping, eigenvalue decomposition, principal component analysis, and small helper tools.


- **Plots**: The `plots.py` file contains different functions for plotting the prediction horizons and loss curves.

## Getting Started

To get started, refer to the documentation and python files within the respective folders for detailed instructions on running experiments and training neural network models.

## Dependencies

Make sure you have the required Python libraries and packages installed. You can find the dependencies listed in the individual Python scripts and notebooks.

## License

This repository is licensed under [LICENSE NAME]. See the [LICENSE](LICENSE) file for details.