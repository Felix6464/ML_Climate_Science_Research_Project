import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns


def plot_model_forecast(lstm_model, train_data, target_data, test_data, test_target, rand, num_rows=5):
    """
    Plot examples of the LSTM encoder-decoder evaluated on the training/test data.

    Args:
        lstm_model (LSTM): Trained LSTM encoder-decoder model.
        train_data (np.array): Windowed training input data.
        target_data (np.array): Windowed training target data.
        test_data (np.array): Windowed test input data.
        test_target (np.array): Windowed test target data.
        rand: Identifier.
        num_rows (int): Number of training/test examples to plot.

    Returns:
        None
    """

    train_data = train_data.detach().cpu()
    target_data = target_data.detach().cpu()
    test_data = test_data.detach().cpu()
    test_target = test_target.detach().cpu()

    print("Xtrain.shape: ", train_data.shape)
    print("Ytrain.shape: ", test_target.shape)

    # Input nd output window size
    input_window = train_data.shape[0]
    output_window = test_target.shape[0]

    fig, ax = plt.subplots(num_rows, 2, figsize=(13, 15))

    # Plot training/test predictions for a manually set index i (for better visualization)
    i = 300

    # Plot training/test predictions
    for row in range(num_rows):
        # Train set
        i += 20
        x_train = train_data[:, row+i, :]
        y_train_pred = lstm_model.predict(x_train, target_len=output_window, prediction_type="forecast").cpu()

        ax[row, 0].plot(np.arange(0, input_window), train_data[:, row+i, 0].cpu(), 'k', linewidth=2, label='Input')
        ax[row, 0].plot(np.arange(input_window - 1, input_window + output_window),
                        np.concatenate([[train_data[-1, row+i, 0]], target_data[:, row+i, 0]]),
                        color="blue", linewidth=2, label='Target')
        ax[row, 0].plot(np.arange(input_window - 1, input_window + output_window),
                        np.concatenate([[train_data[-1, row+i, 0]], y_train_pred[:, 0, 0]]),
                        color="red", linewidth=2, label='Prediction')
        ax[row, 0].set_xlim([0, input_window + output_window - 1])
        ax[row, 0].set_xlabel('$Timestep$')
        ax[row, 0].set_ylabel('$Prediction Value$')

        # Test set
        x_test = test_data[:, row+i, :]
        y_test_pred = lstm_model.predict(x_test, target_len=output_window, prediction_type="forecast").cpu()
        ax[row, 1].plot(np.arange(0, input_window), test_data[:, row+i, 0], 'k', linewidth=2, label='Input')
        ax[row, 1].plot(np.arange(input_window - 1, input_window + output_window),
                        np.concatenate([[test_data[-1, row+i, 0]], test_target[:, row+i, 0]]),
                        color="blue", linewidth=2, label='Target')
        ax[row, 1].plot(np.arange(input_window - 1, input_window + output_window),
                        np.concatenate([[test_data[-1, row+i, 0]], y_test_pred[:, 0, 0]]),
                        color="red", linewidth=2, label='Prediction')
        ax[row, 1].set_xlim([0, input_window + output_window - 1])
        ax[row, 1].set_xlabel('$Timestep$')
        ax[row, 1].set_ylabel('$Prediction Values$')

        if row == 0:
            ax[row, 0].set_title('Prediction on Train Data')
            ax[row, 1].legend(bbox_to_anchor=(1, 1))
            ax[row, 1].set_title('Prediction on Test Data')

    plt.suptitle('LSTM Encoder-Decoder Predictions', x=0.445, y=1., fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(f'trained_models/predictions_{rand}.png')
    plt.show()
    plt.close()

    return

def plot_model_forecast_PC(lstm_model, train_data, target_data, test_data, test_target, rand, num_rows=5):
    """
    Plot examples of the LSTM encoder-decoder evaluated on the training/test data.

    Args:
        lstm_model (LSTM): Trained LSTM encoder-decoder model.
        train_data (np.array): Windowed training input data.
        target_data (np.array): Windowed training target data.
        test_data (np.array): Windowed test input data.
        test_target (np.array): Windowed test target data.
        rand: Identifier.
        num_rows (int): Number of training/test examples to plot.

    Returns:
        None
    """

    train_data = train_data.detach().cpu()
    train_data = train_data.view(train_data.shape[2], train_data.shape[0], train_data.shape[1])
    target_data = target_data.detach().cpu()
    target_data = target_data.view(target_data.shape[2], target_data.shape[0], target_data.shape[1])
    test_data = test_data.detach().cpu()
    test_data = test_data.view(test_data.shape[2], test_data.shape[0], test_data.shape[1])
    test_target = test_target.detach().cpu()
    test_target = test_target.view(test_target.shape[2], test_target.shape[0], test_target.shape[1])

    print("Xtrain.shape: ", train_data.shape)
    print("Ytrain.shape: ", test_target.shape)

    # Input nd output window size
    input_window = train_data.shape[0]
    output_window = test_target.shape[0]

    fig, ax = plt.subplots(num_rows, 2, figsize=(13, 15))

    # Plot training/test predictions for a manually set index i (for better visualization)
    i = 300

    # Plot training/test predictions
    for row in range(num_rows):
        # Train set
        x_train = train_data[:, row, :]
        y_train_pred = lstm_model.predict(x_train, target_len=output_window, prediction_type="forecast").cpu()

        ax[row, 0].plot(np.arange(0, input_window), train_data[:, 0, row].cpu(), 'k', linewidth=2, label='Input')
        ax[row, 0].plot(np.arange(input_window - 1, input_window + output_window),
                        np.concatenate([[train_data[-1, 0, row]], target_data[:, 0, row]]),
                        color="blue", linewidth=2, label='Target')
        ax[row, 0].plot(np.arange(input_window - 1, input_window + output_window),
                        np.concatenate([[train_data[-1, 0, row]], y_train_pred[:, 0, row]]),
                        color="red", linewidth=2, label='Prediction')
        ax[row, 0].set_xlim([0, input_window + output_window - 1])
        ax[row, 0].set_xlabel('$Timestep$', fontsize=15)
        ax[row, 0].set_ylabel('$Prediction Value$', fontsize=15)

        # Test set
        x_test = test_data[:, row, :]
        y_test_pred = lstm_model.predict(x_test, target_len=output_window, prediction_type="forecast").cpu()
        ax[row, 1].plot(np.arange(0, input_window), test_data[:, 0, row], 'k', linewidth=2, label='Input')
        ax[row, 1].plot(np.arange(input_window - 1, input_window + output_window),
                        np.concatenate([[test_data[-1, 0, row]], test_target[:, 0, row]]),
                        color="blue", linewidth=2, label='Target')
        ax[row, 1].plot(np.arange(input_window - 1, input_window + output_window),
                        np.concatenate([[test_data[-1, 0, row]], y_test_pred[:, 0, row]]),
                        color="red", linewidth=2, label='Prediction')
        ax[row, 1].set_xlim([0, input_window + output_window - 1])
        ax[row, 1].set_xlabel('$Timestep$', fontsize=15)
        ax[row, 1].set_ylabel('$Prediction Values$', fontsize=15)

        if row == 0:
            ax[row, 0].set_title('Prediction on Train Data', fontsize=15)
            ax[row, 1].legend(bbox_to_anchor=(1, 1))
            ax[row, 1].set_title('Prediction on Test Data', fontsize=15)

    plt.suptitle('LSTM Encoder-Decoder Predictions', x=0.445, y=1.)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(f'trained_models/predictions_PC_{rand}.png')
    plt.show()
    plt.close()

    return



def plot_loss(loss_train, loss_eval, identifier, loss_type):
    """
    Plot the loss values over epochs.

    Args:
        loss_values (list): List of loss values.
        identifier: Identifier for the plot.
        loss_type (str): Type of loss (e.g., training loss, validation loss).

    Returns:
        None
    """
    epochs = range(1, len(loss_train) + 1)
    plt.plot(epochs, loss_train, 'b-', marker='o', markersize=5, linewidth=1, label='Loss-Train')
    plt.plot(epochs, loss_eval, color="r", marker='o', markersize=5, linewidth=1, label='Loss-Eval')
    plt.title(f'{loss_type} per Epoch', fontsize=16, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()  # Adjust spacing and margins
    plt.savefig(f'trained_models/loss_{loss_type}_combined_{identifier}.png', dpi=300)
    plt.show()

import matplotlib.pyplot as plt

def plot_loss_horizon_combined(loss_values_lstm, loss_values_lstm_inp, loss_values_ffn, identifier, loss_type):
    """
    Plot the loss values over epochs.

    Args:
        loss_values (list): List of loss values.
        identifier: Identifier for the plot.
        loss_type (str): Type of loss (e.g., training loss, validation loss).

    Returns:
        None
    """
    epochs = range(1, len(loss_values_lstm) + 1)
    plt.plot(epochs, loss_values_lstm, 'b-', marker='o', markersize=5, linewidth=1, label='Loss-LSTM')
    plt.plot(epochs, loss_values_lstm_inp, 'g-', marker='o', markersize=5, linewidth=1, label='Loss-LSTM-Input')
    plt.plot(epochs, loss_values_ffn, 'r-', marker='o', markersize=5, linewidth=1, label='Loss-FFN')
    plt.title(f'{loss_type} per Horizon Length on Test Set', fontsize=16, fontweight='bold')
    plt.xlabel('Prediction Horizon', fontsize=12)
    plt.ylabel('Test Loss', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()  # Adjust spacing and margins
    plt.savefig(f'trained_models/loss_{loss_type}_horizon_combined_{identifier}.png', dpi=300)
    plt.show()

def plot_loss_horizon(loss_values, identifier, loss_type):
    """
    Plot the loss values over epochs.

    Args:
        loss_values (list): List of loss values.
        identifier: Identifier for the plot.
        loss_type (str): Type of loss (e.g., training loss, validation loss).

    Returns:
        None
    """
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, 'b-', marker='o', markersize=5, linewidth=1, label='Loss')
    plt.title(f'{loss_type} per Horizon Length on Test Set', fontsize=16, fontweight='bold')
    plt.xlabel('Prediction Horizon', fontsize=12)
    plt.ylabel('Test Loss', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()  # Adjust spacing and margins
    plt.savefig(f'trained_models/loss_{loss_type}_horizon_{identifier}.png', dpi=300)
    plt.show()


def plot_correlation_heatmap(correlation_matrix):
    """
    Plot a correlation heatmap based on a correlation matrix.

    Args:
        correlation_matrix (pandas.Series): Pandas Series representing the correlation matrix.

    Returns:
        None
    """
    # Convert the correlation series to a DataFrame
    correlation_df = correlation_matrix.unstack().reset_index()
    correlation_df.columns = ['Feature 1', 'Feature 2', 'Correlation']

    # Pivot the DataFrame to create a correlation matrix
    correlation_matrix = correlation_df.pivot('Feature 1', 'Feature 2', 'Correlation')

    # Plot the correlation heatmap using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()