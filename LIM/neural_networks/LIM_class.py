import torch

from LIM.neural_networks import utilities as ut
import numpy as np
from numpy.linalg import pinv, eigvals, eig, eigh
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


# Implementing Linear inverse model (LIM)
'''
The LIM class creates a Linear Inverse Model for time series analysis.

The class has one input parameter, tau, which represents the time lag between the input and output data.

The class has three class variables that are initialized as None:

G: Green's function
L: Logarithmic matrix
Q: Noise covariance matrix

The class has a fit method that takes in data, which is a numpy array with dimensions (n_components, n_time).
 The method computes the C_0 and C_tau covariance matrices of the input data, and then computes the time-evolution operator G
  using these covariance matrices.

The method then performs a matrix decomposition on G to obtain its eigenvectors and eigenvalues, and sorts them in descending order
 based on their decay time. Using these sorted eigenvectors and eigenvalues, it computes the logarithmic matrix L,
  which represents the linear relationship between the input and output data.

The method checks for the existence of a Nyquist mode, which occurs when the imaginary part of L is greater than a small epsilon value. 
If a Nyquist mode exists, a warning message is printed and the imaginary part is kept in the L matrix. If not, only the real part of L is stored. 
Finally, the method computes the noise covariance matrix Q using the C_0 and L matrices.

The fit method does not return any value, but it stores the computed L and Q matrices as class variables.

Eigenvectors are vectors that satisfy the equation Ax = λx for some scalar λ. 
They represent the directions in which the linear transformation represented by A acts by simply scaling the vector.

Eigenvalues are the scalars λ that satisfy the equation Ax = λx for some vector x. 
They represent the magnitude of the linear transformation along the eigenvectors.

'''


class LIM:
    """
    Linear Inverse Model for time series analysis.

    Args:
        tau (int): Time-lag for the model.
    """

    def __init__(self, tau) -> None:
        # Initialize time-lag
        self.tau = tau

        # Initialize class variables
        self.green_function = None
        self.g_eigenvalues = None
        self.logarithmic_matrix = None
        self.noise_covariance = None

    def fit(self, data, eps=0.01):
        """
        Fit LIM to data.

        Args:
            data (np.ndarray): Input data to estimate Green's function from.
                Dimensions (n_components, n_time).
        """
        # Split data into x and x_tau
        x = data[:, :-self.tau]
        x_tau = data[:, self.tau:]
        assert x.shape == x_tau.shape

        # Compute the number of time points
        n_time = data.shape[1] - self.tau

        # Compute covariance matrices C_0 and C_tau
        self.C_0 = (x @ x.T) / n_time
        self.C_tau = (x_tau @ x.T) / n_time

        # Compute Green's function as C_tau x the inverse of C_0
        # time-evolution operator
        #print("Min eigenvalue of C_0:", np.min(np.linalg.inv(self.C_0)))
        #self.C_0 = self.C_0 + eps
        #print("Min eigenvalue of C_0:", np.min(np.clip(np.linalg.inv(self.C_0), a_min=0.00001, a_max=None)))

        self.green_function = self.C_tau @ np.linalg.inv(self.C_0)
        #print("Min of G:", np.min(self.green_function))

        # Compute Logarithmic Matrix = ln(green_function)/tau
        eigenvalues, eigenvectors, _ = ut.matrix_decomposition(self.green_function)
        self.g_eigenvalues = eigenvalues
        #print("Min eigenvalue of G:", np.min(self.g_eigenvalues))


        self.G_tau_independence_check(data)
        self.G_eigenvalue_check(eigenvalues)

        # Sort eigenvalues and eigenvectors in descending order based on decay time
        decay_time = -self.tau / np.log(eigenvalues)
        idx_sort = np.argsort(decay_time)[::-1]
        eigenvalues = eigenvalues[idx_sort]
        eigenvectors = eigenvectors[:, idx_sort]

        # Compute logarithmic matrix
        log_eigenvalues = np.diag(np.log(eigenvalues) / self.tau)
        self.logarithmic_matrix = eigenvectors @ log_eigenvalues @ np.linalg.inv(eigenvectors)

        # Check for Nyquist mode
        eps = 1e-5
        if np.max(np.abs(np.imag(self.logarithmic_matrix))) > eps:
            print("WARNING: Risk of nyquist mode.")
            print(f"WARNING: The imaginary part of L is {np.max(np.abs(np.imag(self.logarithmic_matrix)))}!")
            print(f"WARNING: Eigenvalues of G are [{np.min(eigenvalues)}, {np.max(eigenvalues)}]!")
            self.logarithmic_matrix = self.logarithmic_matrix
        else:
            self.logarithmic_matrix = np.real(self.logarithmic_matrix)

        # Compute noise covariance matrix Q
        self.noise_covariance = self.noise_covariance_func()

    def noise_covariance_func(self):
        """
        Estimate the noise covariance matrix.

        Assumes that the system is stationary, and estimates the noise covariance by solving the Lyapunov equation 0 = L @ C_0 + C_0 @ L.T + Q, where L is the time derivative of the Green's function and C_0 is the covariance matrix of the input data.

        Returns:
            Q (np.ndarray): Estimated noise covariance matrix of dimensions (n_components, n_components).
        """

        # Estimate the noise covariance using the Lyapunov equation
        noise_covariance = -(self.logarithmic_matrix @ self.C_0 + self.C_0 @ self.logarithmic_matrix.T)

        # Check if the covariance matrix has negative values
        if np.min(noise_covariance) < -1e-5:
            print(f"Covariance matrix has negative values!")

        # Check for Nyquist mode
        # If the imaginary part of the largest eigenvalue of Q is too large, print a warning message
        eigenvalues_noise, _, _ = ut.matrix_decomposition(noise_covariance)

        return noise_covariance

    def forecast(self, input_data, forecast_leads):
        """Forecast of input_data at time specified by the forecast_lead times
         using the Green's function G.

        Args:
            input_data (np.ndarray): Input data to estimate Green's function from.
                Dimensions (n_components, n_time).
            forecast_leads List<int>: Time-lag for forecasting. Each value is interpreted as a tau value
                                      for which the forecast matrix is determined as G_1^tau

        Returns:
            x_frcst (np.ndarray): Forecast.
                Dimensions (n_components, n_time).
        """

        print('Performing LIM forecast for tau values: '
                    + str(forecast_leads))

        num_forecast_times = len(forecast_leads)

        try:
            forecast_output_shape = (num_forecast_times, input_data.shape[0], input_data.shape[1])
        except:
            forecast_output_shape = (num_forecast_times, input_data.shape[0])

        forecast_output = np.zeros(forecast_output_shape)

        for i, tau in enumerate(forecast_leads):
            #g_tau = np.linalg.matrix_power(self.green_function, tau)
            g_tau = self.get_G_tau(tau)

            forecast = np.einsum('ij,jk', np.real(g_tau), input_data)
            forecast_output[i] = forecast

        return forecast_output

    def noise_integration(self, input_data, timesteps, out_arr=None, seed=None, num_comp=10):

        """Perform a numerical integration forced by stochastic noise

        Performs LIM forecast over the times specified by the
        forecast_lead times.

        Parameters
        ----------
        input_data (np.ndarray):
            Input data to estimate Green's function from.
            Dimensions (n_components, n_time).
        out_arr: Optional, ndarray
            Optional output container for data at the resolution of deltaT.
            Expected dimensions of (timesteps + 1, input_data.shape[0], input_data.shape[0])
        timesteps: int
            Number of timesteps in a single tau segment of the noise
            integration.
            E.g., for tau=1-year, 1440 timesteps is ~6hr timestep.
        seed: Optional, int
            Seed for the random number generator to perform a reproducible
            stochastic forecast

        Returns
        -----
        ndarray
            Final state of the LIM noise integration forecast. Same dimension
            as input_data.
        """

        if seed is not None:
            np.random.seed(seed)

        state_start = np.array(input_data)
        print("state_start: {} and type : {}".format(state_start.shape[0], type(state_start)))
        out_arr = np.zeros((timesteps + 1, state_start.shape[0]))


        out_arr[0] = state_start

        t_decay = [-(1 / np.log(eigenvalue.real)) for eigenvalue in self.g_eigenvalues]
        t_decay = min(t_decay) -1e-5

        if 2 < t_decay:
            t_delta_int = 1
        else:
            t_delta_int = t_decay

        print("t_delta: {}".format(t_delta_int))


        for t in range(timesteps):
            deterministic_part = np.array((self.logarithmic_matrix @ state_start) * t_delta_int)
            random_part = np.array(
                np.random.multivariate_normal([0 for n in range(num_comp)], self.noise_covariance))
            stochastic_part = np.array(random_part * np.sqrt(t_delta_int))

            state_new = state_start + deterministic_part + stochastic_part
            state_mid = (state_start + state_new) / 2
            state_start = state_new

            out_arr[t + 1] = state_mid
            times = np.arange(timesteps + 1) * t_decay

        return out_arr, times

    def get_noise_eigenvalues(self):

        q_eigenvalues, q_eigenvectors = eigh(self.noise_covariance)

        sort_idx = q_eigenvalues.argsort()
        q_eigenvalues = q_eigenvalues[sort_idx][::-1]
        q_eigenvectors = q_eigenvectors[:, sort_idx][:, ::-1]

        num_neg = (q_eigenvalues < 0).sum()
        max_neg_vals = 5

        if num_neg > 0:
            num_left = len(q_eigenvalues) - num_neg
            if num_neg > max_neg_vals:
                print('Found {:d} modes with negative eigenvalues in'
                             ' the noise covariance term, Q.'.format(num_neg))
                raise ValueError('More than {:d} negative eigenvalues of Q '
                                 'detected.  Consider further dimensional '
                                 'reduction.'.format(max_neg_vals))

            else:
                print('Removing negative eigenvalues and rescaling {:d} '
                            'remaining eigenvalues of Q.'.format(num_left))
                pos_q_evals = q_eigenvalues[q_eigenvalues > 0]
                scale_factor = q_eigenvalues.sum() / pos_q_evals.sum()
                print('Q eigenvalue rescaling: {:1.2f}'.format(scale_factor))

                q_eigenvalues = q_eigenvalues[:-num_neg] * scale_factor
                q_eigenvectors = q_eigenvectors[:, :-num_neg]
        else:
            scale_factor = None

        return q_eigenvalues, q_eigenvectors, scale_factor

    def get_G_tau(self, tau, lag=1):
        """
        Compute the Green's function G at time tau.
        """

        # Compute the matrix decomposition of G.
        eigenvalues, eigenvectors_left, eigenvectors_right = ut.matrix_decomposition(self.green_function)

        # Sort the eigenvalues in decreasing order of decay time.
        decay_times = -self.tau / np.log(eigenvalues)
        sorted_indices = np.argsort(decay_times)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors_left = eigenvectors_left[:, sorted_indices]
        eigenvectors_right = eigenvectors_right[:, sorted_indices]

        # Compute weights to normalize eigenvectors left and right.
        weights = eigenvectors_left.T @ eigenvectors_right
        eigenvectors_left_norm = eigenvectors_left @ np.linalg.inv(weights)

        # Compute the Green's function at time tau*lag.
        G_tau = eigenvectors_left_norm @ np.diag(eigenvalues ** (lag / self.tau)) @ eigenvectors_right.T

        return G_tau

    def G_tau_independence_check(self, data):

        x_1 = data[:, :-(self.tau+1)]
        x_tau_1 = data[:, (self.tau+1):]
        assert x_1.shape == x_tau_1.shape

        n_time_1 = data.shape[1] - (self.tau+1)

        C_0_1 = (x_1 @ x_1.T) / n_time_1
        C_tau_1 = (x_tau_1 @ x_1.T) / n_time_1

        green_function_1 = C_tau_1 @ np.linalg.inv(C_0_1)

        # Check if G is independent of tau -> eigenvalues should be equal
        # Compute the Frobenius norm of the difference between G and G_1
        fro_norm = np.linalg.norm(self.green_function - green_function_1, ord='fro')
        #print("Frobenius norm: {}".format(fro_norm))

    def G_eigenvalue_check(self, eigenvalues):

        # Check for stability -> eigenvalues of G should be between 0 and 1
        if min(eigenvalues.real) < 0:
            print("Negative eigenvalues detected.")
        if max(eigenvalues.real) > 1:
            print("Eigenvalues greater than 1 detected.")




if __name__ == "__main__":
    model = LIM(1)

