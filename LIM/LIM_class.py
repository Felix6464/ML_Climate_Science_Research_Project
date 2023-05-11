import utils as ut
import numpy as np

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
        self.logarithmic_matrix = None
        self.noise_covariance = None

    def fit(self, data):
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
        self.green_function = self.C_tau @ np.linalg.inv(self.C_0)

        # Compute Logarithmic Matrix = ln(green_function)/tau
        eigenvalues, eigenvectors, _ = ut.matrix_decomposition(self.green_function)
        print("Eigenvalues - real: {}".format(eigenvalues.real))

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
            self.logarithmic_matrix = self.logarithmic_matrix.real
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
        noise_covariance = -self.logarithmic_matrix @ self.C_0 - self.C_0 @ self.logarithmic_matrix.T

        # Check if the covariance matrix has negative values
        if np.min(noise_covariance) < -1e-5:
            print(f"WARNING: Covariance matrix has negative values!")

        # Check for Nyquist mode
        # If the imaginary part of the largest eigenvalue of Q is too large, print a warning message
        eigenvalues_noise, _, _ = ut.matrix_decomposition(noise_covariance)

        # Compute the eigenvalues of Q
        eps = 1e-5
        if np.max(np.abs(np.imag(eigenvalues_noise))) > eps:
            print(f"WARNING: Imaginary part of max eigval of Q is {np.max(np.abs(np.imag(eigenvalues_noise)))}!")

        return noise_covariance

    def forecast_mean(self, x, lag=1):
        """Forecast the mean of x at time t+tau using the Green's function G.

        Args:
            x (np.ndarray): Input data to estimate Green's function from.
                Dimensions (n_components, n_time).
            lag (int): Time-lag for forecasting.

        Returns:
            x_frcst (np.ndarray): Forecast.
                Dimensions (n_components, n_time).
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

        print("G_tau: {}".format(G_tau))
        # Compute the forecast x(t+tau) = G_tau * x(t).
        x_frcst = np.einsum('ij,jk', np.real(G_tau), x)

        return x_frcst

    def euler_maruyama(self, x, dt, n_samples=1):
        """Simulate the system using the Euler-Maruyama method.

        Args:
            x (np.ndarray): Input data to estimate Green's function from.
                Dimensions (n_components, n_time).
            dt (float): Time step.
            n_samples (int): Number of samples to simulate.

        Returns:
            x_sim (np.ndarray): Simulated data.
                Dimensions (n_components, n_time).
        """
        pass

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
        print("Frobenius norm: {}".format(fro_norm))

    def G_eigenvalue_check(self, eigenvalues):

        # Check for stability -> eigenvalues of G should be between 0 and 1
        if min(eigenvalues.real) < 0:
            print("WARNING: Negative eigenvalues detected.")
            print("WARNING: The logarithmic matrix may not be stable.")
            print("WARNING: Consider using a larger time-lag.")
        if max(eigenvalues.real) > 1:
            print("WARNING: Eigenvalues greater than 1 detected.")