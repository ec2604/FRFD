import numpy as np
import rff
import fd
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.linalg import cho_solve, cho_factor
from sklearn.linear_model import Ridge


class RandomFrequentFourierDirections:
    """
    Class implementing Random Frequent Fourier Directions
    """

    def __init__(self, rff_dim: int, col_dim: int, fd_sketch_dim: int, sigma: float, noise: float):
        """

        :param rff_dim: Dimension of RFF embedding
        :param col_dim: Dimension of original data
        :param fd_sketch_dim: Sketch size for Frequent Directions
        :param sigma: Sigma for RBF kernel
        """
        self.rff_dim = rff_dim
        self.fd_sketch_dim = fd_sketch_dim
        self.noise = noise
        self.rff = rff.RFF(rff_dim=rff_dim, col_dim=col_dim, sigma=sigma)
        self.fd_sketch = fd.FrequentDirections(d=rff_dim, ell=fd_sketch_dim)
        self.z_t_y = np.zeros((self.rff_dim, 1))
        self.cho_factor = None
        self.krr_solution = None

    def append(self, x, y, rff_map=True):
        """
        Adds data from stream to sketch
        :param x: Data, either row of A or already transformed to RFF, depending on rff_map
        :param y: label
        :param rff_map: Indicates whether x is transformed or not
        """
        if not rff_map:
            z_x = self.rff.calc_z(x)
        else:
            z_x = x
        # Z_T y needs to be calculated on the fly as the stream comes in
        self.z_t_y += z_x*y
        self.fd_sketch.append(z_x)

    def get_kernel_approximation(self):
        """
        Gets FD sketch where H^TH ~= Z^TZ
        :return:
        """
        return self.fd_sketch.get()

    def set_krr_solution(self):
        """
        Solves KRR solution after stream is done. Implementation of Algorithm 1 in the paper.
        """

        l, low = cho_factor(self.fd_sketch.get().T@self.fd_sketch.get() + self.noise*np.eye(self.rff_dim), lower=True)
        self.cho_factor = (l, low)
        self.krr_solution = cho_solve((l, low), self.z_t_y)

    def calc_gpr_mean(self, test_point):
        """
        Performs inference for GP mean, implements Algorithm 2 in the paper.
        """
        return self.krr_solution.T @ self.rff.calc_z(test_point.reshape(1, -1))

    def calc_gpr_var(self, test_point):
        """
        Performs inference for GP variance, implements Algorithm 3 in the paper
        """
        test_z = self.rff.calc_z(test_point.reshape(1, -1))
        stable_kernel_inv_z_test = cho_solve(self.cho_factor, self.fd_sketch.get().T@self.fd_sketch.get()
                                             @ test_z.flatten())
        return np.linalg.norm(test_z)**2 - test_z.T @ stable_kernel_inv_z_test


def load_sgemm_dataset(subset_num=-1):
    ds = pd.read_csv('sgemm_product_dataset/sgemm_product.csv').to_numpy()
    x_y = ds[:subset_num, :15].copy()
    scaler = preprocessing.StandardScaler()
    x_y = scaler.fit_transform(x_y)
    return x_y, scaler


def calc_RMSE(rffd, X_y):
    rss = 0
    for row in X_y:
        rss += (rffd.krr_solution.T @ rffd.rff.calc_z(row[:-1].reshape(1, -1)) - row[-1]) ** 2
    return np.sqrt(rss / len(X_y))[0][0]


def generate_synthetic_dataset_and_graph():
    rff_dim = 1000
    fd_ell = 250
    sigma = 1.2
    noise = 1e-1
    a = np.array([0, 1, 2, 3, 4, 10, 17, 18, 19])
    np.sort(a)
    a = a.reshape(-1, 1)
    b = np.sin(a) + np.random.normal(0, 1, a.shape)
    rffd = RandomFrequentFourierDirections(rff_dim=rff_dim, col_dim=1, fd_sketch_dim=fd_ell, sigma=sigma, noise=noise)
    Z = rffd.rff.calc_z(a).T
    # Simulate stream

    # Calculate FRFD solution
    for i in range(len(Z)):
        rffd.append(Z[i, :].reshape(-1, 1), b[i])
    rffd.set_krr_solution()
    b_test = []
    b_std = []
    a_test = np.linspace(0, 19, 1000)
    a_test = np.sort(a_test.flatten())

    # Calculate vanilla KRR with RFF as baseline
    clf = Ridge(alpha=noise, fit_intercept=False)
    clf.fit(Z, b)

    # Calculate inference for test
    for point in a_test:
        b_test.append(rffd.calc_gpr_mean(point).reshape(-1)[0])
        b_var = np.abs(rffd.calc_gpr_var(point).reshape(-1)[0])
        if b_var <= 0:
            b_std.append(0.)
        else:
            b_std.append(np.sqrt(b_var))
    b_std = np.array(b_std)
    b_test = np.array(b_test)
    z_test = rffd.rff.calc_z(a_test.reshape(-1, 1)).T
    b_rff = clf.predict(z_test)

    plt.scatter(a, b, color='b', marker='o', label='Noisy observations')
    plt.plot(a_test, b_test, color='y', label='predictions (mean)')
    plt.plot(a_test, np.sin(a_test), color='r', linestyle='-.', label='Ground truth')
    plt.fill_between(a_test.flatten(), (b_test - 1.96*b_std).flatten(), (b_test+1.96*b_std).flatten(), color='r',
                     alpha=.5, label='confidence interval 95%')
    plt.plot(a_test, b_rff, linestyle='--', label='RFF')
    plt.legend()
    plt.title('KRR solution - RFF vs. FRFD')
    plt.savefig('FRFD_RFF.png')


def calc_rmse_sgemm_ds(fd_ell_):

    rff_dim = 300
    fd_ell = fd_ell_
    sigma = 3
    noise = 1e-3
    x_y, _ = load_sgemm_dataset(20000)
    rffd = RandomFrequentFourierDirections(rff_dim=rff_dim, col_dim=x_y.shape[1] - 1, fd_sketch_dim=fd_ell, sigma=sigma,
                                           noise=noise)

    Z = rffd.rff.calc_z(x_y[:, :-1]).T
    # Simulate stream
    for i in range(len(Z)):
        rffd.append(Z[i, :].reshape(-1, 1), x_y[i, -1])
    rffd.set_krr_solution()
    clf = Ridge(alpha=noise, fit_intercept=False)
    clf.fit(Z, x_y[:, -1:])
    y_pred = clf.predict(Z)
    rff_rmse = np.sqrt(((y_pred - x_y[:, -1]) ** 2).mean())
    print(f'RMSE for RFF: {rff_rmse}')

    # Calc RMSE for RFFD
    print(f'RMSE for RFFD: {calc_RMSE(rffd, x_y)} ')
    return calc_RMSE(rffd, x_y), np.sqrt(((y_pred - x_y[:, -1]) ** 2).mean())


def generate_performance_as_function_of_sketch_size():
    """
     Calculates performance of FRFD as a function of the sketch size
    """
    rmse_list = []
    for fd_ell in np.arange(10, 500, 50):
        rmse_list.append(calc_rmse_sgemm_ds(fd_ell)[0])
    rmse_rff = calc_rmse_sgemm_ds(fd_ell)[1]
    plt.plot(np.arange(10, 500, 50), rmse_list, label='FRFD', marker='o')
    plt.axhline(rmse_rff, 0, 1, label='RFF baseline', color='r')
    plt.xlabel('l (sketch size)')
    plt.ylabel('RMSE')
    plt.yscale('log')
    plt.legend()
    plt.title('FRFD performance as a function of sketch size')
    plt.savefig('FRFD_f_el.png')


if __name__ == '__main__':
    np.random.seed(1)
    # generate_synthetic_dataset_and_graph()
    # generate_performance_as_function_of_sketch_size()
