import numpy as np


class RFF:
    """
    Class implementing Random Fourier Features described in Rahimi and Recht's paper
    https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf
    """
    def __init__(self, rff_dim: int, col_dim: int, sigma: float = 0.1):
        self.rff_dim = rff_dim
        self.col_dim = col_dim
        self.W = np.random.normal(loc=0, scale=1, size=(self.rff_dim, col_dim))
        self.b = np.random.uniform(0, 2*np.pi, size=(self.rff_dim,1))
        self.norm = np.sqrt(2) / np.sqrt(self.rff_dim)
        self.sigma = sigma

    def calc_z(self, x):
        z_x = self.norm * np.cos(self.sigma* np.dot(self.W ,x.T) + self.b)
        return z_x

