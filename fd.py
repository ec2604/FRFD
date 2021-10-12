import numpy as np
from scipy.linalg import svd as scipy_svd


class FrequentDirections:
    def __init__(self, d, ell):
        self.d = d
        self.ell = ell
        self.m = 2 * self.ell
        self._sketch = np.zeros((self.m, self.d))
        self.nextZeroRow = 0

    def append(self, vector):
        if self.nextZeroRow >= self.m:
            self.__rotate__()
        self._sketch[self.nextZeroRow, :] = vector.flatten()
        self.nextZeroRow += 1

    def __rotate__(self):
        try:
            [_, s, Vt] = np.linalg.svd(self._sketch, full_matrices=False)
        except np.linalg.LinAlgError as err:
            [_, s, Vt] = scipy_svd(self._sketch, full_matrices=False)

        if len(s) >= self.ell:
            sShrunk = np.sqrt(np.maximum(s[:self.ell] ** 2 - s[self.ell - 1] ** 2,0))
            self._sketch[:self.ell:, :] = np.dot(np.diag(sShrunk), Vt[:self.ell, :])
            self._sketch[self.ell:, :] = 0
            self.nextZeroRow = self.ell
        else:
            self._sketch[:len(s), :] = np.dot(np.diag(s), Vt[:len(s), :])
            self._sketch[len(s):, :] = 0
            self.nextZeroRow = len(s)

    def get(self):
        self.__rotate__()
        return self._sketch[:self.ell, :]