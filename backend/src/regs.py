import numpy as np

from abc import ABC
from abc import abstractmethod

from numpy import ndarray

class BaseRegularizer(ABC):

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    @abstractmethod
    def get_component(self, phi: np.ndarray | None = None, theta: np.ndarray | None = None) -> np.ndarray | float:
        raise NotImplementedError


class SparsePhiRegularizer(BaseRegularizer):

    def __init__(self, alpha: float) -> None:
        super().__init__(alpha)

    def get_component(self, phi: np.ndarray | None = None, theta: np.ndarray | None = None) -> np.ndarray | float:
        return -self.alpha
    

class SparseThetaRegularizer(BaseRegularizer):

    def __init__(self, alpha: float) -> None:
        super().__init__(alpha)

    def get_component(self, phi: ndarray | None = None, theta: ndarray | None = None) -> ndarray | float:
        return -self.alpha
    

class CovariancePhiRegurlarizer(BaseRegularizer):

    def __init__(self, alpha: float) -> None:
        super().__init__(alpha)

    def get_component(self, phi: ndarray | None = None, theta: ndarray | None = None) -> ndarray | float:
        tmp = phi.sum(axis=0)
        tmp = np.tile(tmp, (phi.shape[0], 1))
        tmp = tmp - phi

        return -self.alpha * phi * tmp
