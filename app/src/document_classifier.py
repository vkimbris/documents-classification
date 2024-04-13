import joblib

from abc import ABC, abstractmethod

from typing import List

class BaseDocumentClassifier(ABC):
    """
    A base class for Document classification model.

    Attributes
    ----------
    model_path : str
        the location of the model in the local folder

    Methods
    -------
    predict(contents: List[str]) -> List[str]:
        Returns labels of documents contents
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    @abstractmethod
    def predict(self, contents: List[str]) -> List[str]:
        pass


class SklearnDocumentClassifier(BaseDocumentClassifier):
    """
    A class for Document classification model with Sklearn.

    Attributes
    ----------
    model_path : str
        the location of the <pipeline>.pkl in the local folder

    Methods
    -------
    predict(contents: List[str]) -> List[str]:
        Returns labels of documents contents
    """

    def __init__(self, model_path: str) -> None:
        super().__init__(model_path)

        self.pipeline = joblib.load(model_path)

    def predict(self, contents: List[str]) -> List[str]:
        return self.pipeline.predict(contents)
