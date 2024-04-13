import joblib
import pandas as pd

from abc import ABC, abstractmethod

from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


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

    train(contents: List[str], labels: List[str]) -> List[str]:
        Returns labels of documents contents
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    @abstractmethod
    def predict(self, contents: List[str]) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def train(self, contents: List[str], labels: List[str]):
        raise NotImplementedError


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

    train(contents: List[str], labels: List[str]) -> List[str]:
        Returns labels of documents contents
    """

    def __init__(self, model_path: str, path_to_old_train_data: str) -> None:
        super().__init__(model_path)

        self.pipeline = joblib.load(model_path)
        self.path_to_old_train_data = path_to_old_train_data

    def predict(self, contents: List[str]) -> List[str]:
        return self.pipeline.predict(contents)
    
    def train(self, new_train_data: pd.DataFrame):
        old_train_data = pd.read_csv(self.path_to_old_train_data)

        new_train_data = pd.concat([
            old_train_data, new_train_data
        ])

        vectorizer = TfidfVectorizer().fit(new_train_data.text)

        X_train = vectorizer.transform(new_train_data.text)
        y_train = new_train_data["class"]

        model = LogisticRegression().fit(X_train, y_train)

        pipe = Pipeline(steps=[
            ("Vectorizer", vectorizer), ("LogisticRegression", model)
        ])

        self.pipeline = pipe


