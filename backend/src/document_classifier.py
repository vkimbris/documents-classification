import joblib
import pandas as pd

from abc import ABC, abstractmethod

from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from typing import Tuple


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

    def __init__(self, 
                 model_path: str, 
                 path_to_old_train_data: str,
                 random_state: int,
                 test_size: float) -> None:
        
        super().__init__(model_path)

        self.pipeline = joblib.load(model_path)
        self.path_to_old_train_data = path_to_old_train_data

        self.random_state = random_state
        self.test_size = test_size

    def predict(self, contents: List[str]) -> List[str]:
        return self.pipeline.predict(contents)
    
    def train(self, new_data: pd.DataFrame):
        old_data = pd.read_csv(self.path_to_old_train_data)

        new_data = pd.concat([
            old_data, new_data
        ])

        vectorizer = TfidfVectorizer().fit(new_data.text)

        train, test = train_test_split(new_data, test_size=self.test_size, random_state=self.random_state, stratify=new_data["class"])

        X_train = vectorizer.transform(train.text)
        y_train = train["class"]

        X_test = vectorizer.transform(test.text)
        y_test = test["class"]

        model = LogisticRegression().fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics = classification_report(y_test, y_pred, output_dict=True)

        pipe = Pipeline(steps=[
            ("Vectorizer", vectorizer), ("LogisticRegression", model)
        ])

        self.pipeline = pipe

        return metrics



