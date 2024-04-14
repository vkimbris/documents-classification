import pandas as pd
from typing import Dict, Union, List
import requests

from src.constants import SERVER_API



def get_labels() -> List[str]:
    """
    Retrieves available labels from the server API using GET request.

    This function is cached by Streamlit and will run only once until its output changes or the app restarts.

    Returns
    -------
    List[str]
        A list of available labels returned by the server API.
    """
    available_labels = requests.request("GET", f"{SERVER_API}/labels").json()["labels"]
    return available_labels


def create_classification_report(classification_report_dict: Dict[str, Union[float, int]]) -> str:
    """
    Create the classification report from the dictionary provided.

    The classification_report_dict should contain classification metrics (precision, recall, f1-score, support) 
    for each class label and 'accuracy', 'macro avg', 'weighted avg' metrics.

    Parameters
    ----------
    classification_report_dict: Dict[str, Union[float, int]]
        This dictionary should follow the structure of the dictionary output by sklearn.metrics.classification_report 
        with output_dict=True. It's a dictionary with class labels as keys and dictionaries of metrics as values.  
        
    Returns
    -------
    str
        Returns a string containing the markdown table representation of the classification report.

    """
    df = pd.DataFrame(classification_report_dict).transpose()
    return df.to_markdown()



def create_dataframe(count_labels: Dict[str, int], bool_labels: Dict[str, bool]) -> pd.DataFrame:
    """
    Creates a dataframe from the input label dictionaries (count_labels and bool_labels).
    
    Parameters
    ----------
    count_labels : Dict[str, int]
        A dictionary where keys are labels and values are their counts.
    bool_labels : Dict[str, bool]
        A dictionary where keys are labels and values are their statuses (True if present, else False).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing 'Статус' and 'Количество' for each label.
    """
    combined_dict = {k: [bool_labels.get(k, False), count_labels.get(k, 0)] 
                     for k in set(count_labels) | set(bool_labels)}

    df = pd.DataFrame.from_dict(combined_dict, orient='index', columns=['Статус', 'Количество'])
    df.style.set_properties(**{"vertical-align": "text-top"})
    return df


def get_status(status: bool) -> str:
    """
    Returns a check mark if the input status is True, otherwise returns a cross mark.

    Parameters
    ----------
    status : bool
        The status to convert.

    Returns
    -------
    str
        '✅' if status is True, '❌' if status is False.
    """
    if status:
        symbol = '✅'
    else:
        symbol = '❌'      

    return symbol