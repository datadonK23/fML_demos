#!/usr/bin/python
""" util_nlp_classif_complaints

Util methods for nlp_classif_complaints demo.

Author: datadonk23
Date: 21.05.19 
"""

from typing import Dict
import pandas as pd
from fklearn.validation.evaluators import precision_evaluator, \
    recall_evaluator, fbeta_score_evaluator


def load_data(filepath: str) -> pd.DataFrame:
    """ Loads dataset into memory.

    Parameters
    ----------
    filepath : str
        Path to data file.

    Returns
    -------
    data : pd.DataFrame
        Raw dataset.

    """

    data = pd.read_csv(filepath, dtype=object,
                       usecols=["Product", "Consumer complaint narrative",
                                "Date received", "Complaint ID"],
                       parse_dates=["Date received"])

    return data


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """ Cleans given dataframe.

    Renames column names. Generates target column. Drops NaN's.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Returns
    -------
    cleaned_data : pd.DataFrame
        Cleaned dataset.

    """
    data = df.rename(columns={"Product": "product",
                              "Consumer complaint narrative": "text",
                              "Date received": "input_time",
                              "Complaint ID": "id"})

    data["target"] = (
            data["product"] ==
            "Credit reporting, credit repair services, or other personal consumer reports"
    ).astype(int)

    cleaned_data = data.dropna()

    return cleaned_data


def print_eval(pred: pd.DataFrame, name: str, log: Dict) -> None:
    """ Prints evaluation metrices to StdOut.

    Most side-effecting output of evaluation metrices (Precision, Recall &
    F1-Score.

    Parameters
    ----------
    pred : pd.DataFrame
        Predicted values
    name : str
        Model name
    log : Dict
        Model training log

    Returns
    -------
    None

    """

    print("Model: {}".format(name))
    print("Parameters", log["nlp_logistic_classification_learner"][
        "parameters"])
    precision = precision_evaluator(pred)
    print("Precision", precision["precision_evaluator__target"])
    recall = recall_evaluator(pred)
    print("Recall", recall["recall_evaluator__target"])
    f1_score = fbeta_score_evaluator(pred)
    print("F1 Score", f1_score["fbeta_evaluator__target"])
    print("Training time",
          log["nlp_logistic_classification_learner"]["running_time"],
          "\n")
