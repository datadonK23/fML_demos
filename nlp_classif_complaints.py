#!/usr/bin/python
""" nlp_classif_complaints

Binary classification of user complaints. Predicts whether user asks about
'Credit reporting, credit repair services, or other personal consumer reports'
or not.

Modified from NLP Classification demo of fklearn.
(https://fklearn.readthedocs.io/en/latest/examples/nlp_classification.html)

Author: datadonk23
Date: 17.05.19 
"""

import pandas as pd
from fklearn.preprocessing.splitting import time_split_dataset
from fklearn.training.classification import nlp_logistic_classification_learner
from fklearn.validation.evaluators import precision_evaluator, \
    recall_evaluator, fbeta_score_evaluator
import os, logging
logging.getLogger().setLevel(logging.INFO)


# Load and clean dataset
def load_data(filepath):
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


def clean_data(df):
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


path = "data/"
file = "Consumer_Complaints.csv"
filepath = os.path.join(path, file)

logging.info("Loading and data cleaning")
data = clean_data(load_data(filepath))


# Time-based Train/Dev split
logging.info("Perform Train/Dev split")
train, dev = time_split_dataset(data, time_column="input_time",
                                train_start_date="2017-01-01",
                                train_end_date="2018-01-01",
                                holdout_end_date="2019-01-01")
logging.info("Train set (Complaints 2017): {}".format(train.shape))
logging.info("Dev set (Complaints 2018): {}".format(dev.shape))


# Train
logging.info("Start baseline training")
baseline_p_fn, _, baseline_log = nlp_logistic_classification_learner(
    train,
    logistic_params={"solver": "liblinear"},
    text_feature_cols=["text"],
    target="target"
)
logging.info("Finished baseline training")

logging.info("Start model training")
p_fn, train_pred, log = nlp_logistic_classification_learner(
    train,
    vectorizer_params={
        "strip_accents": "unicode",
        "stop_words": "english"
    },
    logistic_params={
        "solver": "liblinear",
        "class_weight": "balanced"
    },
    text_feature_cols=["text"],
    target="target"
)
logging.info("Finished model training")


# Evaluation
logging.info("Evaluating models")

dev_pred = p_fn(dev)
dev_pred_baseline = baseline_p_fn(dev)

print("Baseline model:")
print("Parameters", baseline_log["nlp_logistic_classification_learner"][
    "parameters"])
precision_baseline = precision_evaluator(dev_pred_baseline)
print("Precision", precision_baseline["precision_evaluator__target"])
recall_baseline = recall_evaluator(dev_pred_baseline)
print("Recall", recall_baseline["recall_evaluator__target"])
f1_score_baseline = fbeta_score_evaluator(dev_pred_baseline)
print("F1 Score", f1_score_baseline["fbeta_evaluator__target"], "\n")

print("Model:")
print("Parameters", log["nlp_logistic_classification_learner"]["parameters"])
precision = precision_evaluator(dev_pred)
print("Precision", precision["precision_evaluator__target"])
recall = recall_evaluator(dev_pred)
print("Recall", recall["recall_evaluator__target"])
f1_score = fbeta_score_evaluator(dev_pred)
print("F1 Score", f1_score["fbeta_evaluator__target"])
