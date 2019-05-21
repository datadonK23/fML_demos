#!/usr/bin/python
""" nlp_classif_complaints

Binary classification of user complaints. Predicts whether user asks about
'Credit reporting, credit repair services, or other personal consumer reports'
or not.

Modified from NLP Classification demo of fklearn.
(https://fklearn.readthedocs.io/en/latest/examples/nlp_classification.html)

Author: datadonk23
Date: 21.05.19
"""

import os, logging
logging.getLogger().setLevel(logging.INFO)

from fklearn.preprocessing.splitting import time_split_dataset
from fklearn.training.classification import nlp_logistic_classification_learner

from utils.util_nlp_classif_complaints import load_data, clean_data, print_eval


# Load and clean dataset
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

dev_pred_baseline = baseline_p_fn(dev)
print_eval(dev_pred_baseline, "Baseline NLP logistic classification learner",
           baseline_log)

dev_pred = p_fn(dev)
print_eval(dev_pred, "Tuned NLP logistic classification learner", log)
