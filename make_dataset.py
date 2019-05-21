#!/usr/bin/python
""" make_dataset

Downloads datasets used in the demos and persist them in /data

Author: datadonk23
Date: 17.05.19 
"""

import pandas as pd
import os, logging
logging.getLogger().setLevel(logging.INFO)


root_path = os.path.abspath(os.curdir)
data_dir = os.path.join(root_path, "data")


### NLPClassif_complaints
# Consumer complaints dataset from Consumer Complaints Database, ~740MB

url = "https://data.consumerfinance.gov/api/views/s6ew-h6mp/rows.csv?accessType=DOWNLOAD"
file_name = "Consumer_Complaints.csv"
file_path = os.path.join(data_dir, file_name)
test_file_path = os.path.join(data_dir, "test_" + file_name)

try:
    logging.info("Start downloading complaints dataset")
    consumer_complaints = pd.read_csv(url, dtype=object)
    logging.info("Start persisting complaints dataset")
    consumer_complaints.to_csv(file_path, index=False)
    logging.info("Sample and persisting complaints tests dataset")
    test_sample = consumer_complaints.sample(50000, random_state=23)
    test_sample.to_csv(test_file_path, index=False)
except Exception as e:
    logging.error("Unable to make nlp_classif_complaints dataset. "
                  "Check for correct URL and path.",
                  exc_info=e)
