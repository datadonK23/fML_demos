#!/usr/bin/python
""" test_nlp_classif_complaints

Author: datadonk23
Date: 21.05.19 
"""

import unittest
import unittest.mock
from io import StringIO
import os
import numpy as np
import pandas as pd

from utils.util_nlp_classif_complaints import load_data, clean_data, print_eval


class TestDataLoading(unittest.TestCase):
    """ Test data loading util method """

    def test_load_data(self):
        filepath = os.path.join("../data/", "test_Consumer_Complaints.csv")
        test_data = load_data(filepath)

        shape = test_data.shape
        self.assertEqual(shape[0], 50000,
                         "Incorrect number of rows in loaded DF")
        self.assertEqual(shape[1], 4,
                         "Incorrect number of cols in loaded DF")

        types = test_data.dtypes.to_list()
        self.assertEqual(types[0], np.dtype("<M8[ns]"),
                         "First col is not a datetime64[ns]")
        for i in range(1, len(types)):
            self.assertEqual(types[i], np.dtype("O"),
                             "Incorrect dtype of col in loaded DF")

        columns = test_data.columns.to_list()
        correct_colnames = ["Date received", "Product",
                            "Consumer complaint narrative", "Complaint ID"]
        self.assertListEqual(columns, correct_colnames,
                             "Incorrect column names in loaded DF")


class TestDataCleaning(unittest.TestCase):
    """ Test data cleaning util method """

    def setUp(self) -> None:
        self.data = load_data(os.path.join("../data/",
                                           "test_Consumer_Complaints.csv"))

    def test_clean_data(self):
        test_data = clean_data(self.data)

        self.assertEqual(test_data.shape[1], 5,
                         "Incorrect number of cols in cleaned DF, maybe "
                         "target col not added")

        self.assertEqual(test_data.target.dtype, np.int64,
                         "Target col is not of type Int64")

        columns = test_data.columns.to_list()
        correct_colnames = ["input_time", "product", "text", "id", "target"]
        self.assertListEqual(columns, correct_colnames,
                             "Incorrect column names in cleaned DF")

        self.assertEqual(test_data.isna().sum().sum(), 0,
                         "NaN removal incorrect, still NaN's in cleaned DF")


class TestEvalPrinting(unittest.TestCase):
    """ Test evaluation metrices printing util method """

    @unittest.mock.patch("sys.stdout", new_callable=StringIO)
    def assert_stdout(self, pred, name, log, expected_output, mock_stdout):
        print_eval(pred, name, log)
        self.assertEqual(mock_stdout.getvalue(), expected_output)

    def test_print_eval(self):
        mock_pred = pd.DataFrame.from_dict({"input_time":
            {0: pd.Timestamp("2018-01-01 00:00:00"),
             1: pd.Timestamp('2018-01-02 00:00:00')},
                                            "product":
            {0: "Debt collection",
             1: "Mortgage"},
                                            "text":
            {0: "Product is crap, of course",
             1: "House burned, what else"},
                                            "id":
            {0: "0815",
             1: "0816"},
                                            "target":
            {0: 0,
             1: 1},
                                            "prediction":
            {0: 0.01,
             1: 0.99}})
        mock_name = "Test-Model"
        mock_log = {"nlp_logistic_classification_learner": {
            "parameters": {"test": "parameter"},
            "running_time": "23.456 s"}}
        expected_output = """Model: Test-Model
Parameters {'test': 'parameter'}
Precision 1.0
Recall 1.0
F1 Score 1.0
Training time 23.456 s 

"""
        self.assert_stdout(mock_pred, mock_name, mock_log, expected_output)


if __name__ == '__main__':
    unittest.main()
