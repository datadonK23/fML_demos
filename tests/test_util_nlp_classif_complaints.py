#!/usr/bin/python
""" test_nlp_classif_complaints

Author: datadonk23
Date: 21.05.19 
"""

import unittest
import os
import numpy as np
import pandas as pd

from utils.util_nlp_classif_complaints import load_data, clean_data, print_eval


class TestDataLoading(unittest.TestCase):

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



if __name__ == '__main__':
    unittest.main()
