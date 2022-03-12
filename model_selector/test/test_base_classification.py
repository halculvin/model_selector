"""
Unit tests for base_classification.py
"""

import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from model_selector.classification import base_classification as bc

dataset = pd.read_csv("data_c.csv")


class TestBaseClassification(unittest.TestCase):
    """
    Testing the plots in the base classification module
    """
    def test_fix_missing_data(self):
        """
        Test to make sure there are no missing data
        """
        dataset_no = bc.fix_missing_data(dataset)
        # assert that there are no missing values
        self.assertTrue(pd.notnull(dataset_no).all().all())

    def test_import_dataset(self):
        dataset_no = bc.fix_missing_data(dataset)
        self.x = dataset_no.iloc[:, :-1].values
        self.y = dataset_no.iloc[:, -1].values
        self.assertTrue(len(self.x) != 0)
        self.assertTrue(len(self.y) != 0)

    def test_split_dataset(self, test_size=0.2):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=test_size,
                                                                                random_state=0)
        self.assertTrue(len(self.x_train) > len(self.x_test))

    def test_predict_test(self):
        self.y_pred = self.regressor.predict(self.x_test)
        assert self.y_pred.shape == (self.x_train.shape[0],)

    def test_evaluate_metrics(self):
        r2_score = bc.accuracy_score()
        assert r2_score > 0
