import pandas as pd
import unittest

from model_selector.classification.models import LogisticReg
from model_selector.regression import evaluate as eval
from model_selector.classification import evaluate as cls_eval
from model_selector.regression.models import MultipleLinear

dataset_r = pd.read_csv("data_r.csv")
dataset_c = pd.read_csv("data_c.csv")

multiple_linear = MultipleLinear(dataset_r)
logistic_regression = LogisticReg(dataset_c)


class TestEvaluation(unittest.TestCase):

    def test_perform_regression(self):
        """
            Test perform regression function
            """
        metrics = eval.perform_regression(self, multiple_linear, 0.2)
        assert len(metrics) != 0

    def test_perform_classification(self):
        """
        Test perform classification function
        """
        metrics = cls_eval.evaluate_classification(logistic_regression, 0.25)
        assert len(metrics) != 0

    def test_evaluate_regression(self):
        """
         Test evaluate regression
        """
        result = eval.evaluate_regression(dataset_r, 0.2, 4)
        assert type(result) is pd.DataFrame

    def test_evaluate_classification(self):
        """
         Test evaluate classification
        """
        result = cls_eval.evaluate_classification(dataset_c, 0.2, 4)
        assert type(result) is pd.DataFrame


if __name__ == "__main__":
    test = unittest.TestLoader().loadTestsFromTestCase(TestEvaluation)
    unittest.TextTestRunner(verbosity=2).run(test)