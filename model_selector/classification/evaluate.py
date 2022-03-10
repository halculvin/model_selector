import pandas as pd
from model_evaluator.classification.models import LogisticReg, DecisionTree, KNearestNeighbors, KernelSVM, NaiveBayes, \
    RandomForest, SupportVectorMachine
import errno
import os


def perform_classification(classification, test_size):
    classification.import_dataset().split_dataset(test_size=test_size).feature_scale().train_model()
    metrics = classification.confusion_matrix(classification.__str__())
    return metrics


def evaluate_classification(filename, test_size=0.2):
    file_exists = os.path.exists(filename)
    if file_exists:
        logistic_regression = LogisticReg(filename)
        decision_tree = DecisionTree(filename)
        k_nearest_neighbors = KNearestNeighbors(filename)
        kernel_svm = KernelSVM(filename)
        naive_bayes = NaiveBayes(filename)
        random_forest = RandomForest(filename)
        svm = SupportVectorMachine(filename)

        result = [perform_classification(logistic_regression, test_size=test_size),
                  perform_classification(decision_tree, test_size=test_size),
                  perform_classification(k_nearest_neighbors, test_size=test_size),
                  perform_classification(kernel_svm, test_size=test_size),
                  perform_classification(naive_bayes, test_size=test_size),
                  perform_classification(random_forest, test_size=test_size),
                  perform_classification(svm, test_size=test_size),
                  ]

        df = pd.DataFrame(result)
        return df
    else:
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), filename)
