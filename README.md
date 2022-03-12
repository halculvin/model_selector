
# Model-Selection
There are a lot of predictive modeling algorithms in machine learning (ML) in engineering applications. Predictive modeling is the problem of developing a model using historical data to predict new data where we do not have the answer. Predictive modeling can be described as the mathematical problem of approximating a mapping function (f) from input variables (X) to output variables (y). This is called the problem of function approximation. Generally, we can divide all function approximation tasks into classification and regression tasks. 
Regression models include Single and Multiple Linear Regression, Decision Tree, Polynomial, Random Forest, and Support Vector. Classification models include Decision Tree, K-Nearest Neighbors, Kernel SVM, Logistic Regression, Naïve Bayes, Random Forest, and Support Vector Machine.
One of the most frequently asked questions in the data science community seeks to determine which regression or classification model is best suited to be used on different datasets. This project aims to build a python package that will help you evaluate your regression models to select the best model for your dataset quickly and efficiently. Provide the file path to your dataset with some optional parameters and watch this package do the rest of the work for you.

## Project Organization
```
model_selector/
  |- README.md
  |- model_selector/
    |- __init__.py
    |- classification/
      |- __init__.py
      |- base_classification.py
      |- evaluate.py
      |- models.py
    |- regression/
      |- __init__.py
      |- base_regression.py
      |- evaluate.py
      |- models.py
    |- tests/
      |- __init__.py
      |- data_c.csv
      |- data_r.csv
      |- test_base_classification.py
      |- test_base_regression.py
      |- test_evaluate.py
  |- data/
    |- Data_classification.csv
    |- Sales_Used_Cars.csv
  |- docs/
    |- DATA515 Project Presentation.pdf
    |- Functional Requirements.pdf
    |- Software Components.pdf
  |-example/
    |- example.ipynb
    |- README.md
  |- setup.py
  |- requirements.txt
  |- LICENSE.txt
```
---
## Installation
Clone the repo and create a virtual environment in the root of the repo
```bash
python -m venv venv
source venv/bin/activate
```
If you're using Anaconda, create and activate a new conda environment.
For conda run
```bash
conda create --name model_selector
conda activate model_selector
```

Install the dependencies from the `requirements.txt` file using
```bash
python -m pip install -r requirements.txt
```

If you don't have `setuptools` and `wheel` install them using
```bash
python -m pip install --upgrade setuptools wheel
```

Install the package using the following command
```bash
python setup.py sdist bdist_wheel
```

This will generate the pip installation package `model_selector-1.0.2-py3-none-any.whl` in the `dist/` directory.
The package `model-selector` can now be installed using

```bash
pip install model_selector-1.0.2-py3-none-any.whl
```

## Usage

To see how to use the package to get instance recommendation, 
refer to the [example notebook](example/example.ipynb)