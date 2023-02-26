# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This is the clean code project of the Udacity Machine Learning Engineer Nanodegree. The goal of this project is to build a model that predicts customer churn on a fictional dataset. 
The dataset used is from kaggle and available at https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers.
Firstly a EDA is performed on the dataset to get an overview of the data. The data is then cleaned and prepared for the model. The models that are trained are a Random Forest Classifier using 5 Fold Cross Validation and a Logistic Regression Classifier, the library used for the models is scikit-learn. (https://scikit-learn.org/stable/). The models are then stored in the models folder als pickle file. The complete project includes a python package for the machine learning projekt that follows the python coding standard PEP8 and the engineering best practices for implementing software (modular, documented and tested). The package includes a Jupyter Notebook for a more graphical approach to the project as well as a script that can be run from the command line interface (CLI). The package also includes a logging file and modular tests for every function.

## Files and data description
Overview of the files and data present in the root directory. 

```

├── README.md
├── churn_library.py - The basic functions for the project
├── churn_notebook.ipynb - The notebook for the project
├── churn_script_logging_and_tests.py - The testing script
├── data
│   └── bank_data.csv - The data for the project
├── images
│   ├── eda - The images for the EDA
│   │   ├── churn_distribution.png
│   │   ├── custom_age_distribution.png
│   │   ├── heatmap.png
│   │   ├── marital_status_distribution.png
│   │   └── total_transition_distribution.png
│   └── results - The images for the results
│       ├── classification_report.png
│       ├── classification_report_lr.png
│       ├── lr_results.png
│       └── rf_results.png
├── logs - The logs for the project
│   └── churn_library.log
├── models - The saved models for the project
│   ├── logistic_model.pkl
│   └── rfc_model.pkl
├── requirements_py3.8.txt - The requirements for the project
└── test - The test results for the project
    ├── churn_distribution.png
    ├── custom_age_distribution.png
    ├── heatmap.png
    ├── marital_status_distribution.png
    └── total_transition_distribution.png

```

## Running Files
How do you run your files? What should happen when you run your files?

First it is necessary to install the requirements to your repository.
    
    ```bash
    pip install -r requirements_py3.8.txt
    ```

To start the process start the churn script with the following command:

    ```bash
    python churn_library.py
    ```

To run the tests for the project run the following command:

    ```bash
    pytest churn_script_logging_and_tests.py
    ```
