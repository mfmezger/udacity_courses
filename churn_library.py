"""
Churn Prediction Library.

This library contians methods for performing EDA, feature engineering, and model 
training. It uses the libraries pandas, numpy, matplotlib, seaborn, sklearn, and joblib.
The results of the eda are saved to the images/eda folder. The resulting models are 
saved to the models folder. 
The resulting metrics are saved to the images/results folder. The dataset used is a 
churn prediction dataset goal is it to predict the churn of a customer with a high 
confidence.

Author: Marc

Date: February 2023
"""
import warnings

# import libraries
import logging
import os
from typing import List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split

warnings.simplefilter(action="ignore")
sns.set()


os.environ["QT_QPA_PLATFORM"] = "offscreen"

logging.basicConfig(
    filename="./logs/churn_library.log",
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)


def import_data(pth) -> pd.DataFrame:
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            dataframe: pandas dataframe
    """
    try:
        # check if pth is a valid path
        assert os.path.exists(pth) is True, "pth must be a valid path"
        dataframe = pd.read_csv(pth)

        return dataframe
    except IOError:
        logging.error("ERROR: data import failed")
        return None


def perform_eda(dataframe: pd.DataFrame, save_path: str) -> None:
    """
    perform eda on dataframe and save figures to images folder
    input:
            dataframe: pandas dataframe

    output:
            None
    """
    try:
        assert (
            isinstance(dataframe, pd.DataFrame) is True
        ), "dataframe must be a pandas dataframe"
    except AssertionError:
        logging.error("ERROR: dataframe must be a pandas dataframe")

    dataframe["Churn"] = dataframe["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )

    plt.figure(figsize=(20, 10))
    dataframe["Churn"].hist()

    # save histogramm of churn to images folder
    plt.savefig(f"./{save_path}/churn_distribution.png")

    plt.figure(figsize=(20, 10))
    dataframe["Customer_Age"].hist()

    # save picture of the age distribution to images folder
    plt.savefig(f"./{save_path}/custom_age_distribution.png")

    plt.figure(figsize=(20, 10))
    dataframe.Marital_Status.value_counts("normalize").plot(kind="bar")

    # save picture of the marital status distribution to images folder
    plt.savefig(f"./{save_path}/marital_status_distribution.png")

    plt.figure(figsize=(20, 10))
    # distplot is deprecated. Use histplot instead
    # sns.distplot(dataframe['Total_Trans_Ct']);
    # Show distributions of 'Total_Trans_Ct' and add a smooth curve obtained
    # using a kernel density estimate
    sns.histplot(dataframe["Total_Trans_Ct"], stat="density", kde=True)

    # save picture of the total transaction count distribution to images folder
    plt.savefig(f"./{save_path}/total_transition_distribution.png")

    plt.figure(figsize=(20, 10))
    sns.heatmap(dataframe.corr(), annot=False, cmap="Dark2_r", linewidths=2)

    # save picture of the correlation matrix to images folder
    plt.savefig(f"./{save_path}/heatmap.png")

    logging.info("SUCCESS: EDA has been saved.")
    plt.clf()

    return dataframe


def encoder_helper(
    dataframe: pd.DataFrame, category_lst: list, response: str
) -> pd.DataFrame:
    """
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            dataframe: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that
            could be used for naming variables or index y column]

    output:
            dataframe: pandas dataframe with new columns for
    """
    for category in category_lst:
        tmp_list = []
        calculate_mean_list = dataframe.groupby(category).mean()[response]

        for val in dataframe[category]:
            tmp_list.append(calculate_mean_list.loc[val])

        dataframe[f"{category}_{response}"] = tmp_list

    logging.info("SUCESS: Categories are encoded.")

    return dataframe


def perform_feature_engineering(dataframe: pd.DataFrame):
    """
    This method performs feature engineering and performs a train test split using
    sklean methods.
    input:
            dataframe: pandas dataframe

    output:
            x_train: X training data
            x_test: X testing data
            y_train: y training data
            y_test: y testing data
    """
    y_data = dataframe["Churn"]
    keep_cols = [
        "Customer_Age",
        "Dependent_count",
        "Months_on_book",
        "Total_Relationship_Count",
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Avg_Open_To_Buy",
        "Total_Amt_Chng_Q4_Q1",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Total_Ct_Chng_Q4_Q1",
        "Avg_Utilization_Ratio",
        "Gender_Churn",
        "Education_Level_Churn",
        "Marital_Status_Churn",
        "Income_Category_Churn",
        "Card_Category_Churn",
    ]

    x_data = pd.DataFrame()
    x_data[keep_cols] = dataframe[keep_cols]
    return train_test_split(x_data, y_data, test_size=0.3, random_state=42)


def classification_report_image(y_train, y_test, predictions, save_path) -> None:
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            predictions: tuple of predictions from logistic regression and random forest
            save_path: path to save image
    output:
            None
    """

    y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf = predictions

    plt.rc("figure", figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(
        0.01,
        1.25,
        str("Random Forest Train"),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.05,
        str(classification_report(y_test, y_test_preds_rf)),
        {"fontsize": 10},
        fontproperties="monospace",
    )  # approach improved by OP -> monospace!
    plt.text(
        0.01,
        0.6,
        str("Random Forest Test"),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.7,
        str(classification_report(y_train, y_train_preds_rf)),
        {"fontsize": 10},
        fontproperties="monospace",
    )  # approach improved by OP -> monospace!
    plt.axis("off")

    # save picture of the classification report to images folder
    plt.savefig(f"./{save_path}/classification_report.png")

    plt.rc("figure", figsize=(5, 5))
    plt.text(
        0.01,
        1.25,
        str("Logistic Regression Train"),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.05,
        str(classification_report(y_train, y_train_preds_lr)),
        {"fontsize": 10},
        fontproperties="monospace",
    )  # approach improved by OP -> monospace!
    plt.text(
        0.01,
        0.6,
        str("Logistic Regression Test"),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.7,
        str(classification_report(y_test, y_test_preds_lr)),
        {"fontsize": 10},
        fontproperties="monospace",
    )  # approach improved by OP -> monospace!
    plt.axis("off")

    # save picture of the classification report to images folder
    plt.savefig(f"./{save_path}/classification_report_lr.png")


def feature_importance_plot(cv_rfc, x_data, output_pth) -> None:
    """
    creates and stores the feature importances in pth. The plot is created
    using matplotlib.
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
            None
    """
    # Calculate feature importances
    importances = cv_rfc.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel("Importance")

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)

    # save picture of the feature importance to images/results folder
    plt.savefig(output_pth)


def train_save_models(x_train, y_train):
    """
    train, store model results: images + scores, and store models
    the random forest model is trained using grid search with cross validation.
    the logistic regression is trained using default parameters.

    The models are trainined with an increased number of jobs to speed up the
    training process. The number of jobs is set to 12, which is the number of
    cores on my machine. If you are running this on a different machine, you
    may need to change this value.
    input:
            X_train: X training data
            y_train: y training data
    output:
            None
    """
    # grid search
    rfc = RandomForestClassifier(random_state=42, verbose=0, n_jobs=12)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver="lbfgs", max_iter=3000, verbose=0, n_jobs=12)

    param_grid = {
        "n_estimators": [200, 500],
        "max_features": ["sqrt"],  # removing auto because it is sqrt
        "max_depth": [4, 5, 100],
        "criterion": ["gini", "entropy"],
    }

    cv_rfc = GridSearchCV(
        estimator=rfc, param_grid=param_grid, cv=5, verbose=0, n_jobs=12
    )
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)

    # save best model
    joblib.dump(cv_rfc.best_estimator_, "./models/rfc_model.pkl")
    joblib.dump(lrc, "./models/logistic_model.pkl")

    return lrc, cv_rfc


def model_predictions(x_train, x_test, cv_rfc, lrc) -> List:
    """
    test model on test data, for the grid search random forrest the
    best estimator is used to make predictions. For the logistic regression
    the default model is used.
    input:
            X_train: X training data
            X_test: X testing data
            cv_rfc: Random Forest Classifier
            lrc: Logistic Regression Classifier

    output:
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest
    """

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    return [y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf]


def plot_roc(classifier, x_test, y_test, save_path) -> None:
    """Plots the ROC curve for the classifier.

    input:
            classifier: model object
            x_test: test data
            y_test: test labels
            save_path: path to save the figure
    """
    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    roc_plot = plot_roc_curve(classifier, x_test, y_test)
    roc_plot.plot(ax=axis, alpha=0.8)
    plt.savefig(f"./{save_path}")
    plt.clf()


def create_dirs(paths) -> None:
    """
    creates directories if they don't exist
    input:
            paths: list of paths to create
    output:
            None
    """
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


if __name__ == "__main__":
    RESPONSE = "Churn"
    SOURCE_PATH = "./data/bank_data.csv"

    # create the necessary directories
    create_dirs(["images", "images/eda", "images/results", "models"])

    # start by reading in the data
    RAW_BANK_DATA = import_data(SOURCE_PATH)
    logging.info("SUCCESS: data imported successfully")

    # perform the eda
    EDA_DATA = perform_eda(RAW_BANK_DATA, save_path="images/eda")

    # encoding
    ENCODED_BANK_DATA = encoder_helper(
        EDA_DATA,
        category_lst=[
            "Gender",
            "Education_Level",
            "Marital_Status",
            "Income_Category",
            "Card_Category",
        ],
        response=RESPONSE,
    )

    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(
        dataframe=ENCODED_BANK_DATA
    )
    logging.info("SUCCESS: Train Test Split created.")

    # train models
    LR_MODEL, RF_MODEL = train_save_models(X_TRAIN, Y_TRAIN)
    logging.info("SUCCESS: Models trained & saved.")

    # test models
    PREDICTIONS = model_predictions(X_TRAIN, X_TEST, RF_MODEL, LR_MODEL)
    logging.info("SUCCESS: Models tested.")

    # create classification report
    classification_report_image(
        Y_TRAIN, Y_TEST, PREDICTIONS, save_path="images/results"
    )
    logging.info("SUCCESS: Classification Results stored.")

    # save roc curve
    plot_roc(RF_MODEL, X_TEST, Y_TEST, save_path="images/results/rf_results.png")

    plot_roc(LR_MODEL, X_TEST, Y_TEST, save_path="images/results/lr_results.png")
    logging.info("SUCCESS: ROC Plots saved.")
