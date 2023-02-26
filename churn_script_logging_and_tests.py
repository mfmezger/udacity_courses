import os
import logging
from churn_library import (
    encoder_helper,
    import_data,
    perform_eda,
    perform_feature_engineering,
    train_save_models,
    model_predictions,
)
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(
    filename="./logs/churn_library_tests.log",
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)


def test_import():
    """
    test data import - this example is completed for you to assist with the other test functions
    """
    try:
        dataframe = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns"
        )
        raise err


def test_eda():
    """
    test perform eda function
    """
    try:
        dataframe = import_data("./data/bank_data.csv")
        perform_eda(dataframe, "test")
        logging.info("Testing perform_eda: SUCCESS")
        assert len(os.listdir("./test")) > 0
    except AssertionError as err:
        logging.error(
            "Testing perform_eda: The dataframe doesn't appear to have the correct columns"
        )
        raise err


def test_encoder_helper():
    """
    test encoder helper
    """
    try:
        dataframe = import_data("./data/bank_data.csv")
        dataframe = perform_eda(dataframe, "test")
        dataframe = encoder_helper(
            dataframe,
            category_lst=[
                "Gender",
                "Education_Level",
                "Marital_Status",
                "Income_Category",
                "Card_Category",
            ],
            response="Churn",
        )
        logging.info("Testing encoder_helper: SUCCESS")
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The dataframe doesn't appear to have the correct columns"
        )
        raise err


def test_perform_feature_engineering():
    """
    test perform_feature_engineering
    """
    try:
        dataframe = import_data("./data/bank_data.csv")
        dataframe = perform_eda(dataframe, "test")
        dataframe = encoder_helper(
            dataframe,
            category_lst=[
                "Gender",
                "Education_Level",
                "Marital_Status",
                "Income_Category",
                "Card_Category",
            ],
            response="Churn",
        )
        X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(
            dataframe=dataframe
        )
        assert X_TRAIN.shape[0] > 0
        assert X_TRAIN.shape[1] > 0
        assert X_TEST.shape[0] > 0
        assert X_TEST.shape[1] > 0
        assert Y_TRAIN.shape[0] > 0
        assert Y_TEST.shape[0] > 0
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The dataframe doesn't appear to have the correct columns"
        )
        raise err


def test_train_models():
    """
    test train_models
    """
    try:
        dataframe = import_data("./data/bank_data.csv")
        dataframe = perform_eda(dataframe, "test")
        dataframe = encoder_helper(
            dataframe,
            category_lst=[
                "Gender",
                "Education_Level",
                "Marital_Status",
                "Income_Category",
                "Card_Category",
            ],
            response="Churn",
        )
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            dataframe=dataframe
        )
        lr_model, rf_model = train_save_models(x_train, y_train)
        predictions = model_predictions(x_train, x_test, rf_model, lr_model)
        logging.info("SUCCESS: Models tested.")
        assert len(predictions) > 0
        assert len(os.listdir("./models")) > 0

        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The dataframe doesn't appear to have the correct columns"
        )
        raise err
