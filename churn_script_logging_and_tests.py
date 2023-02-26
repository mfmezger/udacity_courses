import logging

from churn_library import (
    encoder_helper,
    import_data,
    perform_eda,
    perform_feature_engineering,
    train_save_models,
)

logging.basicConfig(
    filename='./logs/churn_library_tests.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
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
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    '''
    test perform eda function
    '''
    try:
        dataframe = import_data("./data/bank_data.csv")
        perform_eda(dataframe, "test")
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_eda: The dataframe doesn't appear to have the correct columns")
        raise err


def test_encoder_helper():
    '''
    test encoder helper
    '''
    try:
        dataframe = import_data("./data/bank_data.csv")
        dataframe = perform_eda(dataframe, "test")
        dataframe = encoder_helper(dataframe, category_lst=[
            "Gender",
            "Education_Level",
            "Marital_Status",
            "Income_Category",
            "Card_Category"],
            response="Churn")
        logging.info("Testing encoder_helper: SUCCESS")
        return dataframe
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The dataframe doesn't appear to have the correct columns")
        raise err


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    try:
        dataframe = import_data("./data/bank_data.csv")
        dataframe = perform_eda(dataframe, "test")
        dataframe = encoder_helper(dataframe, category_lst=[
            "Gender",
            "Education_Level",
            "Marital_Status",
            "Income_Category",
            "Card_Category"],
            response="Churn")
        X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(
            dataframe=dataframe)
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The dataframe doesn't appear to have the correct columns")
        raise err


def test_train_models():
    '''
    test train_models
    '''
    try:
        dataframe = import_data("./data/bank_data.csv")
        dataframe = perform_eda(dataframe, "test")
        dataframe = encoder_helper(dataframe, category_lst=[
            "Gender",
            "Education_Level",
            "Marital_Status",
            "Income_Category",
            "Card_Category"],
            response="Churn")
        X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(
            dataframe=dataframe)
        LR_MODEL, RF_MODEL = train_save_models(X_TRAIN, Y_TRAIN)
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The dataframe doesn't appear to have the correct columns")
        raise err
