# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code
# First, we import necessary libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic
from sklearn.kernel_approximation import PolynomialCountSketch
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


def data_loading():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    # Load training data
    train_df = pd.read_csv("train.csv")

    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(2))
    print('\n')

    # Load test data
    test_df = pd.read_csv("test.csv")

    print("Test data:")
    print(test_df.shape)
    print(test_df.head(2))

    # Dummy initialization of the X_train, X_test and y_train
    X_train = np.zeros_like(train_df.drop(['price_CHF'], axis=1))
    y_train = np.zeros_like(train_df['price_CHF'])
    X_test = np.zeros_like(test_df)

    # print(train_df)

    # Perform data preprocessing, imputation and extract X_train, y_train and X_test using mean values
    new_df = train_df.fillna(train_df.mean())
    y_train = new_df["price_CHF"].to_numpy()
    X_train = new_df.drop(columns=["price_CHF"]).to_numpy()

    new_test_df = test_df.fillna(test_df.mean())
    X_test = new_test_df.to_numpy()

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (
                X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test


def calculate_RMSE(y_pred, y):
    """This function takes test data points (X and y), and computes the empirical RMSE of
    predicting y from X using a linear model with weights w.

    Parameters
    ----------
    y_pred: array of floats
    y: array of floats, dim = (15,), input labels

    Returns
    ----------
    RMSE: float: dim = 1, RMSE value
    """

    # Determine mean squared error
    RMSE = mean_squared_error(y, y_pred) ** 0.5

    assert np.isscalar(RMSE)
    return RMSE


def modeling_and_prediction(X_train, y_train, X_test):
    """
    This function defines the model, fits training data and then does the prediction with the test data

    Parameters
    ----------
    X_train: matrix of floats, training input with 10 features
    y_train: array of floats, training output
    X_test: matrix of floats: dim = (100, ?), test input with 10 features

    Returns
    ----------
    y_test: array of floats: dim = (100,), predictions on test set
    """

    # Create storage array for predicted y values
    y_pred1 = np.zeros(X_test.shape[0])

    # Leave out seasons data
    X_train = X_train[:, 1:]

    # Define final test data
    X_test1 = X_test

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    '''
    gpr = GaussianProcessRegressor(kernel=DotProduct())
    gpr.fit(X_train[:,1:], y_train)
    '''

    # Storage array for RMSE values
    RMSE_mat = np.zeros(9)

    # Deine n_fold cross validation
    kf = KFold(9)
    mat_i = 0

    # Storage for all n_fold models
    models = []

    # Perform Cross validation
    for train_index, test_index in kf.split(X_train):
        X_train_folds = X_train[train_index]
        y_train_folds = y_train[train_index]

        gpr = GaussianProcessRegressor(kernel=RBF())
        gpr.fit(X_train_folds, y_train_folds)

        X_test = X_train[test_index]
        y_test = y_train[test_index]

        y_pred, sigma = gpr.predict(X_test, return_std=True)

        # print(sigma)

        models.append(gpr)

        RMSE_mat[mat_i] = calculate_RMSE(y_pred, y_test)

        mat_i = mat_i + 1

    print(RMSE_mat)

    # Determine best model
    best_index = np.argmin(RMSE_mat)

    final_gpr = models[best_index]

    y_pred = final_gpr.predict(X_test1[:, 1:])

    # print(gpr.score(X_train[:,1:], y_train))

    '''
    ps = PolynomialCountSketch(degree=3, random_state=1)
    X_features = ps.fit_transform(X)
    #print(np.exp(X_train[5,1]))
    '''
    # plt.plot(np.exp(X_train[:,1]),np.exp(y_train),'.')

    assert y_pred.shape == (100,), "Invalid data shape"
    return y_pred


# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loaditaFrame(y_png
    X_train, y_train, X_test = data_loading()
    # The function retrieving optimal LR parameters
    y_pred = modeling_and_prediction(X_train, y_train, X_test)
    # Save results in the required format

    dt = pd.DataFrame(y_pred)
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")
