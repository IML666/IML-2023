# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error


def transform_data(X):
    """
    This function transforms the 5 input features of matrix X (x_i denoting the i-th component of X) 
    into 21 new features phi(X) in the following manner:
    5 linear features: phi_1(X) = x_1, phi_2(X) = x_2, phi_3(X) = x_3, phi_4(X) = x_4, phi_5(X) = x_5
    5 quadratic features: phi_6(X) = x_1^2, phi_7(X) = x_2^2, phi_8(X) = x_3^2, phi_9(X) = x_4^2, phi_10(X) = x_5^2
    5 exponential features: phi_11(X) = exp(x_1), phi_12(X) = exp(x_2), phi_13(X) = exp(x_3), phi_14(X) = exp(x_4), phi_15(X) = exp(x_5)
    5 cosine features: phi_16(X) = cos(x_1), phi_17(X) = cos(x_2), phi_18(X) = cos(x_3), phi_19(X) = cos(x_4), phi_20(X) = cos(x_5)
    1 constant features: phi_21(X)=1

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features

    Returns
    ----------
    X_transformed: array of floats: dim = (700,21), transformed input with 21 features
    """
    X_transformed = np.zeros((700, 21))
    X_transformed[:, :5] = X
    X_transformed[:, 5: 10] = X ** 2
    X_transformed[:, 10: 15] = np.exp(X)
    X_transformed[:, 15:20] = np.cos(X)
    X_transformed[:, 20] = np.ones(700)
    # TODO: Enter your code here
    assert X_transformed.shape == (700, 21)
    return X_transformed


def fit(X, y, lam):
    """
    This function receives training data points, transform them, and then fits the linear regression on this 
    transformed data. Finally, it outputs the weights of the fitted linear regression. 

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features
    y: array of floats, dim = (700,), input labels)

    Returns
    ----------
    w: array of floats: dim = (21,), optimal parameters of linear regression
    """
    w = np.zeros((21,))
    # X_transformed = transform_data(X)
    X_transformed = X

    # Setup the model for linear regression
    # w[:] = np.dot(
    #             np.dot(
    #                 np.linalg.inv(
    #                 np.dot(X_transformed.T, X_transformed))
    #                 , X_transformed.T)
    #                 , y)

    # Second set with lambda to run ridge regression.
    w[:] = np.dot(
        np.dot(
            np.linalg.inv(
                np.dot(X_transformed.T, X_transformed) + lam * np.identity(21))
            , X_transformed.T)
        , y)

    assert w.shape == (21,)
    return w


def calculate_RMSE(w, X, y):
    """This function takes test data points (X and y), and computes the empirical RMSE of
    predicting y from X using a linear model with weights w.

    Parameters
    ----------
    w: array of floats: dim = (13,), optimal parameters of ridge regression
    X: matrix of floats, dim = (15,13), inputs with 13 features
    y: array of floats, dim = (15,), input labels

    Returns
    ----------
    RMSE: float: dim = 1, RMSE value
    """
    # Calculate predicted y value
    y_pred = X.dot(w)

    # Determine mean squared error
    RMSE = mean_squared_error(y, y_pred) ** 0.5

    assert np.isscalar(RMSE)
    return RMSE


def K_fold_regression(X, y, n_folds, n_repeats):
    """"
    Calculate repeated n-fold random regression, pick the best model and fit this to the data for a particular
    lambda, chosen.

    :param X: data
    :parama y: result vector
    :param n_folds int: amount of folds.
    :param n_repeats int: amount of repeats for drawing folds.
    """
    # Setup the model
    # n_folds amount of folds, n_repeats, amount of times the random draw of n_folds is repeated, random_state the seed
    # chosen to do the random drawing, having a static number yields reliable redraw. Simply generate more folds.
    rkf = RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=0)

    # Choose lambda
    lambdas = np.linspace(10, 16, 400)

    # Transform the data
    X_transformed = transform_data(X)

    # Storage array for w, RMSE
    w_storage = np.zeros((np.size(lambdas), n_repeats * n_folds, 21))
    RMSE_storage = np.zeros((np.size(lambdas), n_repeats * n_folds))

    # Repeated K-fold regression
    for i, lam in enumerate(lambdas):
        j = 0
        for train_index, test_index in rkf.split(X_transformed):
            X_train = X_transformed[train_index]
            y_train = y[train_index]

            w = fit(X_train, y_train, lam)
            w_storage[i, j, :] = w

            X_test = X_transformed[test_index]
            y_test = y[test_index]

            RMSE_storage[i, j] = calculate_RMSE(w, X_test, y_test)

            j += 1

    best_lambda_index = np.argmin(np.mean(RMSE_storage, axis=1))
    best_fold_index = np.argmin(RMSE_storage[best_lambda_index, :])
    w_hat = w_storage[best_lambda_index, best_fold_index, :]

    return w_hat, lambdas[best_lambda_index]


# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    data = pd.read_csv("train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns=["Id", "y"])
    # print a few data samples
    # print(data.head())

    X = data.to_numpy()
    # The function retrieving optimal LR parameters
    w, lamb = K_fold_regression(X, y, n_folds=8, n_repeats=5)
    # Save results in the required format
    np.savetxt("./results.csv", w, fmt="%.12f")
