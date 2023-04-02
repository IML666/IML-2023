# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code
# First, we import necessary libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic, ExpSineSquared
from sklearn.kernel_approximation import PolynomialCountSketch
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
    train_df = pd.read_csv("P2/train.csv")

    # print("Training data:")
    # print("Shape:", train_df.shape)
    # print(train_df.head(2))
    # print('\n')

    # Load test data
    test_df = pd.read_csv("P2/test.csv")

    # print("Test data:")
    # print(test_df.shape)
    # print(test_df.head(2))

    # # Dummy initialization of the X_train, X_test and y_train
    # X_train = np.zeros_like(train_df.drop(['price_CHF'], axis=1))
    # y_train = np.zeros_like(train_df['price_CHF'])
    # X_test = np.zeros_like(test_df)

    # print(train_df)

    # Perform data preprocessing, imputation and extract X_train, y_train and X_test using mean values

    # Use interpolation for missing values. Interpolate cannot handle missing starting or end values. So fill these up with mean()

    # Why does this not work with RBF?

    ### Training set

    new_train_df = train_df.interpolate(method="akima")
    new_train_df = new_train_df.fillna(train_df.mean())

    # Encode season data

    binary_version_seasons = pd.get_dummies(new_train_df['season'])
    new_train_df = new_train_df.drop(columns=['season']).join(binary_version_seasons)

    y_train = new_train_df["price_CHF"].to_numpy()
    X_train = new_train_df.drop(columns=["price_CHF"]).to_numpy()

    ### Test set

    new_test_df = test_df.interpolate(method="akima")
    new_test_df = new_test_df.fillna(new_test_df.mean())

    # Encode season data

    new_test_df = new_test_df.drop(columns=['season']).join(binary_version_seasons)

    X_test = new_test_df.to_numpy()

    # Use sklearn imputation

    # new_df = train_df.fillna(train_df.mean())
    # y_train = new_df["price_CHF"].to_numpy()
    # X_train = new_df.drop(columns=["price_CHF"]).to_numpy()

    # new_test_df = test_df.fillna(test_df.mean())
    # X_test = new_test_df.to_numpy()

    print(new_train_df.head())
    print(new_test_df.head())



    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (
                X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test


def calculate_R2(y_pred, y):

    R2 = r2_score(y, y_pred)

    assert np.isscalar(R2)
    return R2


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
    # X_train = X_train[:, 1:]

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
    R2_mat = np.zeros(9)

    # Define n_fold cross validation
    kf = KFold(9)
    mat_i = 0

    # Storage for all n_fold models
    models = []

    # Perform Cross validation
    for train_index, test_index in kf.split(X_train):
        X_train_folds = X_train[train_index]
        y_train_folds = y_train[train_index]

        gpr = GaussianProcessRegressor(kernel=DotProduct())
        gpr.fit(X_train_folds, y_train_folds)

        X_test = X_train[test_index]
        y_test = y_train[test_index]

        y_pred, sigma = gpr.predict(X_test, return_std=True)

        # print(sigma)

        models.append(gpr)

        RMSE_mat[mat_i] = calculate_RMSE(y_pred, y_test)
        R2_mat[mat_i] = calculate_R2(y_pred, y_test)

        mat_i = mat_i + 1

    print(RMSE_mat)
    print(R2_mat)

    # Determine best model
    best_index_RMSE = np.argmin(RMSE_mat)
    best_index_R2 = np.argmin(R2_mat)

    final_gpr_RMSE = models[best_index_RMSE]
    final_gpr_R2 = models[best_index_R2]

    # y_pred = final_gpr_RMSE.predict(X_test1[:, 1:])
    y_pred = final_gpr_R2.predict(X_test1[:, :])


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
    print(dt.head(5))
    dt.to_csv('P2/results.csv', index=False)
    print("\nResults file successfully generated!")
