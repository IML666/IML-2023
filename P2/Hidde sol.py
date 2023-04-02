# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic
from sklearn.model_selection import KFold

from functools import partial
from sklearn.gaussian_process import GaussianProcessRegressor
import scipy.optimize

class MyGPR(GaussianProcessRegressor):
    def __init__(self, *args, max_iter=15000, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_iter = max_iter

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        def new_optimizer(obj_func, initial_theta, bounds):
            return scipy.optimize.minimize(
                obj_func,
                initial_theta,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                max_iter=self._max_iter,
            )
        self.optimizer = new_optimizer
        return super()._constrained_optimization(obj_func, initial_theta, bounds)

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

def calculate_R2(y_pred, y):

    R2 = r2_score(y, y_pred)

    assert np.isscalar(R2)
    return R2



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
    train_df = pd.read_csv("/home/otps3141/Documents/Git Hub/IML-2023/P2/train.csv")
    
    # print("Training data:")
    # print("Shape:", train_df.shape)
    # print(train_df.head(2))
    # print('\n')
    #
    # Load test data
    test_df = pd.read_csv("/home/otps3141/Documents/Git Hub/IML-2023/P2/test.csv")

    # print("Test data:")
    # print(test_df.shape)
    # print(test_df.head(2))

    # Dummy initialization of the X_train, X_test and y_train   
    X_train = np.zeros_like(train_df.drop(['price_CHF'],axis=1))
    y_train = np.zeros_like(train_df['price_CHF'])
    X_test = np.zeros_like(test_df)

    # TODO: Perform data preprocessing, imputation and extract X_train, y_train and X_test
    # Split dataframe per season
    # Note all rows contain at least one NaN. This can be checked by the following line. Resulting in an empty dict.
    # modified_seasonal_data[season] = season_split_data[season].dropna(axis=0, how='any')

    seasonal_data_indexes = pd.DataFrame(test_df['season'])

    seasons = train_df['season'].unique().tolist()
    season_split_data = {}
    y_values_seasons = {}
    test_season_split_data = {'seasonal_data': seasonal_data_indexes}
    # Impute and split the data over the seasons.
    for season in seasons:
        temp = train_df.loc[train_df['season'] == season]
        temp_interpolid = temp.interpolate('akima').fillna(temp.mean())
        season_split_data[season] = temp_interpolid.drop(columns=['price_CHF'], axis=1)
        y_values_seasons[season] = temp_interpolid['price_CHF']

        temp_test = test_df.loc[test_df['season'] == season]
        temp_test_interpolid = temp_test.interpolate('akima').fillna(temp_test.mean())
        test_season_split_data[season] = temp_test_interpolid

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    # return X_train, y_train, X_test

    return season_split_data, y_values_seasons, test_season_split_data

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

    # y_pred=np.zeros(X_test.shape[0])
    y_pred = np.zeros(X_test['seasonal_data'].shape[0])
    #TODO: Define the model and fit it using training data. Then, use test data to make predictions

    # Define n_fold cross validation
    kf = KFold(10)

    # Define error
    error_model = 'R2'

    # Define model
    model = GaussianProcessRegressor(kernel=DotProduct())

    # Define seasons
    seasonal_data_indexes = X_test['seasonal_data']
    seasons = seasonal_data_indexes['season'].unique().tolist()


    # Define storage for y prediction
    y_pred_dict = {}

    for season in seasons:
        X_test_season = X_test[season].drop(['season'], axis=1).to_numpy()
        X_train_season = X_train[season].drop(['season'], axis=1).to_numpy()
        y_train_season = y_train[season].to_numpy()

        y_pred_dict[season] = _fit_kfold(kf_splitter=kf, X_test=X_test_season, X_train=X_train_season,
                                    y_train=y_train_season, error_method=error_model, model=model)

        indices = seasonal_data_indexes.index[seasonal_data_indexes['season'] == season].to_numpy()
        y_pred[indices] = y_pred_dict[season]


    assert y_pred.shape == (100,), "Invalid data shape"
    return y_pred

def _fit_kfold(kf_splitter, X_test, X_train, y_train, model, error_method):
    """"
    Helper function to compute the KFolds and keep the rest of the modeling function cleaner.
    The model is the machine learning model that fits the data.
    """
    # Storage for errors
    error_matrix = np.zeros(kf_splitter.get_n_splits())

    # Storage for all n_fold models
    models = []

    # Perform Cross validation
    for i, (train_index, test_index) in enumerate(kf_splitter.split(X_train)):
        X_train_folds = X_train[train_index]
        y_train_folds = y_train[train_index]

        X_test_folds = X_train[test_index]
        y_test_folds = y_train[test_index]

        # The model is refitted every time.
        model.fit(X_train_folds, y_train_folds)
        y_pred_folds, sigma = model.predict(X_test_folds, return_std=True)

        models.append(model)

        if error_method == 'RMSE':
            error_matrix[i] = calculate_RMSE(y_pred_folds, y_test_folds)

        elif error_method == 'R2':
            error_matrix[i] = calculate_R2(y_pred_folds, y_test_folds)

        else:
            raise NotImplemented

    best_index = np.argmin(error_matrix)
    best_fit = models[best_index]

    y_pred = best_fit.predict(X_test)

    return y_pred


# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = data_loading()
    # # The function retrieving optimal LR parameters
    y_pred = modeling_and_prediction(X_train, y_train, X_test)
    # # Save results in the required format
    dt = pd.DataFrame(y_pred)
    dt.columns = ['price_CHF']

    dt.to_csv('/home/otps3141/Documents/Git Hub/IML-2023/P2/results.csv', index=False)
    print("\nResults file successfully generated!")



