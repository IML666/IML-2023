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
from sklearn.impute import KNNImputer
from sklearn.impute import IterativeImputer

def data_imputation(df):
    
    """
    This function imputes a dataframe with empty values

    Parameters
    ----------
    df: Data frame with NaN entries

    Returns
    ----------
    df: Data frame imputed with missing values
    """
    
    '''
    # Create a new dataframe by filling in all NaN's with mean values
    mean_imputation_df = df.fillna(df.mean())
    
    
    print(mean_imputation_df)
    
    for i in range(len(df.columns)-4):
        
        # Save indices of all NaN's
        indices = df.index[pd.isnull(df[df.columns[i]])]
        
        # Split data (X,y) format
        X = mean_imputation_df.drop(mean_imputation_df.columns[i],axis=1)
        y = mean_imputation_df[mean_imputation_df.columns[i]]
        
        # Define Machine learning model
        gpr = GaussianProcessRegressor(kernel=DotProduct())
        gpr.fit(X, y)
        
        # Predict y_values
        y_pred = gpr.predict(X,return_std=False)
        
        # Overwrite NaN values
        df[df.columns[i]].values[indices] = y_pred[indices]
    '''
    
    imp = KNNImputer(n_neighbors=5)
    
    # imp = IterativeImputer(random_state=0)
    imp.fit(df)
    df_imputed = imp.transform(df)
    df = pd.DataFrame(df_imputed, columns=df.columns)

    return df

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



    # Load test data
    test_df = pd.read_csv("P2/test.csv")



    # Perform data preprocessing, imputation and extract X_train, y_train and X_test using mean values

    # Use interpolation for missing values. Interpolate cannot handle missing starting or end values. So fill these up with mean()

    # Why does this not work with RBF?

    ### Training set

    #new_train_df = train_df.interpolate(method="akima")
    #new_train_df = new_train_df.fillna(train_df.mean())


    # Encode season data

    binary_version_seasons = pd.get_dummies(train_df['season'])
    new_train_df = train_df.drop(columns=['season']).join(binary_version_seasons)
    
    new_train_df = data_imputation(new_train_df)
    ### Test set

    #new_test_df = test_df.interpolate(method="akima")
    #new_test_df = new_test_df.fillna(new_test_df.mean())

    # Encode season data
    
    new_test_df = test_df.drop(columns=['season']).join(binary_version_seasons)
    new_test_df = data_imputation(new_test_df)
    
    ###
    X_train = new_train_df.drop(columns = ['price_CHF']).to_numpy()
    y_train = new_train_df['price_CHF'].to_numpy()
    X_test = new_test_df.to_numpy()
    

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

    y_pred=np.zeros(X_test.shape[0])
    #TODO: Define the model and fit it using training data. Then, use test data to make predictions

    # Define n_fold cross validation
    kf = KFold(10)

    # Define error
    error_model = 'R2'

    # Define model
    model = GaussianProcessRegressor(kernel=ExpSineSquared())

    # Define seasons
    '''
    for season in seasons:
        X_test_season = X_test[season].drop(columns=seasons).to_numpy()
        X_train_season = X_train[season].drop(columns=seasons).to_numpy()
        y_train_season = y_train[season].to_numpy()

        y_pred_dict[season] = _fit_kfold(kf_splitter=kf, X_test=X_test_season, X_train=X_train_season,
                                    y_train=y_train_season, error_method=error_model, model=model)

        indices = seasonal_data_indexes.index[seasonal_data_indexes['season'] == season].to_numpy()
        y_pred[indices] = y_pred_dict[season]

    '''
    
    
    y_pred = _fit_kfold(kf_splitter=kf, X_test=X_test, X_train=X_train,
                                y_train=y_train, error_method=error_model, model=model)
    
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
            raise NotImplementedError
        
    print(error_matrix)
    best_index = np.argmax(error_matrix)
    best_fit = models[best_index]

    y_pred = best_fit.predict(X_test)

    np.set_printoptions(suppress=True)

    print(y_pred)

    return y_pred

# Data load
X_train, y_train, X_test = data_loading()
# The function retrieving optimal LR parameters
y_pred = modeling_and_prediction(X_train, y_train, X_test)
# Save results in the required format


dt = pd.DataFrame(y_pred)

dt.columns = ['price_CHF']
dt.to_csv('P2/results.csv', index=False)
print("\nResults file successfully generated!")