# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import pandas as pd
import numpy as np
import torch
import torch.nn as nn


import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from tqdm import tqdm



def load_data(path_dict):
    """
    This function loads the data from the csv files and returns it as numpy arrays.

    input: None
    
    output: x_pretrain: np.ndarray, the features of the pretraining set
            y_pretrain: np.ndarray, the labels of the pretraining set
            x_train: np.ndarray, the features of the training set
            y_train: np.ndarray, the labels of the training set
            x_test: np.ndarray, the features of the test set
    """
    x_pretrain = pd.read_csv(path_dict["pretrain_features"], index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y_pretrain = pd.read_csv(path_dict["pretrain_labels"], index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_train = pd.read_csv(path_dict["train_features"], index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y_train = pd.read_csv(path_dict["train_labels"], index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_test = pd.read_csv(path_dict["test"], index_col="Id", compression='zip').drop("smiles", axis=1)
    return x_pretrain, y_pretrain, x_train, y_train, x_test



class NN(nn.Module):
    """
    The model class, which defines our feature extractor used in pretraining.
    """
    def __init__(self, in_features, activation):
        """
        The constructor of the model.
        """

        super().__init__()

        first_layer_nodes = in_features*0.5
        second_layer_nodes = in_features*0.25
        third_layer_nodes = in_features*0.125
        fourth_layer_nodes = in_features*0.0625
        
        
        self.fc1 = nn.Linear(in_features, first_layer_nodes)
        self.fc2 = nn.Linear(first_layer_nodes, second_layer_nodes)
        self.fc3 = nn.Linear(second_layer_nodes, third_layer_nodes)
        self.fc4 = nn.Linear(third_layer_nodes, fourth_layer_nodes)
        self.dropout = nn.Dropout(0.6)

        self.output = nn.Linear(fourth_layer_nodes, 1)
        self.bn1 = nn.BatchNorm1d(first_layer_nodes)
        self.bn2 = nn.BatchNorm1d(second_layer_nodes)
        self.bn3 = nn.BatchNorm1d(third_layer_nodes)
        self.bn4 = nn.BatchNorm1d(fourth_layer_nodes)


        

        
        self.function = function

    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.function(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.function(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.function(x)
        x = self.dropout(x)

        x = self.fc4(x)
        x = self.bn4(x)
        x = self.function(x)
        x = self.dropout(x)

        x = self.output(x)
        x = F.sigmoid(x)
        x = x.squeeze()
        return x
    
class Autoencoder(nn.Module):
    """
    The model class, which defines our feature extractor used in pretraining.
    """
    def __init__(self, in_features, activation):
        """
        The constructor of the model.
        """
        super().__init__()
        # TODO: Define the architecture of the model. It should be able to be trained on pretraing data 
        # and then used to extract features from the training and test data.

        # Autoencoder setup
        # first_layer_nodes = in_features*0.5
        # second_layer_nodes = in_features*0.25
        # third_layer_nodes = in_features*0.125
        # fourth_layer_nodes = in_features*0.0625


        # Define number of nodes in extraction layer
        feature_layer = 64

        nodes = [in_features, 800, 400, 200, 128]

        self.activation = activation
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(nodes[0], nodes[1])
        self.fc2 = nn.Linear(nodes[1], nodes[2])
        self.fc3 = nn.Linear(nodes[2], nodes[3])
        self.fc4 = nn.Linear(nodes[3], nodes[4])

        self.output = nn.Linear(nodes[4], feature_layer)
        


        self.encoder = torch.nn.Sequential(
            
            self.fc1,
            # torch.nn.BatchNorm1d(500),
            self.activation(),
            # self.dropout,

            self.fc2,
            # torch.nn.BatchNorm1d(250),
            self.activation(),
            # self.dropout,

            self.fc3,
            # torch.nn.BatchNorm1d(64),
            self.activation(),
            # self.dropout,

            self.fc4,
            # torch.nn.BatchNorm1d(32),
            self.activation(),
            # self.dropout,

            self.output,
            self.activation(),
            # self.dropout
        )

        self.decoder = torch.nn.Sequential(

            torch.nn.Linear(feature_layer, nodes[4]),
            # torch.nn.BatchNorm1d(32),
            self.activation(),
            # self.dropout,

            torch.nn.Linear(nodes[4], nodes[3]),
            # torch.nn.BatchNorm1d(64),
            self.activation(),
            # self.dropout,

            torch.nn.Linear(nodes[3], nodes[2]),
            # torch.nn.BatchNorm1d(125),
            self.activation(),
            # self.dropout,

            torch.nn.Linear(nodes[2], nodes[1]),
            # torch.nn.BatchNorm1d(250),
            self.activation(),
            # self.dropout,

            torch.nn.Linear(nodes[1], nodes[0]), 
            self.activation()
        )

        # Write prediction function
        self.prediction = torch.nn.Sequential(
            torch.nn.Sigmoid(),
            torch.nn.Linear(feature_layer, 1)
        )




    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        # TODO: Implement the forward pass of the model, in accordance with the architecture 
        # defined in the constructor.

        # Autoencoder forward pass
        encoded_array = self.encoder(x)
        decoded_array = self.decoder(encoded_array)
        return decoded_array
    
def make_feature_extractor(x, y, n_epochs, AE, optimizer, loss_function, activation, model, scheduler, batch_size=256, eval_size=1000):
    """
    This function trains the feature extractor on the pretraining data and returns a function which
    can be used to extract features from the training and test data.

    input: x: np.ndarray, the features of the pretraining set
              y: np.ndarray, the labels of the pretraining set
                batch_size: int, the batch size used for training
                eval_size: int, the size of the validation set
                n_epochs: int, the number of epochs to train the model for
            
    output: make_features: function, a function which can be used to extract features from the training and test data
    """
    # Pretraining data loading
    in_features = x.shape[-1]
    x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=eval_size, random_state=0, shuffle=True)
    x_tr, x_val = torch.tensor(x_tr, dtype=torch.float), torch.tensor(x_val, dtype=torch.float)
    y_tr, y_val = torch.tensor(y_tr, dtype=torch.float), torch.tensor(y_val, dtype=torch.float)

    
    # TODO: Implement the training loop. The model should be trained on the pretraining data. Use validation set 
    # to monitor the loss.

    # Training loop
    epochs = n_epochs


    train_val_losses = []

    model.train()

    for epoch in tqdm(range(n_epochs)):
        # shuffle data
        idx = np.arange(x_tr.shape[0])
        np.random.shuffle(idx)
        x_tr = x_tr[idx]
        y_tr = y_tr[idx]

        # training
        for i in range(0, x_tr.shape[0], batch_size):
            x_batch = x_tr[i:i+batch_size]
            y_batch = y_tr[i:i+batch_size]

            optimizer.zero_grad()
            x_pred = model.forward(x_batch)
            loss = loss_function(x_pred, x_batch)
            loss.backward()
            train_loss = loss.item()
            optimizer.step()

        scheduler.step()

        # validation for autoencoder
        with torch.no_grad():
            x_pred = model.forward(x_val)
            loss = loss_function(x_pred, x_val)
            val_loss = loss.item()

        print(
            f"Epoch: {epoch}, Train loss: {train_loss}, Val loss: {val_loss}")

    # for epoch in tqdm(range(epochs)):

        

    #     # Training the model
    #     train_loss = 0
    #     for i in range(0, len(x_tr), batch_size):

    #         X = x_tr[i:i+batch_size].to(device)
    #         y = y_tr[i:i+batch_size].to(device)

    #         optimizer.zero_grad()


    #         reconstructed = model.forward(X)
            
            
    #         loss = loss_function(reconstructed, X)
    #         loss.backward()
    #         train_loss += loss.detach()
    #         optimizer.step()
        
    #     scheduler.step()
              
    #     train_loss = train_loss/len(x_tr)

    #     # Validating the model
    #     model.eval()
    #     with torch.no_grad():
    #         val_loss = 0
    #         for X_valid in x_val:
    #             X_valid = X_valid.to(device)
    #             reconstructed_val = model.forward(X_valid)
                
    #             loss = loss_function(reconstructed_val, X_valid)
    #             val_loss += loss.detach()

    #         val_loss = val_loss/len(x_val)
        
    #     print(f'Train loss in {epoch}: {train_loss}, validation loss: {val_loss}')
    #     train_val_losses.append((train_loss, val_loss))

        

    def make_features(x):
        """
        This function extracts features from the training and test data, used in the actual pipeline 
        after the pretraining.

        input: x: np.ndarray, the features of the training or test set

        output: features: np.ndarray, the features extracted from the training or test set, propagated
        further in the pipeline
        """
        model.eval()
        # TODO: Implement the feature extraction, a part of a pretrained model used later in the pipeline.
        # Extract features from autoencoder
        # Note that we need to convert the data to torch.Tensor and back to numpy array
        x = torch.tensor(x, dtype=torch.float)
        features = model.encoder(x)
        features = features.detach().numpy()

        return x

    return make_features, train_val_losses

def make_pretraining_class(feature_extractors):
    """
    The wrapper function which makes pretraining API compatible with sklearn pipeline
    
    input: feature_extractors: dict, a dictionary of feature extractors

    output: PretrainedFeatures: class, a class which implements sklearn API
    """

    class PretrainedFeatures(BaseEstimator, TransformerMixin):
        """
        The wrapper class for Pretraining pipeline.
        """
        def __init__(self, *, feature_extractor=None, mode=None):
            self.feature_extractor = feature_extractor
            self.mode = mode

        def fit(self, X=None, y=None):
            return self

        def transform(self, X):
            assert self.feature_extractor is not None
            X_new = feature_extractors[self.feature_extractor](X)
            return X_new
        
    return PretrainedFeatures

def get_regression_model():
    """
    This function returns the regression model used in the pipeline.

    input: None

    output: model: sklearn compatible model, the regression model
    """
    # TODO: Implement the regression model. It should be able to be trained on the features extracted
    # by the feature extractor.
    model = LinearRegression()
    return model

# Main function. You don't have to change this
if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Used device: {device}')

    # Define file paths
    pretrain_features = "pretrain_features.csv.zip"
    pretrain_labels = "pretrain_labels.csv.zip"
    train_features = "train_features.csv.zip"
    train_labels = "train_labels.csv.zip"
    test = "test_features.csv.zip"

    # Create dictionary of file paths
    Path = os.path
    dir = Path.join(Path.dirname(__file__))
    path_pretrain_features = Path.join(dir, pretrain_features)
    path_train_features = Path.join(dir, train_features)
    path_pretrain_labels = Path.join(dir, pretrain_labels)
    path_train_labels = Path.join(dir, train_labels)
    path_test = Path.join(dir, test)

    path_dict = {"pretrain_features": path_pretrain_features, "pretrain_labels": path_pretrain_labels, 
                 "train_features": path_train_features, "train_labels": path_train_labels, "test": path_test}


    # Load data
    x_pretrain, y_pretrain, x_train, y_train, x_test = load_data(path_dict)
    print("Data loaded!")

    print("pretrain_features shape: ", x_pretrain.shape)

    print("train_features shape: ", x_train.shape)
    # Utilize pretraining data by creating feature extractor which extracts lumo energy 
    # features from available initial features

    # Set parameters for the feature extractor
    n_epochs = 20

    # Use autoencoder or NN to extract features from the pretraining data
    AE = True

    # Set functions for the feature extractor
    loss_function = torch.nn.MSELoss()
    activation = torch.nn.ReLU
    in_features = x_pretrain.shape[1]

    # model declaration
    if AE:
        model = Autoencoder(in_features, activation)
    else:
        model = NN(in_features, activation)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    

    feature_extractor, train_val_losses =  make_feature_extractor(x_pretrain, y_pretrain, n_epochs, AE, optimizer, loss_function, activation, model, scheduler)
    PretrainedFeatureClass = make_pretraining_class({"pretrain": feature_extractor})
    PretrainedFeature = PretrainedFeatureClass(feature_extractor="pretrain") 
    
    # regression model
    regression_model = get_regression_model()

    # Define scaler
    # The scaler acts like a regularization
    scaler = StandardScaler()

    
    # TODO: Implement the pipeline. It should contain feature extraction and regression. You can optionally
    # use other sklearn tools, such as StandardScaler, FunctionTransformer, etc.

    # Create the pipeline
    pipeline = Pipeline([
        ("feature_extractor", PretrainedFeature),
        ("scaler", scaler),
        ("regression", regression_model),
    ])

    # Train the pipeline
    pipeline.fit(x_train, y_train)

    # Predict on the test set
    # First convert to torch.Tensor
    x_test_to_tensor = torch.tensor(x_test.values, dtype=torch.float)
    y_pred = pipeline.predict(x_test_to_tensor)

    assert y_pred.shape == (x_test.shape[0],)
    y_pred = pd.DataFrame({"y": y_pred}, index=x_test.index)

    path_results = Path.join(dir, "results.csv")
    y_pred.to_csv(path_results, index_label="Id")
    print("Predictions saved, all done!")