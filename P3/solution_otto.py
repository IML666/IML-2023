# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the
# First, we import necessary libraries:
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import os
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights, efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b7, efficientnet_b5
import torchvision.datasets as datasets
from torchvision.models import EfficientNet_B7_Weights

import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchmetrics.classification import BinaryAccuracy


# from copy import deepcopy, Tensors do not support deepcopy


def generate_embeddings(path, batch_size, num_work):
    """
    Transform, resize and normalize the images and then use a pretrained model to extract
    the embeddings.
    """

    # Define transformations, standard as copied from the web, first resize to 224 x 224 image,
    # then transform to tensor and normalize with magic numbers from the web.

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    pretrained_weights = EfficientNet_B7_Weights.IMAGENET1K_V1
    # train_transforms = pretrained_weights.transforms()

    # The train dataset, loaded in batches.
    train_dataset = datasets.ImageFolder(root="P3/dataset/", transform=train_transforms)

    # We load all the data at once. Hence, a massive batch size.
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              pin_memory=True, num_workers=num_work)

    # Pretrained model resnet50, use the newest version with weights IMAGENET1K_V2, 80.858% accuracy.
    # model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    # Pretrained model efficientnet_b0
    # model = efficientnet_b0(weights='IMAGENET1K_V1')

    # Pretrained model efficientnet_b3
    # model = efficientnet_b3(weights='IMAGENET1K_V1')

    # Pretrained model efficientnet_b5
    # model = efficientnet_b5(weights='IMAGENET1K_V1')

    # Pretrained model efficientnet_b7
    model = efficientnet_b7(weights='IMAGENET1K_V1')  

    # Put model into evaluation mode to get the calculated values for each entry.
    model.eval()

    # Sent model to the gpu
    model.to(device)

    # Extract the input of the last layer. These are the features.
    embedding_size = list(model.children())[-1][1].in_features
    num_images = len(train_dataset)
    embeddings = np.zeros((num_images, embedding_size))

    # Solution from url:
    # https://stackoverflow.com/questions/52548174/how-to-remove-the-last-fc-layer-from-a-resnet-model-in-pytorch
    new_model = torch.nn.Sequential(*(list(model.children())[:-1]))

    # Not necessary since the model has already been moved and data gets moved later.
    # # Put model into eval
    # new_model.eval()
    #
    # # Sent model to device
    # new_model.to(device)

    with torch.no_grad():
        # Temporarily turn off the gradient computation.
        print("Evaluating the model for the training data")
        for batch_index, (data, target) in enumerate(tqdm(train_loader)):
            # Create embedding from retrieving the features from the last layer.
            data = data.to(device)
            output = new_model(data)

            # Squeeze feature set down from 4D with 2 redundant dimensions to 2D.
            # .cpu() moves the object from gpu to cpu memory, .numpy() creates a numpy object.
            extracted_features = output.squeeze().cpu().numpy()

            # Since we have chosen batch_size = 64 fill embeddings in this way
            start = batch_index * train_loader.batch_size
            finish = (batch_index + 1) * train_loader.batch_size
            embeddings[start:finish] = extracted_features

    np.save(path, embeddings)


def get_data(file, path, train=True):
    """
    Load the triplets from the file and generate the features and labels.

    input: file: string, the path to the file containing the triplets
          train: boolean, whether the data is for training or testing

    output: X: numpy array, the features
            y: numpy array, the labels
    """
    # Load the data from the triplets.
    triplets = []
    with open(file) as f:
        for line in f:
            triplets.append(line)

    # Generate training data from triplets
    train_dataset = datasets.ImageFolder(root="P3/dataset", transform=None)
    filenames = [s[0].split('/')[-1].replace('.jpg', '').split("\\")[-1] for s in train_dataset.samples]
    embeddings = np.load(path)

    # Use standard normal normalisation
    normalized_embeddings = (embeddings - np.mean(embeddings, axis=0)) / np.std(embeddings, axis=0)
    # use between 0 and 1 normalization.
    # normalized_embeddings = (embeddings - np.min(embeddings, axis=0)) / np.max(embeddings, axis=0)

    # Create a dictionary that stores the embedding for each image. For this step it is important
    # that the embeddings are generated in order, such that the embedding index matches the filename.
    file_to_embedding = {}
    for i in range(len(filenames)):
        file_to_embedding[filenames[i]] = normalized_embeddings[i]

    # Setup empty data lists.
    X = []
    y = []

    # Use the individual embeddings to generate the features and labels for triplets
    # Pick out a triplet t.
    if train:
        dataset_name = 'training'
    else:
        dataset_name = 'testing'

    print(f'Loading the triplets for {dataset_name} dataset')
    for t in tqdm(triplets):
        # Access the dictionary to create a 3 part list that contains the image embedding for all 3 images.
        emb = [file_to_embedding[a] for a in t.split()]

        # Append the data to the list with arrays.
        X.append(np.hstack([emb[0], emb[1], emb[2]]))
        y.append(1)

        # Generating negative samples (data augmentation)
        if train:
            X.append(np.hstack([emb[0], emb[2], emb[1]]))
            y.append(0)

    X = np.vstack(X)
    y = np.hstack(y)

    return X, y


# Hint: adjust batch_size and num_workers to your PC configuration, so that you don't run out of memory
def create_loader_from_np(X, y=None, train=True, batch_size=64, shuffle=True, num_workers=4):
    """
    Create a torch.utils.data.DataLoader object from numpy arrays containing the data.

    input: X: numpy array, the features
           y: numpy array, the labels

    output: loader: torch.data.util.DataLoader, the object containing the data
    """
    if train:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float),
                                torch.from_numpy(y).type(torch.float))
    else:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float))

    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        pin_memory=True, num_workers=num_workers)
    return loader


class Net(nn.Module):
    """
    The model class, which defines our classifier.
    """

    def __init__(self, function):
        """
        The constructor of the model.
        """
        super().__init__()
        # Fully connected model with 3 x the amount of features per picture and binary output.
        # For ResNet18 adjust to 512x3 features per picture.
        # For ResNet50 adjust to 2048x3 features per picture.
        # For EfficientNetB0 adjust to 1280x3 features per picture.
        # For EfficientNetB3 adjust to 1536x3 features per picture.
        # For EfficientNetB7 adjust to 2560x3 features per picture.
        # For EfficientNetB5 2048x3
        
        self.fc1 = nn.Linear(6144, 3072)
        self.fc2 = nn.Linear(3072, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(0.6)

        self.output = nn.Linear(512, 1)
        self.bn1 = nn.BatchNorm1d(3072)
        self.bn2 = nn.BatchNorm1d(2048)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)


        

        
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


def update_line(hl, new_data):
    hl.set_xdata(np.append(hl.get_xdata(), new_data[0]))
    hl.set_ydata(np.append(hl.get_ydata(), new_data[1]))
    plt.draw()


def train_model(model, train_loader, loss_function, optimizer, epochs, device, batch_size, split):
    """
    The training procedure of the model; it accepts the training data, defines the model
    and then trains it.

    input: train_loader: torch.data.util.DataLoader, the object containing the training data

    output: model: torch.nn.Module, the trained model
    """

    validation_loss = []
    losses = []
    metric = BinaryAccuracy().to(device)

    # Setup the lines to live plot the data. TODO: Fix this.
    # hl, = plt.plot([0], [0])

    # Introduce split between training and validation data with split
    dataset = train_loader.dataset

    val_split = split
    train_split = 1 - split
    len_train = int(len(dataset) * train_split)
    len_val = len(dataset) - len_train
    train_set, val_set = torch.utils.data.random_split(dataset, [len_train, len_val])

    # Initialize the minimum validation loss to infinity.
    minimum_val_loss = np.inf

    
    # Apply the training procedure for the specified number of epochs and differentiate between validation and training.
    train_val_loss = []

    # Put the training and validation data in a loader.
    
    train_split_loader = torch.utils.data.DataLoader(
        train_set, batch_size=train_loader.batch_size, shuffle=True)
    val_split_loader = torch.utils.data.DataLoader(
        val_set, batch_size=train_loader.batch_size, shuffle=False)


    for epoch in tqdm(range(epochs)):

        # Training the model
        train_loss = 0
        model.train()
        print('\n')
        print(f'Training the model in epoch {epoch}')
        print('\n')
        for [X, y] in tqdm(train_split_loader):
            X = X.to(device)
            y = y.to(device)
            y_pred = model.forward(X)
            
            
            optimizer.zero_grad()
            loss = loss_function(y_pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.detach()
            
            
        train_loss = train_loss/len(train_split_loader)

        # Validating the model
        model.eval()
        print('\n')
        print(f'Validating the model in epoch {epoch}')
        print('\n')
        with torch.no_grad():
            val_loss = 0
            accuracy = 0
            for [X_valid, y_valid] in tqdm(val_split_loader):
                X_valid = X_valid.to(device)
                y_valid = y_valid.to(device)
                y_valid_pred = model.forward(X_valid)
                
                loss = loss_function(y_valid_pred, y_valid)
                val_loss += loss
                accuracy += metric(y_valid_pred, y_valid)
            val_loss = val_loss/len(val_split_loader)
            accuracy = accuracy/len(val_split_loader)
        
        print('\n')
        print(f'Train {epoch} loss: {train_loss}, validation loss: {val_loss}, accuracy: {accuracy}')
        print('\n')
        train_val_loss.append((train_loss, val_loss))


        ### We can use the validation data again to train. For this we need to define the best epoch model
        if val_loss < minimum_val_loss:
            minimum_val_loss = val_loss
            min_val_state = model.state_dict()

    # The model is trained on the train split. Take the best model and train it on the whole dataset.
    print("Training the model with minimal validation loss on the whole dataset")
    model.load_state_dict(min_val_state)

    # Define number of epochs for the final training
    epochs_final = 5

    # Define training procedure for the final training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    final_train_loss = []

    for epoch in tqdm(range(epochs_final)):
        train_loss = 0
        model.train()
        print('\n')
        print(f'Final training the model in epoch {epoch}')
        print('\n')
        for [X, y] in tqdm(train_loader):
            X = X.to(device)
            y = y.to(device)
            y_pred = model.forward(X)
            
            
            optimizer.zero_grad()
            loss = loss_function(y_pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.detach()
            
            
        train_loss = train_loss/len(train_loader)
        final_train_loss.append(train_loss)
        print('\n')
        print(f'Final train loss in epoch {epoch}: {train_loss}')
        print('\n')


    return model, train_val_loss


def plot_loss(train_val_loss, epochs):
    """
    Plots the training and validation loss over the epochs.

    input: train_val_loss: list, the training and validation loss over the epochs
           epochs: int, the number of epochs
    """
    epoch_range = np.arange(epochs)
    train_val_loss = np.array(train_val_loss)
    plt.plot(epoch_range, train_val_loss[:, 0], label='Training loss')
    plt.plot(epoch_range, train_val_loss[:, 1], label='Validation loss')
    plt.show()


def test_model(model, loader):
    """
    The testing procedure of the model; it accepts the testing data and the trained model and
    then tests the model on it.

    input: model: torch.nn.Module, the trained model
           loader: torch.data.util.DataLoader, the object containing the testing data

    output: None, the function saves the predictions to a results.txt file
    """
    model.eval()
    predictions = []
    # Iterate over the test data
    with torch.no_grad():  # We don't need to compute gradients for testing
        for [x_batch] in loader:
            x_batch = x_batch.to(device)
            predicted = model(x_batch)
            predicted = predicted.cpu().numpy()
            # Rounding the predictions to 0 or 1
            predicted[predicted >= 0.5] = 1
            predicted[predicted < 0.5] = 0
            predictions.append(predicted)
        # Modification from the original
        # predictions = np.vstack(predictions)
        predictions = np.hstack(predictions)
        predictions = np.vstack(predictions)
    np.savetxt("P3/results.txt", predictions, fmt='%i')


# Write initialization for weights
def init_weights(m):
    """
    Initializes the weights of the model.

    input: m: torch.nn.Module, the model

    output: None, the function initializes the weights of the model
    """
    if type(m) == nn.Linear:
        # torch.nn.init.uniform_(m.weight, -0.01, 0.01)
        # torch.nn.init.xavier_uniform(m.weight)
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)

# Main function. You don't have to change this
if __name__ == '__main__':

    # the device objectss indexes all available GPU's, since there is only one NVidea GPU in my pc it is cuda:0,
    # or cuda() .
    # use list(range(torch.cuda.device_count())) to find all avaible GPU's.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Used device: {device}')
    # device = "cpu"

    # Datasets
    TRAIN_TRIPLETS = 'P3/train_triplets.txt'
    TEST_TRIPLETS = 'P3/test_triplets.txt'

    # Path to dataset
    embedding_name = "otto_embeddings_efficientnetb5.npy"
    Path = os.path
    dir = Path.join(Path.dirname(__file__))
    dir_path = Path.join(Path.dirname(__file__), "dataset")
    path = Path.join(dir_path, embedding_name)

    # Create model
    model = Net(F.relu)
    # Apply weight initialization
    # model.apply(init_weights)
    model.to(device)

    # Setup parameters
    batch_size = 100
    num_work = 8
    epochs = 20
    split = 0.1
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)

    loss_function = nn.BCELoss()  # nn.CrossEntropyLoss() only for 2 or more classes
    test = True

    # generate embedding for each image in the dataset
    if (os.path.exists(path) == False):
        generate_embeddings(path, batch_size=batch_size, num_work=num_work)

    # Load the training and testing data
    X, y = get_data(TRAIN_TRIPLETS, path=path)

    # Create data loaders for the training and testing data
    print('Generating loader for training data.')
    train_loader = create_loader_from_np(X, y, train=True, batch_size=batch_size, num_workers=num_work)
    print('Done generating loader for training data.')

    # Define a model and train it
    model, train_val_loss = train_model(model, train_loader,
                                                 loss_function, optimizer,
                                                 epochs, device, batch_size, split)

    

    # test the model on the test data
    if test:
        X_test, _ = get_data(TEST_TRIPLETS, train=False, path=path)
        # For testing we use a larger batch size but adjust to pretrain model
        test_loader = create_loader_from_np(X_test, train=False, batch_size=100, shuffle=False, num_workers=num_work)
        test_model(model, test_loader)
        print("Results saved to results.txt")

    # Plot the training and validation loss
    plot_loss(train_val_loss, epochs)

# What has been tried and yields similar results, batch sizes of 64
# One layer, lr=0.01.
# Two layers, lr=0.01 and 0.001 where the intermediate layer has 2056 nodes.

# Batch sizes of 100
# Three layers, lr=0.01, 0.001 intermediate layer has 3056 and 1024 nodes.