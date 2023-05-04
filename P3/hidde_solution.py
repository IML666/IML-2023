# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the
# First, we import necessary libraries:
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import os
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.datasets as datasets
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

    # The train dataset, loaded in batches.
    train_dataset = datasets.ImageFolder(root="dataset/", transform=train_transforms)

    # We load all the data at once. Hence, a massive batch size.
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              pin_memory=True, num_workers=num_work)

    # Pretrained model resnet50, use the newest version with weights IMAGENET1K_V2, 80.858% accuracy.
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    # Put model into evaluation mode to get the calculated values for each entry.
    model.eval()

    # Sent model to the gpu
    model.to(device)

    # Extract the input of the last layer. These are the features.
    embedding_size = list(model.children())[-1].in_features
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
    train_dataset = datasets.ImageFolder(root="dataset", transform=None)
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
        # self.fc1 = nn.Linear(6144, 2056)
        # self.fc2 = nn.Linear(2056, 2)
        self.fc1 = nn.Linear(6144, 3056)
        self.fc2 = nn.Linear(3056, 1024)
        self.fc3 = nn.Linear(1024, 1)
        self.function = function

    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        x = self.fc1(x)
        x = self.function(x)
        x = self.fc2(x)
        x = self.function(x)
        x = self.fc3(x)
        x = F.sigmoid(x)
        x = x.squeeze()
        return x


def update_line(hl, new_data):
    hl.set_xdata(np.append(hl.get_xdata(), new_data[0]))
    hl.set_ydata(np.append(hl.get_ydata(), new_data[1]))
    plt.draw()


def train_model(model, train_loader, loss_function, optimizer, epochs, device, batch_size):
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

    print(f'Running training over {epochs} epochs')
    for epoch in range(epochs):
        for i, ([X, y]) in enumerate(tqdm(train_loader)):
            X = X.to(device)
            y = y.to(device)
            if i != 1e4 // batch_size:
                model.train()
                y_pred = model.forward(X)
                loss = loss_function(y_pred, y)

                # Since it still needs to run backpropagation this does not really work.
                # value = deepcopy(loss).cpu().numpy()
                # losses.append(value)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            else:
                with torch.no_grad():
                    model.eval()
                    y_pred = model.forward(X)
                    loss = loss_function(y_pred, y)
                    value = loss.cpu().numpy()
                    validation_loss.append(loss)

                    # update_line(hl, [epoch, value])
                    print('\n')
                    print(f'Epoch: {epoch + 1}, Loss: {value}, Badge_number: {1e4 // batch_size}')
                    accuracy = metric(y_pred, y)
                    print(f'Accuracy: {accuracy}')

    return model, losses, validation_loss


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
    np.savetxt("results.txt", predictions, fmt='%i')


# Main function. You don't have to change this
if __name__ == '__main__':

    # the device objectss indexes all available GPU's, since there is only one NVidea GPU in my pc it is cuda:0,
    # or cuda() .
    # use list(range(torch.cuda.device_count())) to find all avaible GPU's.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Used device: {device}')
    # device = "cpu"

    # Datasets
    TRAIN_TRIPLETS = 'train_triplets.txt'
    TEST_TRIPLETS = 'test_triplets.txt'

    # Path to dataset
    embedding_name = "hiddes_embeddings.npy"
    Path = os.path
    dir_path = Path.join(Path.dirname(__file__), "dataset")
    path = Path.join(dir_path, embedding_name)

    # Create model
    model = Net(F.relu)
    model.to(device)

    # Setup parameters
    batch_size = 100
    num_work = 6
    epochs = 15
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.BCELoss()  # nn.CrossEntropyLoss() only for 2 or more classes
    test = False

    # generate embedding for each image in the dataset
    if (os.path.exists(path) == False):
        generate_embeddings(path, batch_size=batch_size, num_work=num_work)

    # Load the training and testing data
    X, y = get_data(TRAIN_TRIPLETS, path=path)

    # Create data loaders for the training and testing data
    print('Generating loader for training data.')
    train_loader = create_loader_from_np(X, y, train=True, batch_size=batch_size, num_workers=num_work)

    # Define a model and train it
    model, losses, validation_loss = train_model(model, train_loader,
                                                 loss_function, optimizer,
                                                 epochs, device, batch_size)

    # test the model on the test data
    if test:
        X_test, _ = get_data(TEST_TRIPLETS, train=False, path=path)
        test_loader = create_loader_from_np(X_test, train=False, batch_size=2048, shuffle=False, num_workers=num_work)
        test_model(model, test_loader)
        print("Results saved to results.txt")

# What has been tried and yields similar results, batch sizes of 64
# One layer, lr=0.01.
# Two layers, lr=0.01 and 0.001 where the intermediate layer has 2056 nodes.

# Batch sizes of 100
# Three layers, lr=0.01, 0.001 intermediate layer has 3056 and 1024 nodes.