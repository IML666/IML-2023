# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import os
import torch
import torchvision
import torchvision.models as models
from torchvision import transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn import preprocessing as pre
from torchvision.models import resnet50, ResNet50_Weights, efficientnet_b3
### Use progress bar
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_embeddings():
    """
    Transform, resize and normalize the images and then use a pretrained model to extract 
    the embeddings.
    """
    # TODO: define a transform to pre-process the images
    train_transforms = transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

    train_dataset = datasets.ImageFolder(root="P3/dataset/", transform=train_transforms)
    # Hint: adjust batch_size and num_workers to your PC configuration, so that you don't 
    # run out of memory
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=64,
                              shuffle=False,
                              pin_memory=True, num_workers=8)

    # TODO: define a model for extraction of the embeddings (Hint: load a pretrained model,
    #  more info here: https://pytorch.org/vision/stable/models.html)

    # model = nn.Module()

    ### Use ResNet18 as pretrained model
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    # model = models.resnet18(pretrained=True)

    ### Use ResNet50
    # model = resnet50(weights="IMAGENET1K_V2")

    ### Use EfficientNetB3
    model = efficientnet_b3(weights='IMAGENET1K_V1')


    # Use the model in evaluation mode to get the calculated features
    model.eval()

      

    embeddings = []
    embedding_size = (list(model.children())[-1][1]).in_features # Get in_features via debugger, for efficientnet_b3 we need list(model.children())[-1][1]
    num_images = len(train_dataset)
    embeddings = np.zeros((num_images, embedding_size))

    # TODO: Use the model to extract the embeddings. Hint: remove the last layers of the 
    # model to access the embeddings the model generates.

    ### Define last layer or specific one

    model = nn.Sequential(*list(model.children())[:-1])     # Takes resnet or efficientnet model and cuts the evaluation stage


    with torch.no_grad():  

        # Note that len(train_loader) = 157 which corresponds to 10048 = 157*64  

        for batch_index, (image, image_index) in enumerate(tqdm(train_loader)):  

            # Create embeddings i.e. the features after avgpool layer

            extracted_features = model(image)

            # Observe that extracted_features.shape = torch.Size([64, 2048, 1, 1]). However, we need extracted_features.squeeze().shape = torch.Size([64, 2048])

            extracted_features = extracted_features.squeeze().cpu().numpy()

            # Since we have chosen batch_size = 64 fill embeddings in this way

            embeddings[batch_index * train_loader.batch_size : (batch_index + 1) * train_loader.batch_size] = extracted_features



    np.save('P3/dataset/embeddings_otto_efficientnetb0.npy', embeddings)


def get_data(file, train=True):
    """
    Load the triplets from the file and generate the features and labels.

    input: file: string, the path to the file containing the triplets
          train: boolean, whether the data is for training or testing

    output: X: numpy array, the features
            y: numpy array, the labels
    """
    triplets = []
    with open(file) as f:
        for line in f:
            triplets.append(line)

    # generate training data from triplets
    train_dataset = datasets.ImageFolder(root="P3/dataset/",
                                         transform=None)
    filenames = [s[0].split('/')[-1].replace('.jpg', '') for s in train_dataset.samples]
    embeddings = np.load('P3/dataset/embeddings_otto_efficientnetb0.npy')
    # TODO: Normalize the embeddings across the dataset
    
    # Use standard normalisation
    embeddings = (embeddings - np.mean(embeddings, axis=0)) / np.std(embeddings, axis=0)

    file_to_embedding = {}
    for i in range(len(filenames)):
        file_to_embedding[filenames[i]] = embeddings[i]
    X = []
    y = []
    # use the individual embeddings to generate the features and labels for triplets
    print(f"Create set from {file}")
    for t in tqdm(triplets):
        emb = [file_to_embedding[a] for a in t.split()]
        X.append(np.hstack([emb[0], emb[1], emb[2]]))
        y.append(1)
        # Generating negative samples (data augmentation)
        if train:
            X.append(np.hstack([emb[0], emb[2], emb[1]]))
            y.append(0)

    X = np.vstack(X)
    y = np.hstack(y)

    # if train:
    #     np.save('P3/dataset/X.npy', X)
    #     np.save('P3/dataset/y.npy', y)
    # else:
    #     np.save('P3/dataset/X_test.npy', X)
    
    print("return")
    return X, y

# Hint: adjust batch_size and num_workers to your PC configuration, so that you don't run out of memory
def create_loader_from_np(X, y = None, train = True, batch_size=64, shuffle=True, num_workers = 4):
    """
    Create a torch.utils.data.DataLoader object from numpy arrays containing the data.

    input: X: numpy array, the features
           y: numpy array, the labels
    
    output: loader: torch.data.util.DataLoader, the object containing the data
    """
    print("Load data")
    if train:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float), 
                                torch.from_numpy(y).type(torch.long))
    else:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float))
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        pin_memory=True, num_workers=num_workers)
    print("finish load")
    return loader

# TODO: define a model. Here, the basic structure is defined, but you need to fill in the details
class Net(nn.Module):
    """
    The model class, which defines our classifier.
    """
    def __init__(self):
        """
        The constructor of the model.
        """
        super().__init__()
        self.fc = nn.Linear(4608, 2)

    def forward(self, x):
        """
        The forward pass of the model.        

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        x = self.fc(x)
        x = F.relu(x)
        return x

def train_model(train_loader):
    """
    The training procedure of the model; it accepts the training data, defines the model 
    and then trains it.

    input: train_loader: torch.data.util.DataLoader, the object containing the training data
    
    output: model: torch.nn.Module, the trained model
    """
    model = Net()
    model.train()
    model.to(device)
    n_epochs = 10

    losses = []
    # TODO: define a loss function, optimizer and proceed with training. Hint: use the part 
    # of the training data as a validation split. After each epoch, compute the loss on the 
    # validation split and print it out. This enables you to see how your model is performing 
    # on the validation data before submitting the results on the server. After choosing the 
    # best model, train it on the whole training data.

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("Train model")
    for epoch in tqdm(range(n_epochs)):        
        for [X, y] in train_loader:
            y_pred = model.forward(X)
            loss = criterion(y_pred, y)
            losses.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model

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
    with torch.no_grad(): # We don't need to compute gradients for testing
        for [x_batch] in loader:
            x_batch= x_batch.to(device)
            predicted = model(x_batch)
            predicted = predicted.cpu().numpy()
            # Rounding the predictions to 0 or 1
            predicted[predicted >= 0.5] = 1
            predicted[predicted < 0.5] = 0
            predictions.append(predicted)
        predictions = np.vstack(predictions)
    np.savetxt("results.txt", predictions, fmt='%i')


# Main function. You don't have to change this
if __name__ == '__main__':
    TRAIN_TRIPLETS = 'P3/train_triplets.txt'
    TEST_TRIPLETS = 'P3/test_triplets.txt'

    # generate embedding for each image in the dataset
    if(os.path.exists('P3/dataset/embeddings_otto_efficientnetb0.npy') == False):
        generate_embeddings()

    # load the training
    X, y = get_data(TRAIN_TRIPLETS)
    # get_data(TRAIN_TRIPLETS)

    # X = np.load('P3/dataset/X.npy')
    # y = np.load('P3/dataset/y.npy')

    train_loader = create_loader_from_np(X, y, train = True, batch_size=64)



    # load testing data
    X_test, _ = get_data(TEST_TRIPLETS, train=False)
    # get_data(TEST_TRIPLETS, train=False)

    # X_test = np.load('P3/dataset/X_test.npy')

    test_loader = create_loader_from_np(X_test, train = False, batch_size=2048, shuffle=False)

        
    # # define a model and train it

   
    model = train_model(train_loader)

    
    # test the model on the test data
    test_model(model, test_loader)
    print("Results saved to results.txt")
