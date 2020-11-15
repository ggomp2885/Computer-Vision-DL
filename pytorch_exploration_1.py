"""This is my documentation file for all Pytorch commands and functions"""

                        # Generic Code Samples
                                 #Imports
# import torch
# import torch.nn as nn                 # prebuilt neural net layers
# import torch.nn.functional as F       # activations functions
# import torch.optim as optim           # optimizer functions
                                #Functions
                                  #many functions can be written explicitly, such as torch.func_name(tens_name) or implicitly as in tens_name.func_name()
                                    #Creating tensors
# torch.tensor([#,#,#],[#,#,#])         # creates a tensor with rows and columns of the input #s
# torch.tensor(list_name)               # creates tensor with data type inferred from input (Copy) (Factory)
# torch.as_tensor(list_name)            # (Share) (Higher performance at scale because there's only one total instance of the tensor) (Factory)
# torch.eye(#)                          # creates a tensor with the “identity matrix” with # of rows (1’s along the diagonal)
# torch.diag(torch.ones(3))             # same as using the .eye(#) function
# torch.zeros(#,#)                      # creates a tensor of shape #,# with all zeros
# torch.ones(#,#)                       # creates tensor of shape #,# with all one's
# torch.rand(#,#)                       # creates tensor of shape #,# with all random values between 0 and 1
# torch.arange(start=#, end=#, step=#)  # creates a tensor starting with the first #, ending at the 2nd # (not inclusive), with a step size of the 3rd # - also, do not have to include these words
# torch.linspace(start=#, end=#, steps=#)   # creates a tensor starting with the first #, ending at the 2nd # (is inclusive), with the 3rd # of steps inbetween - also do not have to include these words
# torch.from_numpy(array_name)          # creates a tensor from a numpy array

                                    #Manipulating tensors
# sorted_tens_1, indices = torch.sort(tens_1, descending=False)
# torch.where(tens_>#, tens_, tens_*2)  # like a small lambda function, finds where values > #, sets value = to middle value, where this is not true, does 3rd operation
# tens_name.permute(#,#,#)              # switches the shape of the tensor around, using the orginal index values. IE, if you wanted a tensor of shape (50, 25, 30) to be (30, 25, 50) in this function you would input (2, 1, 0) # .t() is for ease of use, with only 2 dimensional tensors.
# tens_name.squeeze(#)                  # removes a dimension from a tensor, number here is the index of the dimension you want to remove.
# tens_name.unsqueeze(#)                # adds a dimension to a tensor, number here is the index where the new dimension should be placed, either infront or behind the indexes of the current dimensions.

                                #Methods
# tens_name.reshape(#,#)                # reshapes the tensor, (new shape must have same amount of elements) can put (1,-1) in here, as a keyword to flatten the tensor
# tens_name.view(#,#)                   # reshapes the tensor, only works on tensors in the same contigious memory block, and therefore can be more efficient at times, I probably wont use this for now.
# tens_name.short()                     # converts to int 16 (default is int 32)
# tens_name.half()                      # converts to float 16
# tens_name.float()                     # converts to float 32
# tens_name.long()                      # converts to int 64
# tens_name.double()                    # converts to float 64
# tens_name.bool()                      # converts to boolean
# tens_name.numpy()                     # converts to numpy array

                                #Attributes
# tens_name.shape()                     # returns shape of tensor (.size is the exact same thing)
# tens_name.numel()                     # returns the number of elements in the tensor
# tens_name.dtype                       # returns data type
# tens_name.device                      # returns gpu/cpu
# tens_name.layout                      #
# tens_name.ndimension()                # returns # of dimensions of the tensor

                                # Indexing tensors
# tens_name[#]                          # returns entire row
# tens_name[:,#]                        # returns entire column
# tens_name[(tens_name > # | tens_name < #    # returns elements above OR below these #s, can also use the & sign here

                                # Basic Tensor Math Operations
                                    # I can add a _ after any of these functions to do these operations, "in place," which takes less time and memory
                                    # broadcasting is a built-in feature for ease of use which allows you to do operations between tensors of different sizes
                                    # in each of these parenthesis, you can also specify dim=#, to just do the operation on a specific row. When specifying an axis, the index returned is relative to that axis.
# tens_name.sum()                       # adds all the elements of a tensor into one scalar value.
# tens_name.prod()                      # multiples all elements
# tens_name.std()                       # std all elements
# tens_name.max()                       # returns the max value of all the elements in the array.
# tens_name.mean()                      # returns a single value tensor of the avg of all elements
# tens_name.mean().item()               # returns a single scalar value
# tens_name.argmax()                    # returns index of max value of the elements. index value returned is as if the function was flattened to a 1D array.
# tens_name.argmin()                    # returns index of min value of the elements. index value returned is as if the function was flattened to a 1D array.
# tens_name.unique()                    # returns the unique values of the tensor
# tens_name_1 += tens_name_2            # Easiest way to write Add/Subtract/Multiply/Divide, in place
# tens_name < #                         # returns a tensor containing all elements less than this #
# tens_name ** #                        # raises all elements by this power
# tens_name.mm(tens_name_2)             # raises all elements by the element wise numbers in another matrix
# tens_name.cat(tens_name_2, dim=#)     # this adds tensors together similar to a "union" in SQL, the dim specifies whether you want to add the columns together or the rows, left blank it will do both.
# tens_name.clamp(max=#)                # sets all elements above this #, to this #, can also use min=# here
# tens_name.abs()                       # returns the absolute value of each element
# tens_name.any()                       # returns true if any values are true
# tens_name.all()                       # returns true if all values are true
# tens_name_1.dot(tens_name_2)          # dot product (multiples matrices together and adds all elements to return one single value).
# torch.eq(tens_name_1, tens_name_2)    # (explicit example) returns a boolean tensor where elements of two tensors are equal
# longer setup                          # can also do "batch multiplication" which is like "multiple" multiplication in one, not sure where Id end up using this though

	                                #Functions for creating NNs
# class cls_name(nn.Module):                 # Creates a class of functions
# 	  super().func_name()                 # initiates the nn.module part of the class function
#     self.lay_name_1 = nn.Linear(input #, output #)        # creates a fully connected layer - For 3 hidden layers, repeat this 4 times, because the first is the input layer, and the last line contains the hidden layer, and the output layer.
#     self.lay_name_2 = nn.Linear(input #, output #)        # creates a fully connected layer
#     self.lay_name_3 = nn.Linear(input #, output #)        # creates a fully connected layer
#     self.lay_name_4 = nn.Linear(input #, output #)        # creates a fully connected layer

                                            #EXAMPLE CODE
                                    #Example code to create simple FF NN with 3 hidden layers


# def forward(self, var_name1):
#     var_name = F.act_func(self.lay_name1(var_name1))
#     var_name = F.act_func(self.lay_name2(var_name1))
#     var_name = F.act_func(self.lay_name3(var_name1))
#     var_name = self.lay_name4(var_name1)  ## this may be unnecessary
#     return F.loss_func(var_name1, dim=1)
# 
# opt_var_name = optim.Adam(net.parameters(), lr=0.001)
# 
# EPOCHS = 3
# for epoch in range(EPOCHS):
#     for data in trainset:
#         X, y = data
#         net.zero_grad()
#         output = net(X.view(-1, input  # )
#         loss_var_name = F.loss_func_name(output, y)
#         loss_var_name.backward()
#         opt_var_name.step()
# 
                    # measuring accuracy
# Correct = 0
# Total = 0
# with torch.no_grad()
#    for data in trainset:
#        X, y = data
#        Output = net(X.view(-1, 784)
#             for idx, i in enumerate(output):
#                 if torch.argmax(i) == y[idx]:
#                     correct += 1
#                     total += 1
#        print(“Accuracy “, round(correct / total, 3))


                                            # EXAMPLE CODE OF BASIC FUNCTIONS -- All confirmed working 11/11/20

# import numpy as np
# import torch
# device = "cuda" if torch.cuda.is_available() else "cpu"
#
# my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device, requires_grad=True)
# print(my_tensor)
# print(my_tensor.dtype)
# print(my_tensor.device)
# print(my_tensor.shape)
# print("requires grad? {}".format(my_tensor.requires_grad))
# print()

# x = torch.empty(size=(3,3))
# print(x)
# x = torch.zeros(3,3)
# print(x)
# x = torch.ones(3,3)
# print(x)
# x = torch.rand(3,3)
# print(x)
# x = torch.eye(3,3)
# print(x)
# x = torch.arange(0, 5, 1)
# print(x)
# x = torch.linspace(.1,1,10)
# print(x)
# x = torch.empty(size=(1,5)).normal_(mean=0, std=1)
# print (x)
# x = torch.empty(size=(1,5)).uniform_(0,1)
# print(x)
# x = torch.diag(torch.ones(3))    # same as using the .eye function
# print(x)

# print(x.bool())
# print(x.short())
# print(x.long())

# np_array = np.zeros((5, 5))
# print(np_array.)
# tensor = torch.from_numpy(np_array)
# print(tensor.type)
# np_back = tensor.numpy()
# print(np_back.type)


                                                # EXAMPLE CODE OF 3 Layer NN for MNIST CV
                                                    # Contains: Imports, loading data, create NN, hyperparams, initalize NN, Loss and Optimizer, training, accuracy check
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from tqdm import tqdm


start = timer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


                                                    #Loading in MNIST Train and Test datasets from torchvision.datasets
                                                        # MNIST dataset shape = batch_size, colors=1, 28 pixels, 28 pixels
                                                        # datasets for images can have 2 different import files, one for the png images, the other, a csv for the labels of the images.
                                                        # Depending on the objects your looking to detect, it can increase accuracy to apply a bunch of random transforms to the image, such as 5% of grayscale, 5% of 45 degree rotation, resize the image up 10%, crop it down 10%, random saturation 5%, etc. In this way it learns to really understand the relationships that make up an object, in less than ideal cases. But keep in mind, if you wouldnt recognize it, with the transforms, than the net probably shouldnt be trained like this, for example a 100% vertical flip on a digit, actually changes what digit it is, so more is not better at this point.
                                                        # For FCNN, load as above
                                                        # For CNN, use input_channel=1
                                                        # For RNN, I consider it 28 time stamps, by 28 features, usually you wouldnt use an RNN for images though.
batch_size = 512
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# my_transforms = transforms.Compose([
#                     transforms.ToTensor(),
#                     transforms.Normalize(mean=(.55), std=(1.2))    # This step increases accuracy of the majority of datasets, even MNIST)
# ])

                                    #General notes about NN types:
                                        # RNN is the umbrella term, more specific types are gru's, and LSTMs, this RNN code can be very simply converted to these types by just changing "RNN" to "GRU" in the layer type. Then you can change the layer name in the NN function in 2 places as well.
                                        # No need to create this class if you're using a prebuilt model, simply set model = prebuilt_name(arg, arg, arg) down below

                                    # Create simple fully connected NN
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

                                    # Create simple Convolutional NN
# class CNN(nn.Module):
#     def __init__(self, in_channels = 1, num_classes=10):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1)) # these last 3 arguments are chosen based on a simple formula which feeds the output of this layer into the next layer with the same dimensions
#         self.pool = nn.MaxPool2d(kernel_size=(2,2), stride = (2,2))
#         self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
#         self.fc1 = nn.Linear(16*7*7, num_classes)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.pool(x)
#         x = F.relu(self.conv2(x))
#         x = self.pool(x)
#         x = x.reshape(x.shape[0],-1)
#         x = self.fc1(x)
#         return x
                                    # Create simple Recurrent NN - currently setup for bi-directional LSTM
                                        # a Bi-directional LSTM, is nearly the same as a normal LSTM, simply replace variable names for ease of understanding and change 3 things, set bidirectional=True and multiply hidden_size*2 and num_layers*2
# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, num_classes):
#         super(RNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
#         self.fc = nn.Linear(hidden_size*2, num_classes)                   # adding hidden_size*sequence_length in the first arg will capture all the data from the hidden states, using just hidden_size, will only capture the information from the last hidden state, which will speed up training time, and can increase/decrease accuracy depending on dataset, since the last state already has second hand information from the previous hidden states.
#
#     def forward(self, x):
#         h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)       # num_layers*2 is B-LSTM specific
#         c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)       # LSTM specific, num_layers*2 is B-LSTM specific
#
#         # Forward prop
#         out, _ = self.lstm(x, (h0, c0))                                 # for RNN and GRU, this line only holds (x, h0), for the LSTM, it holds (x, (h0, c0))
#         out = self.fc(out[:, -1, :])                                    # captures information from just last hidden state    #acc 93.4, time 240 sec, B-LTSM with same setup had acc 94.0, time 550 sec
#         # out = out.reshape(out.shape[0], -1)                           # captures information from all hidden states         #acc 96.1, time 320 sec
#         # out = self.fc(out)                                            # ^ included in line above
#         return out

                                    #Checkpoint functions
def save_checkpoint(state, filename="my_checkpoint.pt"):             # this will overwrite the file, so for multiple runs, youll want to change this function slightly, or use a new file.
    print("-> Saving Checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint):
    print("-> Loading Checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

load_model = False                          # for saving results to build on later, cant set to true, until I have loaded data in the file already

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"))

                                    #Loading a pre-trained model


# shape tests of model output with mock data
                                        #FC NN
# model = NN(784,10)
# x = torch.random(64, 784)                # first # is the # of rows, 2nd is the # of features
# print(model(x).shape)                    # checks that the output is correct
                                        #CNN
# model = CNN()
# x = torch.randn(64, 1, 28, 28)
# print(model(x).shape)



                                    # Hyperparameters
                                        # General HyperPs
num_epochs = 2                              # value represents how many times the loop runs to train the model on the same dataset)
num_classes = 10
learning_rate = .001
                                        # FC HyperPs
input_size = 784                            # value represents pixels*pixels
                                        # CNN HyperPs
# in_channels = 1                             # value represents amount of colors in image
                                        # RNN HyperPs
# input_size = 28
# sequence_length = 28
# num_layers = 2
# hidden_size = 256


                                    # create model, set Optimizer and Loss function
                                        #torch.torchvision.prebuilt_model_name(arg, arg, arg) contains alot of prebuilt NN structures

model = NN(input_size=input_size, num_classes=num_classes).to(device)        # FC NN
# model = CNN().to(device)                                                   # CNN
# model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)   # RNN

                                        # Same for all
criterion = nn.CrossEntropyLoss()                       # This line already contains, softmax, and negative log likelyhood
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                                        # For testing your NN on a single batch, add this line, and remove the (in enumerate) loop in the training loop.
data, targets = next(iter(train_loader))                # This line is for testing your NN on a single batch, you also comment out the for loop in the training loop, and "de-dent" the following lines

                                    # NN Training loop
for epoch in range(num_epochs):
    losses = []
                                        # part of the save and load function
    if epoch % 2:
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
    for batch_idx, (data, targets) in loop            # this is the loop that iterates over all batches, and the in enumerate part gives the index for each epoch
            # Put data in Cuda if possible
        targets = targets.to(device=device)
        data = data.to(device=device)               # FC NN and CNN specific
        # data = data.to(device=device).squeeze(1)      # RNN specific
        data = data.reshape(data.shape[0],-1)       # FC NN specific      # reshapes the 28,28 into a 784, by "squeezing" to a flat tensor

            # forward function
        scores = model(data)
        loss = criterion(scores, targets)
            # backward function
        optimizer.zero_grad()                                             # this line tells the network to sum the gradient of each batch, individually, then add it to the total, much higher accuracy this way
        loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1)   # this line works well with RNNs
            # gradient descent function
        optimizer.step()
        # progress bar updates
        loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
        loop.set_postfix(loss = loss.item(), acc=torch.rand(1).item())      # The accuracy here is set to random right now

# Check Accuracy
def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on train set')
    else:
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)#.squeeze(1)               # .squeeze(1) is all RNNs specific
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)                  # FC NN specific

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(f'Got {num_correct} / {num_samples} with accuracy of {float(num_correct)/float(num_samples)*100:.2f}')
    model.train(x)

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)

end = timer()
print(f'Time taken: {end-start:.1f} Seconds')


# all these things, batch size, number of epochs, number of layers, have an effect on accuracy,
# acc 96.9 after 20 epochs of FCNN with batch of 512 in 115 sec
# acc 97.3 after 40 epochs of FCNN with batch 512, in 240 sec (after loading from 20 above)