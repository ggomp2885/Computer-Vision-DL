"""This is my documentation file for all Pytorch commands and functions"""

                        # Generic Code Samples
                                 #Imports
# import torch
# import torch.nn as nn                 # prebuilt neural net layers
# import torch.nn.functional as F       # activations functions
# import torch.optim as optim           # optimizer functions
                                #Functions
                                    #Creating tensors
# torch.tensor(listname)                # creates tensor with same data type as input (Copy) (Factory)  (Data type is “inferred” based on the incoming data) (Type Inference)
# torch.as_tensor(listname)             # (Share) (Higher performance at scale because there's only one total instance of the tensor) (Factory) (Data type is “inferred” based on the incoming data) (Type Inference)
# torch.eye(#)                          # creates a tensor with the “identity matrix” with # of rows (1’s along the diagonal)
# torch.zeros(#,#)                      # creates a tensor of shape #,# with all zeros
# torch.ones(#,#)                       # creates tensor of shape #,# with all one's
# torch.rand(#,#)                       # creates tensor of shape #,# with all random values between 0 and 1

	                                #Functions for creating NNs
# class Net(nn.Module):                 # Creates a class of functions
# 	super().func_name()                 # initiates the nn.module part of the class function
# self.layername = nn.Linear(input #, output #)        # creates a fully connected layer - For 3 hidden layers, repeat this 4 times because one is the input layer, and then the last hidden and output layer are in one line.

                                    #Python Functions
#len(tens_name.shape)                   # returns the “rank” of the tensor

                                    #Pytorch Methods
# tens_name.size()                      # returns shape of tensor
# tens_name.reshape(#,#)                # must reshape into tensor with same amount of elements
# tens_name.numel()                     # returns the number of elements in the array
                                    #To flatten a tensor
# tens_name.reshape(1,-1) + tens_name.squeeze()         # (define this function) flattens (reshape and squeeze together) all the elements into a 1D array. (reducing the rank). The -1 tells pytorch to choose the correct size based on the input data.
# tens_name = tens_name.view(1,#)

                                    # Pytorch Attributes
# tens_name.shape                       # returns shape of tensor
# tens_name.dtype                       # returns data type
# tens_name.device                      # returns gpu/cpu
# tens_name.layout                      # strided

                                    # Tensor Operations
# tens_name.sum()                       # adds all the elements of a tensor into one scalar value. Can add in (dim=0) to sum just the elements of the first axis, and 1 to do the same with just the elements on 2nd axis.
# tens_name.prod()                      # multiples all elements
# tens_name.mean()                      # returns a single value tensor of the avg of all elements
# tens_name.std()                       # std all elements
# tens_name.max()                       # returns the max value of all the elements in the array. Can also use the (dim=#) here to look at the maxes by axes
# tens_name.argmax()                    # returns the index of the max value of all the elements in the array. The index value is returned as if the function is completely “flattened” to a 1D array. Can also use the (dim=#) here to look at the  index of the maxes by axes. Remember that once you start honing in on specific axes, the indices that come up are very specifically the index within that axis.
# tens_name.mean().item()               # returns a single scalar value
# tens_name.mean(dim=#).tolist()        # returns a list of values
# tens_name.mean(dim=#).numpy()         # returns a numpy array of values
