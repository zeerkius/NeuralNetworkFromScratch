import numpy as np

class NN:
    # neural network creation
    # create structure x0 - > w0Tx0 -> x1  -> w1Tx1 - > x2 -> w2Tx2 -> ...... until we get final output y'
    # user creates hidden layers nodes
    # At the base case we get a dense connection to one output node
    def __init__(self,hidden_layers = 0 , neuron_depth = [] , loss = "sse"):
        self.hidden_layers = hidden_layers
        self.neuron_depth = neuron_depth
        self.loss = loss
        if len(set(self.neuron_depth)) != 1:
            raise ValueError("All nodes must be of same size  [3,3,3] , some variations like [3,4,2] work however are not allowed in this version ")
        self.hidden_layers += 1
        self.neuron_depth += [1]

        if 0 in self.neuron_depth:
            raise ValueError("node values must be greater than 0")
        if len(self.neuron_depth) != self.hidden_layers:
            raise ValueError("Incompatible Network Shape , Neurons must be mapped to each layer ie layers = 2 ,"
                             + "neuron_depth = [4,2] , 4 nodes in layer 1 , 2 nodes in layer 2")
        if self.loss != "sse":
            return ValueError("only sse" + "sum of squared error loss supported")

    def sse(self,x , y):
        batch_error = (sum(x) - sum(y)) ** 2
        return batch_error

    class Layer(): # so input from inital vector or the previous layer then does the calcuation to output to next layer
       def __init__(self , input_vector = [] , nodes = 1 , activation = "sigmoid" , next = None):
           self.input_vector = input_vector
           self.nodes = nodes
           self.activation = activation
           self.next = next

       def sse_sigmoid_gradient(self, ground_truth , prediction , variable):
           delta = (ground_truth - prediction) * (1 - prediction) * (variable)
           return delta

       def sse_relu_gradient(self, ground_truth , prediction , variable):
           if prediction <= 0:
               return 1
           else:
               delta = (ground_truth - prediction) * variable
               return delta


       def sigmoid(self,x): 
           s = np.exp(-x)
           s = (1 / (1 + s))
           return s

       def ReLU(self,x):
           if x <= 0: # technically wrog because it real values function arent differentiable on corners so x = 0 is undefined
                # however for this case we will leave it
               return 0
           else:
               return x

       def create_layer_pass(self , weight = 0.5 , err_lis = None , pred = None , possible_matrix = None): # supports only sigmoid
           if possible_matrix == None:
                m = len(self.input_vector)
                n = self.nodes
                lis = [weight for x in range(m * n)]
                weight_matrix = np.reshape(lis , (m , n))
                output_vector = np.transpose(weight_matrix) @ self.input_vector
                if self.activation == "sigmoid":
                    output_vector = self.sigmoid(output_vector)
                    for s in range(len(err_lis)):
                        err_lis[s].append(self.sse_sigmoid_gradient(ground_truth=pred,prediction = output_vector[s],variable = self.input_vector[s-1]))
                        # proper indexing to avoid collistion
                elif self.activation == "relu":
                    output_vector = self.ReLU(output_vector)
                    for s in range(len(err_lis)):
                        err_lis[s].append(self.sse_relu_gradient(ground_truth=pred,prediction = output_vector[s],variable = self.input_vector[s-1]))
                else:
                    raise ValueError("activation not supported , supported activation {sigmoid , relu} , network activations only sigmoid or relu")
                return output_vector
           else: # for gradient descent 
                m = len(self.input_vector)
                n = self.nodes
                possible_matrix = np.reshape(possible_matrix , (m , n))
                output_vector = np.transpose(possible_matrix) @ self.input_vector
                if self.activation == "sigmoid":
                    output_vector = self.sigmoid(output_vector)
                    for s in range(len(err_lis)):
                        err_lis[s].append(self.sse_sigmoid_gradient(ground_truth=pred,prediction = output_vector[s],variable = self.input_vector[s-1]))
                elif self.activation == "relu":
                    output_vector = self.ReLU(output_vector)
                    for s in range(len(err_lis)):
                        err_lis[s].append(self.sse_relu_sigmoid(ground_truth=pred,prediction = output_vector[s],variable = self.input_vector[s-1]))
                else:
                    raise ValueError("activation not supported , supported activation {sigmoid , relu} , network activations only sigmoid or relu")
                return output_vector


    def fit(self , X = None , Y = None , batch_size = None , learning_rate = 0.0000005 , beta = 0.3):
        if batch_size <= 10:
            raise ValueError(" SGD not supported mini batch must be greater than 10")
        error_cache = [[[] for x in range(n)] for n in self.neuron_depth]
        batch_counter = 0
        weight_initial = [0.5 for i in range(self.hidden_layers)] # each previous layer is intialized the same
        velocity = 0
        for i in range(len(X)):
            # iterate through data set
            res = X[i]
            if batch_counter % batch_size == 0 and batch_counter != 0:
                # iterate through the error cache
                # Let user know model status
                print(" The current batch error is " + str(self.sse(X[batch_counter - batch_size : batch_counter] , Y[batch_counter - batch_size : batch_counter])) , end = "\n")
                for vec in range(len(error_cache))[::-1]:  # backwards layer wise updating works exactly like back propogation for only last layer , other weights are averaged
                    # back propogation needs work
                    for wi in range(len(error_cache[vec]))[::-1]:
                        velocity = (velocity * beta) * ((1 - beta) * (sum(error_cache[vec][wi])))
                        weight_initial[vec] -= velocity * learning_rate 
                error_cache = [[[] for x in range(n)] for n in self.neuron_depth]
            else:
                for layers in range(len(self.neuron_depth)): # full pass through network

                    res = self.Layer(input_vector = res, nodes = self.neuron_depth[layers]).create_layer_pass(err_lis = error_cache[layers],weight = weight_initial[layers], pred = Y[i])
            batch_counter += 1

            




            
            


















           

        







            






            


            



