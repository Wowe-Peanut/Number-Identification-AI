from mnist import MNIST
import numpy as np
import math
import pickle
np.set_printoptions(suppress=True)

class Layer:
    def __init__(self, size):
        #“Xavier initialization“ b/c we are using sigmoid
        rr = 2*(1/math.sqrt(size[0]))
        self.size = size
        self.weights = np.random.rand(size[1],size[0])*rr-(rr/2)
        self.baises = np.zeros(size[1])+0.1 #Low starting bais
        self.acts = np.empty(size[1])
        self.z = np.empty(size[1])

        #Gradients
        self.reset_grads()

    def reset_grads(self):
        self.act_grads = np.zeros(self.size[1])
        self.weight_grads = np.zeros((self.size[1],self.size[0]))
        self.bais_grads = np.zeros(self.size[1])
    
def sigmoid(x):
    return 1/(1+math.e**-x)

def sigmoidP(x):
    return sigmoid(x)*(1-sigmoid(x))
      
#When calculating dcda, pre-compute all dc/dAct values before hand backwards

#Note: Too many epochs could result in the model over-fitting to the training data
#This would result it in being good at the training set, but bad at anything else

#Note: Layers has 4 elements and the first one is only for its activation (it has no weights)

mndata = MNIST(r"C:\Users\Ryan O'Mullan\Desktop\samples")
training_images, training_labels = mndata.load_training()
print("Training set loaded...")

learning_rate = 0.1
epochs = 10
batch_size = 32 #MUST BE A FACTOR OF 60000
layer_sizes = [784,16,16,10]

layers = [Layer([layer_sizes[i-1],layer_sizes[i]]) for i in range(len(layer_sizes))]
with open(r"C:\Users\Ryan O'Mullan\Desktop\set1.pkl", "rb") as wb:
    for L in range(len(layer_sizes)):
        layers[L].weights = pickle.load(wb)
        layers[L].baises = pickle.load(wb)

def z(j,L):
    global layers
    return layers[L].z[j]

for e in range(epochs):
    print("Epoch {0}/{1} in progress...".format(e+1,epochs))
    for start in range(0, len(training_images), batch_size):
        #print("{0}/{1}".format(int(start/batch_size), epochs*int(len(training_images)/batch_size)))
        image_batch = training_images[start:start+batch_size]
        label_batch = training_labels[start:start+batch_size]
        
        for batch_index in range(batch_size):
            #Expected output
            expected = [1 if i == label_batch[batch_index] else 0 for i in range(layer_sizes[-1])]
            
            #Process image
            layers[0].acts = np.array(image_batch[batch_index])/255
            for L in range(1,len(layer_sizes)):
                layers[L].z = np.dot(layers[L].weights,layers[L-1].acts)+layers[L].baises
                layers[L].acts = sigmoid(layers[L].z)
    
            #Calculate Activation Gradients
            layers[-1].act_grads = np.array([2*(layers[-1].acts[j]-expected[j]) for j in range(layer_sizes[-1]) ])
            for L in range(len(layer_sizes)-2, 0, -1):
                for k in range(layer_sizes[L]):
                    layers[L].act_grads[k] = sum([(layers[L+1].weights[j][k]*sigmoidP(z(j,L+1))*layers[L+1].act_grads[j]) for j in range(layer_sizes[L+1])])

            #Add to weight and bais gradient
            for L in range(1,len(layer_sizes)):
                for j in range(layer_sizes[L]):
                    for k in range(layer_sizes[L-1]):
                        layers[L].weight_grads[j][k] += layers[L-1].acts[k]*sigmoidP(z(j,L))*layers[L].act_grads[j]
                

            #Average and apply weight gradient to weights
            for L in range(1,len(layer_sizes)):
                layers[L].weights -= layers[L].weight_grads/batch_size * learning_rate
                layers[L].baises -= layers[L].bais_grads/batch_size * learning_rate
                layers[L].reset_grads()
        

with open(r"C:\Users\Ryan O'Mullan\Desktop\set2.pkl", "wb") as wb:
    for L in range(len(layer_sizes)):
        pickle.dump(layers[L].weights, wb, pickle.HIGHEST_PROTOCOL)
        pickle.dump(layers[L].baises, wb, pickle.HIGHEST_PROTOCOL)
print("Done Training...")
input()





