from mnist import MNIST
import numpy as np
import math
import pickle
from matplotlib import pyplot as plt
np.set_printoptions(suppress=True)

mndata = MNIST(r"C:\Users\Ryan O'Mullan\Desktop\samples")
test_images, test_labels = mndata.load_testing()

class Layer:
    def __init__(self, size):
        self.size = size
        self.weights = np.empty((size[1],size[0]))
        self.baises = np.zeros(size[1])
        self.acts = np.empty(size[1])
        self.z = np.empty(size[1])

layer_sizes = [784,16,16,10]
layers = [Layer([layer_sizes[i-1],layer_sizes[i]]) for i in range(len(layer_sizes))]

test_name = input("WB Set Name: ")
#Parse trained weights and baises
with open(r"C:\Users\Ryan O'Mullan\Desktop\{0}.pkl".format(test_name), "rb") as wb:
    for L in range(len(layer_sizes)):
        layers[L].weights = pickle.load(wb)
        layers[L].baises = pickle.load(wb)


def z(j,L):
    global layers
    return layers[L].z[j]

def sigmoid(x):
    return 1/(1+math.e**-x)

correct = [0 for i in range(10)]
total = [0 for i in range(10)]


for test_index in range(len(test_images)):
    layers[0].acts = np.array(test_images[test_index])/255
    for L in range(1,len(layer_sizes)):
        layers[L].z = np.dot(layers[L].weights,layers[L-1].acts)+layers[L].baises
        layers[L].acts = sigmoid(layers[L].z)
    guess = np.argmax(layers[-1].acts)
    actual = int(test_labels[test_index])
    total[actual] += 1
    if guess == actual:
        correct[guess] += 1
    """
    else:
        print("Guess:{0}, Actual:{1}".format(guess, actual))
        first_image = np.array(test_images[test_index], dtype='float')
        pixels = first_image.reshape((28, 28))
        plt.imshow(pixels, cmap='gray')
        plt.show()
        input()
    """
    
        
print("Testing Complete: {0}/{1} Correctly identified\n".format(sum(correct), sum(total)))
for i in range(10):
    print("{2}: {0}/{1}, {3}%".format(correct[i],total[i],i,int(correct[i]/total[i]*10000)/100))
input()
