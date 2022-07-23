from tkinter import *   
from PIL import ImageGrab
from matplotlib import pyplot as plt
import numpy as np
import math
import pickle
np.set_printoptions(suppress=True)


class Paint(object):
    DEFAULT_PEN_SIZE = 42
    DEFAULT_COLOR = 'white'

    def __init__(self):
        self.root = Tk()
        self.root.title("Number Identification")

        self.clear_button = Button(self.root, text='Clear', command=self.clear)
        self.clear_button.grid(row=0, column=1)

        self.guess_label = Label(self.root, text="")
        self.guess_label.grid(row=0,column=2)
        
        self.check_button = Button(self.root, text='Check Digit', command = self.check)
        self.check_button.grid(row=0,column=3)
        
        self.c = Canvas(self.root, bg='black', width=420, height=420)
        self.c.grid(row=1, columnspan=5)
        
        self.setup()
        self.root.mainloop()

    def check(self):
        x = self.root.winfo_rootx() + self.c.winfo_x()
        y = self.root.winfo_rooty() + self.c.winfo_y()
        x1 = x + self.c.winfo_width()
        y1 = y + self.c.winfo_height()
        image = ImageGrab.grab().crop((x+2,y+2,x1-2,y1-2))
        image_data = np.array(image)
        guess = check_digit(image_data)
        self.guess_label.config(text="Guess:" + str(guess))
        
    def clear(self):
        self.c.delete('all')
        self.guess_label.config(text="")

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.DEFAULT_PEN_SIZE
        self.color = self.DEFAULT_COLOR
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)


    def paint(self, event):
        self.line_width = self.DEFAULT_PEN_SIZE
        paint_color = self.DEFAULT_COLOR
        
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None

def display_image(image):
    a = np.array(image, dtype='float')
    pixels = a.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()


class Layer:
    def __init__(self, size):
        self.size = size
        self.weights = np.empty((size[1],size[0]))
        self.baises = np.zeros(size[1])
        self.acts = np.empty(size[1])
        self.z = np.empty(size[1])

layer_sizes = [784,16,16,10]
layers = [Layer([layer_sizes[i-1],layer_sizes[i]]) for i in range(len(layer_sizes))]

#Parse trained weights and baises
with open(r"C:\Users\Ryan O'Mullan\Desktop\set2.pkl", "rb") as wb:
    for L in range(len(layer_sizes)):
        layers[L].weights = pickle.load(wb)
        layers[L].baises = pickle.load(wb)

def z(j,L):
    global layers
    return layers[L].z[j]

def sigmoid(x):
    return 1/(1+math.e**-x)

def pre_process(image):
    #Calculate center of mass, shift image until COM is in the center of the image
    return image

def check_digit(image_data):
    global layers,layer_sizes
    
    #Values are correct but now I need to scale it from 420x420 to 28x28
    high_res = np.array([[image_data[r][c][0] for c in range(image_data.shape[1])]for r in range(image_data.shape[0])])
    low_res = np.array([[sum([high_res[br*15+r][bc*15+c] for r in range(15) for c in range(15)])/225 for bc in range(28)] for br in range(28)])
    processed_image = pre_process(low_res)
    
    #display_image(low_res)
    #display_image(processed_image)
    
    layers[0].acts = np.array(processed_image.flatten())/255
    for L in range(1,len(layer_sizes)):
        layers[L].z = np.dot(layers[L].weights,layers[L-1].acts)+layers[L].baises
        layers[L].acts = sigmoid(layers[L].z)
    guess = np.argmax(layers[-1].acts)
    return guess
    
    
Paint() 
    
    

    
































