
import matplotlib.pyplot as plt
import numpy as np

def sampleShowImage(x, y, prob=.25):
    if(np.random.random_sample() <= prob):
        img = x[0].squeeze()
        label = y[0]
        plt.imshow(img, cmap="gray")
        plt.title(f"label: {label}")
        plt.show()

def calcConvOutDim(inputWidth, filters, fieldSize, 
    stride=1, padding=0, inputHeight=None
):
    if not inputHeight:
        inputHeight = inputWidth
    
    outputWidth = ((inputWidth - fieldSize + 2 * padding) / stride) + 1
    outputHeight = ((inputHeight - fieldSize + 2 * padding) / stride) + 1
    outputDepth = filters

    return (outputWidth, outputHeight, outputDepth)

def calcPoolOutDim(inputWidth, inputDepth, fieldSize=2, 
    stride=2, inputHeight=None
):
    if not inputHeight:
        inputHeight = inputWidth
    
    outputWidth = ((inputWidth - fieldSize) / stride) + 1
    outputHeight = ((inputHeight - fieldSize) / stride) + 1
    outputDepth = inputDepth

    return (outputWidth, outputHeight, outputDepth)
