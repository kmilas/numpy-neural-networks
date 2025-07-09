import numpy as np

def softmax(x):
    '''Softmax f(x) = exp(x)/ sum(exp(x))'''
 
    x = (x - np.max(x, axis=1, keepdims=True)) 
    return (np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True))

def crossentropy(pred, target_one_hot):
    '''Cross Entropy loss'''
    # For stability clipping 

    eps = 1e-8
    softmax_probs = softmax(pred)
    softmax_probs = np.clip(softmax_probs, eps, 1. - eps)
    ce = -np.sum(target_one_hot*np.log(softmax_probs))
    ce = ce/pred.shape[0]
    return ce  

def dcrossentropy(pred, target): 
    '''Derivative of cross entropy loss'''
    return (softmax(pred) - target) / pred.shape[0]

class CrossEntropyLoss(object):
    def __init__(self, num_labels=10):
        self.labels = None
        self.num_labels = num_labels

    def one_hot_encoding(self,target):
        batch_size = target.shape[0]
        self.labels = np.zeros((batch_size,self.num_labels))

        for i in range(batch_size):
            self.labels[i, int(target[i])] = 1

    def forward(self,pred,target):
        self.one_hot_encoding(target)

        eps = 1e-8
        softmax_probs = softmax(pred)
        softmax_probs = np.clip(softmax_probs, eps, 1. - eps)
        ce = -np.sum(self.labels*np.log(softmax_probs))
        ce = ce/pred.shape[0]

        return ce

    def __call__(self,pred,target):
        return self.forward(pred,target)

    def backward(self,pred):
        return (softmax(pred) - self.labels) / pred.shape[0]