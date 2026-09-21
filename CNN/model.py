import numpy as np

def relu(x):
    """
    if x>0:
        x=x 
    else:
        x=0
    """
    return np.maximum(0,x)


def sigmoid(x):
    """
    Determine whether it's the object or not. 
    The closer it is to 1, because the result is between 0 and 1,
    the more convincing the model is that this is the object.
    """
    x = np.clip(x, -500, 500)
    o_x=1/(1+np.exp(-x))
    return o_x

def binary_cross_entropy(y,p):
    """
    y: 0 or 1(0 not the object, 1 in the object)
    p: result of sigmoid
    """
    p=np.clip(p, 1e-15, 1 - 1e-15)# Avoid log(0) bc we cant divide by zero encountered in log
    L=-(y*np.log(p)+(1-y)*np.log(1-p))
    return L

#derivation function is use for backpropagadtion
def relu_deriv(x):
    x=np.where(x>0,1,0)
    return x

def sigmoid_deriv(x):
    s=sigmoid(x)
    omega=s*(1-s)
    return omega

#Metrics
def accuracy(y_true,y_test):
    """
    Define a % for number of true on the number tested
    y_true is 0 or 1
    y_test is from prediction between 0 and 1
    Both are list
    """
    #avoid broadcasting issue
    y_true = np.array(y_true).flatten()
    y_test = np.array(y_test).flatten()

    y_test = (y_test > 0.5).astype(int)
    y_correct = (y_true == y_test)
    return float(np.sum(y_correct)/len(y_test)*100)

def precision(y_true,y_test):
    """
    Define the precision of object is determine True
    y_true is 0 or 1
    y_test is from prediction between 0 and 1
    Both are list
    positive_true take only the sum of y true and y test for 1
    Precision only want result with 1
    """
   #avoid broadcasting issue
    y_true = np.array(y_true).flatten()
    y_test = np.array(y_test).flatten()

    y_test = (y_test > 0.5).astype(int)
    positive_true=np.sum((y_true==1 )& (y_test==1))
    predicted_positive = np.sum(y_test==1)
    if predicted_positive == 0:
        return 0.0
    return float(positive_true/predicted_positive*100)

def recall(y_true,y_test):
    """
    Define the number of true positive on true positive and negative false. 
    y_true is 0 or 1
    y_test is from prediction between 0 and 1
    Both are list
    positive_true take only the sum of y true and y test for 1
    negative_false take only the sum where the model have false and think it's not the object we want
    """

    y_true = np.array(y_true).flatten()
    y_test = np.array(y_test).flatten()
    
    y_test = (y_test > 0.5).astype(int)
    positive_true=np.sum((y_true==1 )& (y_test==1))
    negative_false=np.sum((y_true != y_test)&(y_true==1) )
    return float(positive_true/(positive_true+negative_false)*100)

