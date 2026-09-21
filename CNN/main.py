#import librairies and code part
import numpy as np
import os
from pathlib import Path
from PIL import Image
from reshape import *
from model import *
from conv import *
from pooling import *
from layer import *
import time

#Variable definition
epochs=1
learning_rate=0.001
patience=5
patience_epoch=0
val_loss=np.inf
l2_lambda=0.03

#Load image dataset
BASE_DIR = Path(__file__).resolve().parent
X_train, y_train = data_loading(str(BASE_DIR / "img_64x64" / "train"))
X_test, y_test = data_loading(str(BASE_DIR / "img_64x64" / "test"))

#Resize and reshape
X_train=resize(X_train)
X_test=resize(X_test)

X_train,y_train=new_img(X_train,y_train)

X_train=reshape(X_train)
X_test=reshape(X_test)

print("Shape of the image :")
print(X_train.shape)
print(X_test.shape)

start_time = time.time()

def save_weights(epoch,loss,conv,dense,filename):
    np.savez(filename, epoch=epoch, loss=loss, conv_filters=conv.filters, dense_weight=dense.weight, dense_bias=dense.bias)

def early_stop(patience,epoch,conv,dense,patience_epoch,val_loss,avg_loss):
    if avg_loss < val_loss:
        val_loss=avg_loss
        patience_epoch=0
        save_weights(epoch,val_loss,conv,dense,str(BASE_DIR / "model_weight"))
    else:
        patience_epoch+=1
    
    if patience <= patience_epoch:
        return True,patience_epoch,val_loss
    return False,patience_epoch,val_loss


#load the CNN model
conv=Conv3x3x3(8,3)#see if we modify the number of filter
pool=MaxPool2()
conv2 = Conv3x3x3(8,8) 
pool2 = MaxPool2()
flatten=Flatten()
#Calculate dense input size
dummy_out = conv.forward(X_train[0]) 
dummy_out = pool.forward(dummy_out)
dummy_out = conv2.forward(dummy_out)
dummy_out = pool2.forward(dummy_out)
dense_input_size = dummy_out.size
dense=Dense(dense_input_size,1)
dropout = Dropout(drop_rate=0.5)
#Add different layer


#evaluation
def evaluate(X_data, y_data, is_training=False):
    "Evaluate the model on a dataset and return Loss, Acc, Precision, Recall"
    total_loss = 0.0
    
    all_preds = []
    all_labels = []
    
    for image, label in zip(X_data, y_data):
 
        sigmoid_output = forward_pass(image, is_training=is_training)
        
  
        if is_training:
            d_out = sigmoid_output - label
            backward_pass(d_out) 
            

        loss = binary_cross_entropy(label, sigmoid_output)
        total_loss += loss
        

        all_preds.append(sigmoid_output)
        all_labels.append(label)
    

    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)
    

    avg_loss = float(total_loss.item() / len(X_data))
    acc = accuracy(all_labels_np, all_preds_np)  
    prec = precision(all_labels_np, all_preds_np) 
    rec = recall(all_labels_np, all_preds_np)     
    
    return avg_loss, acc, prec, rec

def backward_pass(d_out):
    d_out = dense.backprop(d_out, learning_rate, l2_lambda)
    d_out = dropout.backprop(d_out)
    d_out = flatten.backprop(d_out)
    d_out = pool2.backprop(d_out)
    d_out = d_out * relu_deriv(pool2.last_input) 
    d_out = conv2.backprop(d_out, learning_rate)
    d_out = pool.backprop(d_out)
    d_out = d_out * relu_deriv(pool.last_input)
    d_out = conv.backprop(d_out, learning_rate)
    return d_out


def forward_pass(image, is_training=True):
    out = conv.forward(image)
    out = relu(out)
    out = pool.forward(out)
    out = conv2.forward(out)
    out = relu(out)
    out = pool2.forward(out)
    out = flatten.forward(out)
    out = dropout.forward(out, is_training=is_training)
    out = dense.forward(out)
    return sigmoid(out)


for epoch in range(1, epochs + 1):
    print(f"starting epoch number: {epoch}")

    train_loss, train_acc, train_prec, train_rec = evaluate(X_train, y_train, is_training=True)
    print(f"Train -> Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | Prec: {train_prec:.2f}% | Recall: {train_rec:.2f}%")


    val_loss_curr, val_acc, val_prec, val_rec = evaluate(X_test, y_test, is_training=False)
    print(f"Valid -> Loss: {val_loss_curr:.4f} | Acc: {val_acc:.2f}% | Prec: {val_prec:.2f}% | Recall: {val_rec:.2f}%")

    #Call early stopping and save weight
    state, patience_epoch, val_loss = early_stop(patience, epoch - 1, conv, dense, patience_epoch, val_loss, val_loss_curr)
    if state:
        print("Early stopping triggered.")
        break

end_time = time.time() 
elapsed_time = end_time - start_time 
print(elapsed_time)

