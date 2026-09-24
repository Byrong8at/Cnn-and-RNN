import os
import numpy as np
from PIL import Image

def data_loading(folder_path):
    """
    Load the different image and label on the two list
    X: image (jpg, png)
    Y: 0 or 1 for determine if it's the true object
    """
    x=[]
    y=[]

    for label in os.listdir(folder_path):
        class_folder = os.path.join(folder_path, label)

        if not os.path.isdir(class_folder):
            continue  

        if "false" in label.lower():
            numeric_label = 0
        else:
            numeric_label = 1

        for filename in os.listdir(class_folder):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(class_folder, filename)
                
                try:
                    img = Image.open(img_path).convert("RGB")
                    x.append(np.array(img)) 
                    y.append(numeric_label)
                except Exception as e:
                    print(f"Erreur lecture {filename}: {e}")

    return x, np.array(y)

def train_val_test_split(X, y, val_ratio=0.15, test_ratio=0.15):
    """
    Sépare les données aléatoirement en Train, Validation et Test.
    Par défaut : 70% Train, 15% Validation, 15% Test.
    """
    # Mélanger les indices aléatoirement
    indices = np.random.permutation(len(X))
    
    # Calculer le nombre d'images pour le test et la validation
    test_size = int(len(X) * test_ratio)
    val_size = int(len(X) * val_ratio)
    
    # Séparer les indices
    test_idx = indices[:test_size]
    val_idx = indices[test_size : test_size + val_size]
    train_idx = indices[test_size + val_size:]
    
    # Retourner les 3 jeux de données
    return (X[train_idx], X[val_idx], X[test_idx], 
            y[train_idx], y[val_idx], y[test_idx])

def resize(x, size=(64, 64)):
    """
    Resize the image if it's necessary
    x: all image
    size: the size we want
    """
    X_resized = []

    for img in x:
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize(size)
        X_resized.append(np.array(pil_img))

    return np.array(X_resized)

def reshape(x):
    """
    Normalize x on 255
    """
    X= np.array(x)
    X=X/255.0
    return X

def new_img(X,y):
    """
    Create new images with flip for data augmentation
    """

    fake_img=np.where(y == 0)[0]
    true_img=np.where(y == 1)[0]
    
    diff = len(true_img) - len(fake_img)

    if diff == 0:
        return X, y
    if diff > 0:
        clone = np.random.choice(fake_img, size=diff, replace=True)
        
        x_supp = X[clone]
        y_supp = y[clone]

        x_supp = np.array([np.fliplr(img) for img in x_supp])

        X = np.concatenate([X, x_supp], axis=0)
        y = np.concatenate([y, y_supp], axis=0)
    else:
        clone = np.random.choice(true_img, size=abs(diff), replace=True)
                
        x_supp = X[clone]
        y_supp = y[clone]
        
        x_supp = np.array([np.fliplr(img) for img in x_supp])
        
        X = np.concatenate([X, x_supp], axis=0)
        y = np.concatenate([y, y_supp], axis=0)
        
    perm = np.random.permutation(len(X))
    X = X[perm]
    y = y[perm]

    return X,y