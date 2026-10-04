import os
import re
import gdown
import zipfile
from pathlib import Path
import shutil
import tensorflow as tf

#Choisir les étapes qu'on souhaite lancé comme le dl des fichiers, l extraction etc...


def download_file(url, folder):
    """
    Extract the Google Drive File ID from the URL and download it
    """
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
    if not match:
        print(f"Error on {url}: Could not extract Google Drive File ID.")
        return
    
    file_id = match.group(1)
    output = f"{folder}/downloaded_file{file_id}.zip"
    try:
        gdown.download(url, output, quiet=False)
        print("Donwload complete")
    except Exception as e:
        print(f"Error for {url}: ", e)

def Get_file(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith('.zip'):  
                try:
                    zip_name=os.path.join(root, file) 
                    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
                        zip_ref.extractall(root)
                    print(f"File extract for {os.path.join(root, file)}")
                    os.remove(zip_name)
                except Exception as e:
                    print("ERROR : ",e)

def Check_corruption(dossier_racine):
    for filepath in dossier_racine.rglob('*'):
        if filepath.is_file():
            try:
                # Attempt to read and decode the image exactly as the dataset pipeline does
                image_contents = tf.io.read_file(str(filepath))
                _ = tf.image.decode_image(image_contents, expand_animations=False)
            except tf.errors.InvalidArgumentError:
                print(f"Corrupted image detected and deleted: {filepath}")
                os.remove(filepath) # Deletes the bad file automatically
            except Exception as e:
                print(f"Other issue with file {filepath}: {e}")

