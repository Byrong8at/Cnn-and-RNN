import os
import re
import gdown
import zipfile
from pathlib import Path
import shutil

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

def copy(folder):
    """
    Copy the folder from raw data to the new one to after transform with the size wanted
    Folder: Folder where the raw data will be copy
    """
    folder_raw = Path("./data/raw")
    if not folder_raw.exists():
        print(f"Source folder not found: {folder_raw}")
        return []

    created_paths = []
    subdirectories = [p for p in folder_raw.iterdir() if p.is_dir()]

    for src_subfolder in subdirectories:
        if src_subfolder.name=="Photo":
            dest_subfolder = folder / "True"
            created_paths.append(dest_subfolder)

        else:
            dest_subfolder = folder / "False"
            if dest_subfolder not in created_paths:
                created_paths.append(dest_subfolder)

        dest_subfolder.mkdir(parents=True, exist_ok=True)        
        for file_path in src_subfolder.iterdir():
            if file_path.is_file():
                dest_file = dest_subfolder / f"{src_subfolder.name}_{file_path.name}"
                if not dest_file.exists():
                    shutil.copy2(file_path, dest_file)
        print(f"{src_subfolder.name} to {dest_subfolder}")





