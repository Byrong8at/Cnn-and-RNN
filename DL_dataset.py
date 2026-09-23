import os
import re
import gdown
import zipfile

#Choisir les étapes qu'on souhaite lancé comme le dl des fichiers, l extraction etc...
step_DL=False      
step_unzip=True

def download_file(url, folder):
    # 1. Extract the unique Google Drive File ID from the URL
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
    if not match:
        print(f"Error on {url}: Could not extract Google Drive File ID.")
        return
    
    file_id = match.group(1)
    output = f"data/downloaded_file{file_id}.zip"
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

urls=[
    "https://drive.google.com/file/d/1Gco_h5WCkqHJhVzUf5nOXVWa7wjKv-oX/view?usp=sharing",
    "https://drive.google.com/file/d/1SvFZNAQ7rD5IQUGNwbuKfBb0ateQmOYy/view?usp=sharing",
    "https://drive.google.com/file/d/1J-WupkQTuwmniUiO-E0rMQzsl2Pi1J5o/view?usp=sharing",
    "https://drive.google.com/file/d/1Khs9gpNywmoAfQMQ1Z7_ggge5wZepKj_/view?usp=sharing",
    "https://drive.google.com/file/d/16CatMjdgNhGkvlss5lmpzvf-AbV-6Q8T/view?usp=sharing"
]


folder = "data"
if not os.path.exists(folder):
    os.makedirs(folder)

if step_DL==True:
    print("File download:")
    for link in urls:
        download_file(link,folder)

if step_unzip==True:
    Get_file(folder)