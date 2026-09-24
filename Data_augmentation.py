import os
import random
from PIL import Image, ImageEnhance
from tqdm import tqdm

SIZES = ["64", "128", "256", "512", "1024"]

def get_image_files(folder_path):
    """Recupère la liste des chemins des images valident"""
    if not os.path.exists(folder_path):
        return []
    return [
        os.path.join(folder_path, f) 
        for f in os.listdir(folder_path) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
    ]

def augment_image(image_path):
    """Modifie une image en suivant des caractéristique (inclinaison/luminosité/symétrie) afin de rajouter des images dans le dataset"""
    with Image.open(image_path) as img:
        angle = random.uniform(-15, 15)
        img = img.rotate(angle, resample=Image.BICUBIC, expand=False)
        
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            
        enhancer = ImageEnhance.Brightness(img)
        factor = random.uniform(0.8, 1.2)
        img = enhancer.enhance(factor)
        
        return img

def balance_dataset(size_folder):
    """Compare les datasets afin de vérifier l'équilibre des classe et génère des images si necessaire"""
    photos_dir = os.path.join(size_folder, "photo")
    others_dir = os.path.join(size_folder, "others")
    
    photos_files = get_image_files(photos_dir)
    others_files = get_image_files(others_dir)
    
    count_photos = len(photos_files)
    count_others = len(others_files)
    
    print(f"\n--- Équilibrage pour la taille : {size_folder} ---")
    print(f"Photos initiales : {count_photos} | Others initiaux : {count_others}")
    
    if count_photos == count_others:
        print("Le dataset est déjà équilibré.")
        return
    
    if count_photos > count_others:
        majority_count = count_photos
        minority_files = others_files
        target_dir = others_dir
        class_name = "others"
    else:
        majority_count = count_others
        minority_files = photos_files
        target_dir = photos_dir
        class_name = "photo"
        
    deficit = majority_count - len(minority_files)
    print(f"Classe minoritaire : {class_name}. Génération de {deficit} images...")
    
    if not minority_files:
        print("Erreur : La classe minoritaire est vide.")
        return

    for i in tqdm(range(deficit), desc=f"Balancing {class_name} ({size_folder})", unit="img"):
        source_path = random.choice(minority_files)
        try:
            augmented_img = augment_image(source_path)
            base_name = f"aug_{i}_{os.path.basename(source_path)}"
            save_path = os.path.join(target_dir, base_name)
            augmented_img.save(save_path, "JPEG")
        except Exception:
            continue

if __name__ == "__main__":
    for size in SIZES:
        if os.path.exists(size):
            balance_dataset(size)
        else:
            print(f"\nDossier '{size}' introuvable, ignoré.")