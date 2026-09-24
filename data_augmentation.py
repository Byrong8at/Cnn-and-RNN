"""
Data Augmentation Script pour CNN - Classification Binaire (Photo vs Non-Photo)
================================================================================
Ce script utilise TensorFlow/Keras pour :
1. Charger les images depuis les dossiers True/False
2. Appliquer de la data augmentation à la volée sur la classe minoritaire (True/Photo)
3. Équilibrer les deux classes pour l'entraînement
4. Retourner des datasets prêts pour model.fit()

Utilisation:
    from data_augmentation import create_balanced_datasets
    train_ds, val_ds, test_ds = create_balanced_datasets(
        data_dir="data/process/64x64",
        img_size=(64, 64),
        batch_size=32
    )
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
import os


# ============================================================================
# 1. Pipeline de Data Augmentation (à la volée)
# ============================================================================
def create_augmentation_layer(img_size):
    """
    Crée un pipeline de data augmentation avec TensorFlow.
    Ces transformations sont appliquées UNIQUEMENT pendant l'entraînement
    et UNIQUEMENT sur la classe minoritaire pour l'équilibrer.
    
    Transformations appliquées:
    - Flip horizontal (miroir)
    - Rotation aléatoire (±15%)
    - Zoom aléatoire (±20%)
    - Translation aléatoire (±10%)
    - Contraste aléatoire
    - Luminosité aléatoire
    """
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom((-0.2, 0.2)),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.RandomBrightness(0.1),
    ], name="data_augmentation")
    
    return augmentation


# ============================================================================
# 2. Chargement et préparation des datasets
# ============================================================================
def load_and_count_images(data_dir):
    """
    Compte les images dans chaque sous-dossier (True/False).
    """
    data_path = Path(data_dir)
    counts = {}
    for class_dir in sorted(data_path.iterdir()):
        if class_dir.is_dir():
            n = len([f for f in class_dir.iterdir() 
                     if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
            counts[class_dir.name] = n
    return counts


def create_balanced_datasets(data_dir="data/process/64x64", 
                              img_size=(64, 64), 
                              batch_size=32,
                              val_split=0.15,
                              test_split=0.10,
                              seed=42):
    """
    Crée des datasets TensorFlow équilibrés avec augmentation à la volée.
    
    Stratégie d'équilibrage:
    - On charge toutes les images des deux classes
    - On calcule le ratio de déséquilibre
    - On sur-échantillonne la classe minoritaire (True) via:
        a) Répétition du dataset minoritaire
        b) Application d'augmentation sur les images répétées
    - Résultat: même nombre d'images dans chaque classe pour l'entraînement
    
    Args:
        data_dir: chemin vers le dossier contenant True/ et False/
        img_size: taille des images (hauteur, largeur) 
        batch_size: taille des batchs
        val_split: proportion pour la validation (0.15 = 15%)
        test_split: proportion pour le test (0.10 = 10%)
        seed: graine pour la reproductibilité
    
    Returns:
        train_ds, val_ds, test_ds: datasets TensorFlow prêts pour model.fit()
    """
    
    # --- Étape 1 : Compter les images par classe ---
    counts = load_and_count_images(data_dir)
    print("=" * 60)
    print("DISTRIBUTION DES CLASSES (avant augmentation)")
    print("=" * 60)
    for cls, count in counts.items():
        print(f"  {cls}: {count} images")
    
    total = sum(counts.values())
    minority_class = min(counts, key=counts.get)
    majority_class = max(counts, key=counts.get)
    minority_count = counts[minority_class]
    majority_count = counts[majority_class]
    
    ratio = majority_count / minority_count
    print(f"\n  Classe minoritaire: {minority_class} ({minority_count})")
    print(f"  Classe majoritaire: {majority_class} ({majority_count})")
    print(f"  Ratio de déséquilibre: {ratio:.2f}x")
    print(f"  Images à générer par augmentation: ~{majority_count - minority_count}")
    print("=" * 60)
    
    # --- Étape 2 : Créer le dataset complet avec train/val/test split ---
    train_split = 1.0 - val_split - test_split
    
    # Dataset de TRAIN
    train_ds_full = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split + test_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=None,  # Pas de batch ici, on le fera après l'équilibrage
        label_mode="int",
        class_names=["False", "True"],  # False=0, True=1
    )
    
    # Dataset de VALIDATION + TEST (on va le re-séparer)
    val_test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split + test_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=None,
        label_mode="int",
        class_names=["False", "True"],
    )
    
    # Séparer validation et test
    val_test_count = val_test_ds.cardinality().numpy()
    # Si cardinality retourne -1 ou -2 (inconnu), on compte manuellement
    if val_test_count < 0:
        val_test_count = sum(1 for _ in val_test_ds)
    
    val_count = int(val_test_count * (val_split / (val_split + test_split)))
    test_count = val_test_count - val_count
    
    val_ds = val_test_ds.take(val_count)
    test_ds = val_test_ds.skip(val_count)
    
    # --- Étape 3 : Séparer les classes dans le dataset de train ---
    # Filtrer par label
    train_minority = train_ds_full.filter(
        lambda img, label: tf.equal(label, 1 if minority_class == "True" else 0)
    )
    train_majority = train_ds_full.filter(
        lambda img, label: tf.equal(label, 0 if minority_class == "True" else 1)
    )
    
    # --- Étape 4 : Augmentation et sur-échantillonnage de la classe minoritaire ---
    augmentation_layer = create_augmentation_layer(img_size)
    
    # Compter les éléments dans chaque split de train
    train_minority_count = sum(1 for _ in train_minority)
    train_majority_count = sum(1 for _ in train_majority)
    
    print(f"\nDans le split de TRAIN:")
    print(f"  Classe minoritaire ({minority_class}): {train_minority_count}")
    print(f"  Classe majoritaire ({majority_class}): {train_majority_count}")
    
    # Calculer combien de fois répéter la classe minoritaire
    repeat_factor = int(np.ceil(train_majority_count / train_minority_count))
    
    # Sur-échantillonner la classe minoritaire en la répétant
    train_minority_augmented = train_minority.repeat(repeat_factor)
    
    # Appliquer l'augmentation sur les images minoritaires répétées
    train_minority_augmented = train_minority_augmented.map(
        lambda img, label: (augmentation_layer(img, training=True), label),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Prendre exactement le nombre nécessaire pour équilibrer
    train_minority_augmented = train_minority_augmented.take(train_majority_count)
    
    # --- Étape 5 : Recombiner et mélanger ---
    # Concaténer les deux classes
    train_ds_balanced = train_majority.concatenate(train_minority_augmented)
    
    # Mélanger le tout (important pour l'entraînement)
    buffer_size = train_majority_count + train_majority_count  # Taille du buffer de mélange
    train_ds_balanced = train_ds_balanced.shuffle(
        buffer_size=buffer_size,
        seed=seed,
        reshuffle_each_iteration=True
    )
    
    # --- Étape 6 : Normalisation (0-255 -> 0-1) et batching ---
    normalization = tf.keras.layers.Rescaling(1./255)
    
    # Appliquer la normalisation
    train_ds_final = train_ds_balanced.map(
        lambda img, label: (normalization(img), label),
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_ds_final = val_ds.map(
        lambda img, label: (normalization(img), label),
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    test_ds_final = test_ds.map(
        lambda img, label: (normalization(img), label),
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # --- Stats finales ---
    final_train_count = train_majority_count * 2  # Approximation après équilibrage
    print(f"\n{'=' * 60}")
    print("DATASETS FINAUX (après augmentation)")
    print(f"{'=' * 60}")
    print(f"  Train:      ~{final_train_count} images ({train_majority_count} par classe)")
    print(f"  Validation: {val_count} images")
    print(f"  Test:       {test_count} images")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {img_size}")
    print(f"  Normalisation: [0, 1]")
    print(f"{'=' * 60}")
    
    return train_ds_final, val_ds_final, test_ds_final


# ============================================================================
# 3. Méthode alternative : class_weight (sans sur-échantillonnage)
# ============================================================================
def compute_class_weights(data_dir="data/process/64x64"):
    """
    Calcule les poids de classes pour compenser le déséquilibre.
    À utiliser avec le paramètre class_weight de model.fit().
    
    Alternative plus légère à la data augmentation :
    - Pas de duplication d'images
    - Le modèle pénalise plus les erreurs sur la classe minoritaire
    
    Returns:
        dict: {0: poids_false, 1: poids_true}
    
    Usage:
        weights = compute_class_weights()
        model.fit(train_ds, class_weight=weights, ...)
    """
    counts = load_and_count_images(data_dir)
    total = sum(counts.values())
    n_classes = len(counts)
    
    # Formule standard : total / (n_classes * count_per_class)
    class_weights = {}
    for i, (cls, count) in enumerate(sorted(counts.items())):
        class_weights[i] = total / (n_classes * count)
    
    print("Class weights calculés:")
    for cls_id, weight in class_weights.items():
        cls_name = sorted(counts.keys())[cls_id]
        print(f"  Classe {cls_id} ({cls_name}): {weight:.4f}")
    
    return class_weights


# ============================================================================
# 4. Script principal - Test rapide
# ============================================================================
if __name__ == "__main__":
    import sys
    
    # Par défaut : 64x64
    size = 64
    if len(sys.argv) > 1:
        size = int(sys.argv[1])
    
    data_dir = f"data/process/{size}x{size}"
    img_size = (size, size)
    
    print(f"\n[*] Chargement des donnees depuis: {data_dir}")
    print(f"[*] Taille des images: {img_size}\n")
    
    # Option 1 : Data augmentation (sur-échantillonnage)
    print("=" * 60)
    print("OPTION 1 : Data Augmentation (sur-échantillonnage)")
    print("=" * 60)
    train_ds, val_ds, test_ds = create_balanced_datasets(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=32
    )
    
    # Vérifier un batch
    print("\n[+] Verification d'un batch de train:")
    for images, labels in train_ds.take(1):
        print(f"  Shape images: {images.shape}")
        print(f"  Shape labels: {labels.shape}")
        print(f"  Min/Max pixel: {images.numpy().min():.3f} / {images.numpy().max():.3f}")
        unique, counts_vals = np.unique(labels.numpy(), return_counts=True)
        print(f"  Distribution labels dans ce batch: {dict(zip(unique, counts_vals))}")
    
    # Option 2 : Class weights (alternative)
    print(f"\n{'=' * 60}")
    print("OPTION 2 : Class Weights (alternative)")
    print("=" * 60)
    weights = compute_class_weights(data_dir=data_dir)
