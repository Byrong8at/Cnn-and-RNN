import streamlit as st
from PIL import Image
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input  
from tensorflow.keras.preprocessing.text import tokenizer_from_json  

# Variables
IMG_SIZE = 300
THRESHOLD = 0.3
MAX_WORDS = 20

CLASSIFIER_PATH = Path("workflow/models/classifier.keras")
DENOISER_PATH   = Path("workflow/models/denoiser.keras")
CKPT_DIR        = Path("checkpoints_gru")

#  Côté fonctions 
class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super().__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, axis=1)
        attention_hidden = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        scores = self.V(attention_hidden)
        attention_weights = tf.nn.softmax(scores, axis=1)
        context_vector = tf.reduce_sum(attention_weights * features, axis=1)
        return context_vector, attention_weights

class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size,
                 attention_units=512, dropout_rate=0.3, gru_dropout=0.0):
        super().__init__()
        self.units = units
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.attention = BahdanauAttention(attention_units)
        self.gru = layers.GRU(units, return_sequences=True, return_state=True)
        self.fc1 = layers.Dense(units)
        self.fc2 = layers.Dense(vocab_size)

    def call(self, word_id, features, hidden, training=False):
        context_vector, attention_weights = self.attention(features, hidden)
        x = self.embedding(word_id)
        context = tf.expand_dims(context_vector, axis=1)
        x = tf.concat([context, x], axis=-1)
        output, hidden = self.gru(x, initial_state=hidden, training=training)
        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2]))
        logits = self.fc2(x)
        return logits, hidden, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

@st.cache_resource 
def load_models():
    classifier = tf.keras.models.load_model(CLASSIFIER_PATH, compile=False)
    denoiser = tf.keras.models.load_model(DENOISER_PATH, compile=False)

    config = json.loads((CKPT_DIR / "config.json").read_text())
    tokenizer = tokenizer_from_json((CKPT_DIR / "tokenizer.json").read_text(encoding="utf-8"))
    CAPTION_SIZE = tuple(config["image_size"])
    start_token_id = config["start_token_id"]
    end_token_id = config["end_token_id"]

    inception = InceptionV3(weights="imagenet", include_top=False, input_shape=(*CAPTION_SIZE, 3))
    inception.trainable = False
    inputs = Input(shape=(*CAPTION_SIZE, 3))
    image_encoder = Model(inputs, inception(inputs, training=False))

    encoder = CNN_Encoder(config["embedding_dim"])
    decoder = RNN_Decoder(config["embedding_dim"], config["rnn_units"], config["vocab_size"])

    dummy = encoder(tf.zeros((1, 64, 2048)))
    decoder(tf.constant([[start_token_id]]), dummy, decoder.reset_state(1))

    tf.train.Checkpoint(encoder=encoder, decoder=decoder).read(
        str(CKPT_DIR / "best")
    ).assert_existing_objects_matched()

    return classifier, denoiser, image_encoder, encoder, decoder, tokenizer, config

classifier, denoiser, image_encoder, encoder, decoder, tokenizer, config = load_models()
CAPTION_SIZE = tuple(config["image_size"])
start_token_id = config["start_token_id"]
end_token_id = config["end_token_id"]

# ───────── Pipeline ─────────
def load_image_from_array(pil_image):
    """✅ Accepte directement une image PIL au lieu d'un chemin"""
    img = np.array(pil_image.convert("RGB"))
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return img[None, ...]

def generate_caption(image):
    x = tf.image.resize(image, CAPTION_SIZE)
    x = preprocess_input(x * 255.0)
    features = tf.reshape(image_encoder(x, training=False), (1, 64, 2048))
    encoded = encoder(features)

    hidden = decoder.reset_state(1)
    word_id = tf.constant([[start_token_id]])
    words = []
    for _ in range(MAX_WORDS):
        logits, hidden, _ = decoder(word_id, encoded, hidden)
        scores = logits[0].numpy()
        scores[[0, 1, start_token_id]] = -1e9
        predicted_id = int(np.argmax(scores))
        if predicted_id == end_token_id:
            break
        words.append(tokenizer.index_word.get(predicted_id, ""))
        word_id = tf.constant([[predicted_id]])
    return " ".join(words)

def pipeline(pil_image):
    raw = load_image_from_array(pil_image)
    image = raw / 255.0

    # Partie CNN
    prob = float(classifier.predict(raw, verbose=0)[0][0])
    if prob < THRESHOLD:
        return None, None, prob  

    # Partie DAE
    clean = np.clip(denoiser.predict(image, verbose=0), 0, 1)

    # Partie Captioning
    caption = generate_caption(clean)
    return clean, caption, prob

#Interface
st.set_page_config(
    page_title="TouNum viewver",
    page_icon="🖼️",
    layout="wide" 
)

st.title("🖼️ TouNum viewver : déduit automatiquement ce que ton image contient")

col_texte, col_video = st.columns([2, 1])

with col_texte:
    st.write("Validation si une image ou nan, denoising de l'image et captioning de cette dernière.")
with col_video:
    video_path = Path("src/tuto.mp4")
    if video_path.exists():
        st.video(str(video_path))  
    else:
        st.warning("Vidéo tuto.mp4 introuvable")

st.subheader("Uploadez une image pour l'afficher.")
fichier = st.file_uploader("Choisissez une image", type=["png", "jpg", "jpeg"])

if fichier is not None:
    image = Image.open(fichier)
    st.image(image, caption=fichier.name,  width=300)
    st.success(f"✅ Image **{fichier.name}** chargée avec succès !")

    if st.button("🔍 Valider l'image"):
        with st.spinner("Analyse en cours..."):
            clean, caption, prob = pipeline(image)  

        if clean is None: 
            st.error(f"Votre image n'est pas valide !", icon="❌")
        else:
            st.info("Votre image est bien reconnue")
            st.image(clean.squeeze(), caption=fichier.name,  width=300)  
            st.write("Caption :", caption)
else:
    st.info("Importer votre image")