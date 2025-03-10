import streamlit as st
import cv2
import numpy as np
import joblib
from keras_facenet import FaceNet
from PIL import Image


def get_embedding(model, face_image):
    """Génère un embedding pour FaceNet ou VGG16."""
    # Déterminer la taille cible
    if isinstance(model, FaceNet):
        target_size = model.model.input_shape[1:3]  # (160, 160) pour FaceNet
    else:
        target_size = model.input_shape[1:3]        # (224, 224) pour VGG16

    # Redimensionner l'image
    # Ensure target size is valid
    # Check if target_size contains None and replace with 1 if necessary
    target_size = tuple(dim if dim is not None else 1 for dim in target_size)
    target_size = (max(1, target_size[1]), max(1, target_size[0]))
    face_image = cv2.resize(face_image, (target_size[1], target_size[0]))

    # Prétraitement spécifique au modèle
    if isinstance(model, FaceNet):
        # Conversion BGR → RGB et normalisation [-1, 1]
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_image = (face_image.astype('float32') - 127.5) / 127.5
    else:
        # Conversion BGR → BGR (déjà en BGR via OpenCV) et soustraction moyennes ImageNet
        if face_image.max() <= 1.0:
            face_image = face_image * 255.0  # Si normalisé [0,1] → repasser à [0,255]
        face_image[..., 0] -= 103.939  # B
        face_image[..., 1] -= 116.779  # G
        face_image[..., 2] -= 123.68   # R

    # Ajouter une dimension de batch et prédire
    face_image = np.expand_dims(face_image, axis=0)
    if isinstance(model, FaceNet):
        embedding = model.embeddings(face_image)
    else:
        embedding = model.predict(face_image)

    return embedding.flatten()


# 1. Charger les modèles une fois
@st.cache_resource
def load_models():
    # Charger FaceNet
    embedder = FaceNet()

    # Charger le classificateur SVM
    clf = joblib.load('actor_classifier.pkl')

    # Charger les noms de classes
    class_names = np.load('class_names.npy', allow_pickle=True)
    return embedder, clf, class_names


embedder, clf, class_names = load_models()

# 2. Interface Streamlit
st.title("🎬 Reconnaissance d'Acteurs")
st.markdown("Téléchargez une photo d'un acteur pour l'identifier")

# 3. Upload d'image
uploaded_file = st.file_uploader("Choisir une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Lire et afficher l'image
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)

    # Convertir en BGR pour OpenCV
    face_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Détection et extraction du visage
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) > 0:
        # Prendre le premier visage détecté
        x, y, w, h = faces[0]
        face = face_image[y:y + h, x:x + w]

        # Générer l'embedding
        embedding = get_embedding(embedder, face)

        # Prédiction
        proba = clf.predict([embedding])[0]
        confidence = np.max(proba)
        predicted_class = class_names[np.argmax(proba)]

        # Affichage
        st.image(image, caption='Image téléchargée', width=300)
        st.success(f"**Acteur identifié :** {predicted_class}")
        st.info(f"**Confiance :** {confidence * 100:.2f}%")
    else:
        st.error("Aucun visage détecté dans l'image !")