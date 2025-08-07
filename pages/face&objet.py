import cv2
import numpy as np
import streamlit as st
from PIL import Image
import io

# Fonction pour charger les noms des classes
def load_classes(file):
    with open(file, "r") as f:
        return [line.strip() for line in f.readlines()]

# Fonction pour charger le modèle YOLOv4
def load_yolo_model(cfg_file, weights_file):
    return cv2.dnn.readNet(cfg_file, weights_file)

# Fonction pour effectuer la détection d'objets
def detect_objects(image, net, output_layers):
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return boxes, confidences, class_ids

# Fonction pour dessiner les boîtes englobantes
def draw_labels(image, boxes, confidences, class_ids, classes):
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

# Fonction de détection avec webcam dans Streamlit
def detect_objects_from_webcam(net, output_layers, classes):
    stframe = st.empty()  # Conteneur pour afficher les frames
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Impossible d'accéder à la webcam.")
        return

    stop_button = st.sidebar.button("⛔ Arrêter la webcam")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Erreur lors de la lecture de la webcam.")
            break

        boxes, confidences, class_ids = detect_objects(frame, net, output_layers)
        frame_with_boxes = draw_labels(frame.copy(), boxes, confidences, class_ids, classes)

        # Convertir en RGB
        frame_rgb = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", caption="Détection en direct")

        # Vérifier si le bouton d'arrêt est cliqué
        if stop_button:
            break

    cap.release()
    st.success("🛑 Webcam arrêtée.")

# --------------------- Interface Streamlit ---------------------
st.set_page_config(page_title="YOLOv4 Object Detection", layout="wide")
st.title("🕵️‍♀️ Détection d'Objets avec YOLOv4")

# Fichiers YOLO
cfg_file = "yolov4.cfg"
weights_file = "yolov4.weights"
names_file = "coco .names"  # Vérifie que ce fichier ne contient pas d'espace dans le nom

# Chargement du modèle et des classes
try:
    net = load_yolo_model(cfg_file, weights_file)
    classes = load_classes(names_file)
    output_layers = net.getUnconnectedOutLayersNames()
except Exception as e:
    st.error(f"Erreur de chargement du modèle YOLO : {e}")
    st.stop()

# Sidebar - chargement d'image
st.sidebar.header("📂 Options")
uploaded_file = st.sidebar.file_uploader("Choisir une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = np.array(image.convert('RGB'))

    boxes, confidences, class_ids = detect_objects(img, net, output_layers)
    img_with_boxes = draw_labels(img.copy(), boxes, confidences, class_ids, classes)

    st.image(image, caption="Image originale", use_column_width=True)
    st.image(img_with_boxes, caption="Image annotée", use_column_width=True)

    # Téléchargement
    buffer = io.BytesIO()
    Image.fromarray(img_with_boxes).save(buffer, format="PNG")
    buffer.seek(0)
    st.sidebar.download_button(
        label="📥 Télécharger l'image annotée",
        data=buffer,
        file_name="detection_result.png",
        mime="image/png"
    )

# Bouton Webcam
if st.sidebar.button("🎥 Activer la webcam"):
    st.info("Appuyez sur '⛔ Arrêter la webcam' dans la barre latérale pour arrêter.")
    detect_objects_from_webcam(net, output_layers, classes)

# Footer
st.sidebar.markdown("---")
st.sidebar.text("© 2024 Mamadou MBOW - Deep Learning")
