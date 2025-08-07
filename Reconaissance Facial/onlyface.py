import streamlit as st
import cv2
import os
import time

# Titre principal de l'application
st.header(":blue[Présentation de détection de visages]")
st.markdown("""
* **Présentation** : Cette application illustre les algorithmes utilisés pour la détection de visages en temps réel.
""")

# Sidebar pour la navigation
menu = st.sidebar.selectbox("Menu", ["Tableau de bord", "Détection de visages"])

if menu == "Tableau de bord":
    st.write(":green[Bienvenue sur l'application de détection de visages et d'objets.]")

elif menu == "Détection de visages":
    # Charger le modèle de cascade pour la détection de visages
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Fonction pour détecter des visages et afficher le flux de la webcam
    def detect_faces_in_window():
        cap = cv2.VideoCapture(0)  # Ouvrir la webcam
        captured_face = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Erreur de capture de la webcam")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                captured_face = frame[y:y+h, x:x+w]  # Capture le visage détecté

            # Afficher l'image dans une fenêtre
            cv2.imshow("Détection de Visages", frame)

            # Attendre que l'utilisateur appuie sur 'q' pour quitter ou 'c' pour capturer le visage
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') and captured_face is not None:
                # Enregistrer le visage capturé
                if not os.path.exists("captured_faces"):
                    os.makedirs("captured_faces")

                file_name = f"captured_faces/face_{int(time.time())}.png"
                cv2.imwrite(file_name, captured_face)
                print(f"Visage capturé et enregistré sous {file_name}")
                st.success(f"Visage capturé et enregistré sous {file_name}")

                # Analyse du visage capturé (simple conversion en niveaux de gris)
                analyze_face(captured_face)

        cap.release()
        cv2.destroyAllWindows()

    # Fonction pour analyser le visage capturé
    def analyze_face(face_image):
        gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Analyse du Visage (Grayscale)", gray_face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    st.title("Détection de Visages en Temps Réel")
    st.write("Cliquez sur le bouton ci-dessous pour ouvrir la webcam dans une nouvelle fenêtre.")

    # Bouton pour démarrer la détection de visages
    if st.button("Ouvrir la Webcam"):
        detect_faces_in_window()

# Footer
st.sidebar.text("© 2024 Mamadou MBOW - Machine Learning && Deep Learning")
