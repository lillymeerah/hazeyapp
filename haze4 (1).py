import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import urllib.request

# ==========================
# DOWNLOAD FACE DETECTION MODELS IF NOT EXIST
# ==========================
MODEL_DIR = "models"
PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
MODEL_URL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

PROTO_PATH = os.path.join(MODEL_DIR, r"C:\Users\bibiz\Downloads\input data.txt")
MODEL_PATH = os.path.join(MODEL_DIR, r"C:\Users\bibiz\Downloads\res10_300x300_ssd_iter_140000 (1).caffemodel")

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(PROTO_PATH):
    urllib.request.urlretrieve(PROTO_URL, PROTO_PATH)

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# Load face detection model
face_net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)


# ==========================
# HAZE DETECTION FUNCTION
# ==========================
def detect_haze(image):
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    dark_channel = np.min(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), axis=2)
    haze_level = np.mean(dark_channel)

    if haze_level > 100:
        return "Clear"
    elif 60 < haze_level <= 100:
        return "Medium Haze"
    else:
        return "Heavy Haze"


# ==========================
# FACE DETECTION FUNCTION
# ==========================
def detect_and_blur_faces(image):
    img = np.array(image.convert("RGB"))
    (h, w) = img.shape[:2]

    # Prepare blob for DNN
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Clip to image bounds
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w, endX), min(h, endY)

            # Apply Gaussian blur
            face_region = img[startY:endY, startX:endX]
            if face_region.size > 0:
                face_region = cv2.GaussianBlur(face_region, (99, 99), 30)
                img[startY:endY, startX:endX] = face_region

    return Image.fromarray(img)


# ==========================
# ACCOUNT SYSTEM
# ==========================
ACCOUNTS_FILE = "accounts.txt"

def load_accounts():
    if not os.path.exists(ACCOUNTS_FILE):
        return {}
    accounts = {}
    with open(ACCOUNTS_FILE, "r") as f:
        for line in f:
            if ":" in line:
                username, password = line.strip().split(":", 1)
                accounts[username] = password
    return accounts

def save_account(username, password):
    with open(ACCOUNTS_FILE, "a") as f:
        f.write(f"{username}:{password}\n")

# ==========================
# STREAMLIT APP
# ==========================
def main():
    st.title("🌫️ Haze + Face Detection App")

    menu = ["Login", "Create Account"]
    choice = st.sidebar.selectbox("Menu", menu)

    accounts = load_accounts()

    if choice == "Create Account":
        st.subheader("Create Account")
        new_user = st.text_input("Username")
        new_pass = st.text_input("Password", type="password")
        if st.button("Create"):
            if new_user in accounts:
                st.error("Username already exists!")
            else:
                save_account(new_user, new_pass)
                st.success("Account created successfully!")

    elif choice == "Login":
        st.subheader("Login Section")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username in accounts and accounts[username] == password:
                st.success(f"Welcome {username}!")

                # Upload Image
                uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
                if uploaded_file is not None:
                    image = Image.open(uploaded_file)

                    st.image(image, caption="Original Image", use_column_width=True)

                    haze_result = detect_haze(image)
                    st.write("☁️ Haze Detection Result:", haze_result)

                    blurred_image = detect_and_blur_faces(image)
                    st.image(blurred_image, caption="Processed Image (Faces Blurred)", use_column_width=True)

            else:
                st.error("Invalid Username or Password")


if __name__ == "__main__":
    main()
