import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import joblib
import os
import time
import threading
import pyttsx3

from collections import Counter

# ============================================================
# MEDIAPIPE
# ============================================================

import mediapipe as mp

from mediapipe.tasks.python.core.base_options import BaseOptions

from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="KSL Transformer Interpreter",
    page_icon="🤟",
    layout="wide"
)

# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

body {
    background-color: #0E1117;
}

.main-title {
    font-size:45px;
    font-weight:bold;
    color:#00FFAA;
}

.prediction-box {
    background:#1E1E1E;
    padding:20px;
    border-radius:15px;
    text-align:center;
}

.prediction-text {
    font-size:35px;
    font-weight:bold;
    color:#00FFAA;
}

.confidence-text {
    font-size:22px;
    color:white;
}

.stButton>button {
    width:100%;
    height:50px;
    border-radius:10px;
    font-size:18px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# CPU OPTIMIZATION
# ============================================================

torch.set_num_threads(4)

DEVICE = torch.device("cpu")

# ============================================================
# TRANSFORMER MODEL
# ============================================================

class KSL_Transformer(nn.Module):

    def __init__(self, input_size, num_classes):

        super().__init__()

        self.embedding = nn.Linear(input_size, 128)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.2,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=3
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        x = self.embedding(x)

        x = self.transformer(x)

        x = x.mean(dim=1)

        return self.fc(x)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    config = joblib.load("model_config.pkl")

    model = KSL_Transformer(
        input_size=config["input_size"],
        num_classes=config["num_classes"]
    )

    model.load_state_dict(
        torch.load(
            "ksl_transformer_cpu.pth",
            map_location=DEVICE
        )
    )

    model.eval()

    encoder = joblib.load("label_encoder.pkl")

    mean = np.load("mean.npy")

    std = np.load("std.npy")

    return model, encoder, mean, std, config

model, encoder, mean, std, config = load_model()

SEQUENCE_LENGTH = config["sequence_length"]

# ============================================================
# TEXT TO SPEECH
# ============================================================

def run_speech(text):

    engine = pyttsx3.init()

    engine.say(text)

    engine.runAndWait()

    engine.stop()

def speak_text(text):

    thread = threading.Thread(
        target=run_speech,
        args=(text,)
    )

    thread.start()

# ============================================================
# LOAD HAND LANDMARKER
# ============================================================

@st.cache_resource
def load_detector():

    options = HandLandmarkerOptions(

        base_options=BaseOptions(
            model_asset_path="hand_landmarker.task"
        ),

        running_mode=RunningMode.IMAGE,

        num_hands=2
    )

    detector = HandLandmarker.create_from_options(
        options
    )

    return detector

detector = load_detector()

# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_keypoints(frame):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    features = []

    if result.hand_landmarks:

        for hand_landmarks in result.hand_landmarks:

            wrist_x = hand_landmarks[0].x
            wrist_y = hand_landmarks[0].y
            wrist_z = hand_landmarks[0].z

            for landmark in hand_landmarks:

                features.extend([
                    landmark.x - wrist_x,
                    landmark.y - wrist_y,
                    landmark.z - wrist_z
                ])

        # DRAW LANDMARKS
        for hand_landmarks in result.hand_landmarks:

            for landmark in hand_landmarks:

                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (0, 255, 0),
                    -1
                )

    while len(features) < 126:
        features.append(0.0)

    return np.array(features[:126], dtype=np.float32), frame

# ============================================================
# VIDEO FINDER
# ============================================================

VIDEO_DIR = "videos"

def find_sign_video(word):

    class_folder = os.path.join(
        VIDEO_DIR,
        word.lower()
    )

    if not os.path.exists(class_folder):
        return None

    videos = [
        f for f in os.listdir(class_folder)
        if f.endswith(".mp4")
    ]

    if len(videos) == 0:
        return None

    return os.path.join(
        class_folder,
        videos[0]
    )

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Navigation")

mode = st.sidebar.radio(
    "Select Mode",
    [
        "Realtime Sign Detection",
        "Text To Sign",
        "About"
    ]
)

# ============================================================
# TITLE
# ============================================================

st.markdown("""
<div class='main-title'>
🤟 Kenyan Sign Language Transformer Interpreter
</div>
""", unsafe_allow_html=True)

# ============================================================
# REALTIME DETECTION
# ============================================================

if mode == "Realtime Sign Detection":

    st.subheader("📷 Live Realtime Detection")

    start = st.checkbox("Start Camera")

    frame_placeholder = st.empty()

    prediction_placeholder = st.empty()

    sentence_placeholder = st.empty()

    sentence = []

    prediction_history = []

    sequence_buffer = []

    last_prediction_time = time.time()

    if start:

        cap = cv2.VideoCapture(0)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:

            ret, frame = cap.read()

            if not ret:
                st.error("Cannot access camera")
                break

            frame = cv2.flip(frame, 1)

            frame = cv2.resize(frame, (224, 224))

            # ====================================================
            # FEATURE EXTRACTION
            # ====================================================

            features, frame = extract_keypoints(frame)

            sequence_buffer.append(features)

            if len(sequence_buffer) > SEQUENCE_LENGTH:
                sequence_buffer.pop(0)

            # WAIT FOR FULL SEQUENCE
            if len(sequence_buffer) < SEQUENCE_LENGTH:

                frame_placeholder.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    channels="RGB"
                )

                continue

            # ====================================================
            # CREATE INPUT
            # ====================================================

            sequence = np.array(
                sequence_buffer,
                dtype=np.float32
            )

            # ====================================================
            # MOTION FEATURES
            # ====================================================

            velocity = np.diff(sequence, axis=0)

            velocity = np.vstack([
                velocity,
                velocity[-1]
            ])

            acceleration = np.diff(
                velocity,
                axis=0
            )

            acceleration = np.vstack([
                acceleration,
                acceleration[-1]
            ])

            final_input = np.concatenate([
                sequence,
                velocity,
                acceleration
            ], axis=1)

            # ====================================================
            # NORMALIZATION
            # ====================================================

            final_input = (
                final_input - mean
            ) / std

            final_input = torch.tensor(
                final_input,
                dtype=torch.float32
            ).unsqueeze(0)

            final_input = final_input.to(DEVICE)

            # ====================================================
            # PREDICTION
            # ====================================================

            with torch.no_grad():

                output = model(final_input)

                probs = torch.softmax(
                    output,
                    dim=1
                )

                confidence, pred = torch.max(
                    probs,
                    dim=1
                )

                confidence = confidence.item() * 100

                pred = pred.item()

                label = encoder.inverse_transform(
                    [pred]
                )[0]

            # ====================================================
            # SMOOTHING
            # ====================================================

            prediction_history.append(label)

            if len(prediction_history) > 10:
                prediction_history.pop(0)

            stable_prediction = Counter(
                prediction_history
            ).most_common(1)[0][0]

            # ====================================================
            # BUILD SENTENCE
            # ====================================================

            if (
                confidence > 80
                and
                time.time() - last_prediction_time > 2
            ):

                if (
                    len(sentence) == 0
                    or
                    sentence[-1] != stable_prediction
                ):

                    sentence.append(
                        stable_prediction
                    )

                    speak_text(
                        stable_prediction
                    )

                last_prediction_time = time.time()

            # ====================================================
            # DRAW PREDICTION
            # ====================================================

            cv2.rectangle(
                frame,
                (0, 0),
                (224, 80),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                frame,
                stable_prediction,
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"{confidence:.2f}%",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            # ====================================================
            # STREAMLIT UI
            # ====================================================

            prediction_placeholder.markdown(
                f"""
                <div class='prediction-box'>

                    <div class='prediction-text'>
                        🤟 {stable_prediction}
                    </div>

                    <div class='confidence-text'>
                        Confidence: {confidence:.2f}%
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            sentence_placeholder.markdown(
                f"## 📝 Sentence: {' '.join(sentence)}"
            )

            frame_placeholder.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                channels="RGB"
            )

        cap.release()

    col1, col2 = st.columns(2)

    with col1:

        if st.button("🔊 Speak Sentence"):

            text = " ".join(sentence)

            if text.strip():
                speak_text(text)

    with col2:

        if st.button("🗑 Clear Sentence"):

            sentence.clear()

# ============================================================
# TEXT TO SIGN
# ============================================================

elif mode == "Text To Sign":

    st.subheader("🔤 Text To Sign")

    user_text = st.text_input(
        "Enter Text",
        placeholder="Example: hello thank you"
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button("▶ Translate To Sign"):

            if user_text.strip() == "":

                st.warning("Please enter text")

            else:

                words = user_text.lower().split()

                for word in words:

                    st.markdown(f"## 🤟 {word}")

                    video_path = find_sign_video(word)

                    if video_path:

                        st.video(video_path)

                    else:

                        st.error(
                            f"No sign video found for '{word}'"
                        )

    with col2:

        if st.button("🔊 Speak Text"):

            if user_text.strip():

                speak_text(user_text)

# ============================================================
# ABOUT
# ============================================================

else:

    st.subheader("📘 About System")

    st.markdown("""



""")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    "<center>🤟 Kenyan Sign Language Interpreter</center>",
    unsafe_allow_html=True
)