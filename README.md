# 🤟 Kenyan Sign Language Detection Application

A real-time Kenyan Sign Language (KSL) detection and translation system powered by Machine Learning, OpenCV, and MediaPipe Hand Landmarker.

The application detects Kenyan Sign Language gestures from a webcam feed, translates them into text in real time, converts text into speech, and supports text-to-sign translation using pre-recorded sign videos.

## 🚀 Features

* ✅ Real-time KSL gesture detection
* ✅ Live webcam prediction
* ✅ Sign-to-text translation
* ✅ Text-to-speech conversion
* ✅ Text-to-sign video translation
* ✅ MediaPipe Hand Landmarker integration
* ✅ Machine Learning gesture classification
* ✅ streamlit web application
* ✅ CPU optimized for low-powered laptops

---

## 🧠 Technologies Used

* Python 3.10+
* streamlit
* OpenCV
* MediaPipe Tasks API
* Scikit-learn
* XGBoost
* NumPy
* pyttsx3

Research and implementation were inspired by real-time sign language detection systems and pose-based gesture recognition pipelines. ([arXiv][1])

---

# 📂 Project Structure

```bash
kenya-sign-language-detection-app-application/
│
├── app.py
├── videos/
│       ├── agreement/
│       │   └── sign1.mp4
│       ├── good/
│       └── bad/
│
├── model_config.pkl
├── label_encoder.pkl
├── mean.pkl
├── std.pkl
├── hand_landmarker.task
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation Guide

## 1️⃣ Clone Repository

```bash
git clone https://github.com/JamesMungai254/kenya-sign-language-detection-app-application.git
```

Or visit:

[Kenya Sign Language Detection Application Repository](https://github.com/JamesMungai254/kenya-sign-language-detection-app-application.git?utm_source=chatgpt.com)

---

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing:

```bash
pip install streamlit
pip install opencv-python
pip install mediapipe
pip install numpy
pip install scikit-learn
pip install xgboost
pip install pyttsx3
pip install pillow
```

---

# 📥 Download MediaPipe Hand Landmarker

Download the MediaPipe Hand Landmarker task model:

[MediaPipe Hand Landmarker Task Model](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task?utm_source=chatgpt.com)

Place the file inside the project root:

```bash
hand_landmarker.task
```

---

# ▶️ Running the Application

## Start streamlit Server

```bash
streamlit run app.py
```

Open browser:

```bash
http://localhost:8501/
```

---

# 🤟 Realtime Detection

The application:

* Opens webcam feed
* Detects hand landmarks
* Extracts gesture features
* Predicts KSL sign
* Converts prediction to text
* Builds sentence in real time
* Supports speech output

---

# 🔤 Text-to-Sign Translation

Text entered by the user is mapped to sign videos stored in:

```bash
media/video/<class_name>/
```

Example:

```bash
media/video/agreement/sign1.mp4
```

Typing:

```bash
agreement
```

will display the corresponding sign video.

---

# 🔊 Text-to-Speech

The application uses:

```python
pyttsx3
```

for offline speech synthesis.

No internet connection is required.

---

# 📊 Model Information

The system uses:

* MediaPipe Hand Landmarker
* Landmark feature extraction
* XGBoost classifier
* `model_config.pkl`
* `label_encoder.pkl`

The model was optimized for:

* CPU inference
* Low latency
* Lightweight deployment
* Real-time webcam prediction



# 📈 Future Improvements

* Sentence-level translation
* Transformer-based sequence models
* Mobile deployment
* Swahili translation
* Cloud deployment
* Sign language chatbot
* Gesture confidence visualization

---

# 📚 References

* MediaPipe Hand Landmarker Documentation
  [MediaPipe Tasks Vision Guide](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python?utm_source=chatgpt.com)

* Real-Time Sign Language Detection using Human Pose Estimation
  ([arXiv][1])

* GitHub Sign Language Detection Systems
  ([GitHub][2])

---

# 👨‍💻 Author

## James Mungai

* Data Science & Analytics Graduate
* Machine Learning Developer
* Computer Vision Enthusiast

GitHub:

[JamesMungai254 GitHub Profile](https://github.com/JamesMungai254?utm_source=chatgpt.com)

---

# ⭐ Support the Project

If you found this project useful:

* Star the repository
* Fork the project
* Contribute improvements


---

# 📜 License

This project is for academic and research purposes.

[1]: https://arxiv.org/abs/2008.04637?utm_source=chatgpt.com "Real-Time Sign Language Detection using Human Pose Estimation"
[2]: https://github.com/topics/sign-language-recognition-system?utm_source=chatgpt.com "sign-language-recognition-system · GitHub Topics · GitHub"
