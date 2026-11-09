
import streamlit as st
import os
import cv2
import av
import joblib
import numpy as np
import mediapipe as mp
import requests
import html
import time
import threading

from dotenv import load_dotenv
from collections import deque, Counter
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    VideoProcessorBase
)
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from gnani.tts import GnaniTTSClient, AudioConfig


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="SignBridge AI",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="collapsed"
)

load_dotenv()

API_KEY = os.getenv("GNANI_API_KEY")

STT_URL = "https://api.vachana.ai/stt/v3"

MODEL_PATH = "signbridge_model.pkl"
TASK_PATH = "hand_landmarker.task"

LANGUAGES = {
    "Tamil": "ta-IN",
    "English": "en-IN",
    "Hindi": "hi-IN",
    "Telugu": "te-IN",
    "Malayalam": "ml-IN"
}


# =====================================================
# 2. SESSION STATE
# =====================================================

defaults = {
    "selected_feature": "Speech to Text",
    "recording_version": 0,
    "upload_version": 0,
    "recognized_text": "",
    "generated_audio": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =====================================================
# 3. PREMIUM UI CSS
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

.stApp {
    background: #F4F7F8;
    color: #17212B;
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    max-width: 1250px;
    padding: 2.5rem 2rem;
}

h1, h2, h3, h4 {
    font-family: 'Manrope', sans-serif !important;
    color: #17212B !important;
    font-weight: 700 !important;
}

p, label, li {
    color: #475569;
}

.main-title {
    font-family: 'Manrope', sans-serif;
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -1.5px;
    color: #17212B;
    text-align: center;
}

.subtitle {
    text-align: center;
    font-size: 15px;
    color: #64748B;
    margin-bottom: 30px;
}

.hero-box {
    background: #17212B;
    border-radius: 10px;
    padding: 35px;
    margin: 20px 0 35px;
    border-left: 4px solid #0F766E;
}

.hero-box h3 {
    color: #FFFFFF !important;
    font-size: 26px;
}

.hero-box p {
    color: #CBD5E1 !important;
    font-size: 15px;
}

.section-heading {
    font-family: 'Manrope', sans-serif;
    color: #17212B;
    font-size: 21px;
    font-weight: 750;
    margin-bottom: 5px;
}

.section-description {
    color: #64748B;
    font-size: 14px;
    margin-bottom: 20px;
}

/* BUTTONS */

div.stButton > button {
    width: 100%;
    min-height: 52px;
    background: #FFFFFF;
    color: #334155 !important;
    border: 1px solid #DCE3E8;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    font-weight: 600;
    transition: all 0.2s ease;
}

div.stButton > button p {
    color: inherit !important;
}

div.stButton > button:hover {
    background: #EDF5F4 !important;
    color: #0F766E !important;
    border-color: #0F766E !important;
}

div.stButton > button[kind="primary"] {
    background: #0F766E !important;
    color: #FFFFFF !important;
    border-color: #0F766E !important;
}

div.stButton > button[kind="primary"] p {
    color: #FFFFFF !important;
}

/* INPUTS */

.stTextInput input,
.stTextArea textarea {
    background: #FFFFFF !important;
    color: #17212B !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
}

.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: #0F766E !important;
    box-shadow: 0 0 0 1px #0F766E !important;
}

.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #94A3B8 !important;
}

/* SELECTBOX */

.stSelectbox div[data-baseweb="select"] {
    background: #FFFFFF !important;
    border-radius: 8px;
}

.stSelectbox div[data-baseweb="select"] * {
    color: #17212B !important;
}

/* RADIO */

.stRadio label,
.stRadio label p {
    color: #334155 !important;
}

/* FILE UPLOADER */

[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 1px dashed #94A3B8;
    border-radius: 8px;
    padding: 15px;
}

[data-testid="stFileUploader"] * {
    color: #334155 !important;
}

/* WORKSPACE */

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
}

/* RESULT CARD */

.result-card {
    background: #F0F7F6;
    border: 1px solid #D5E9E6;
    border-radius: 10px;
    padding: 30px 20px;
    text-align: center;
    margin: 18px 0;
    min-height: 150px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.result-label {
    color: #0F766E;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

.result-value {
    color: #17212B;
    font-family: 'Manrope', sans-serif;
    font-size: 32px;
    font-weight: 800;
    margin-top: 12px;
    overflow-wrap: anywhere;
}

/* INFORMATION CARDS */

.info-card {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 9px;
    padding: 20px;
    min-height: 120px;
}

.info-card h4 {
    color: #0F766E !important;
    font-size: 15px;
}

.info-card p {
    color: #64748B !important;
    font-size: 13px;
}

/* METRICS */

[data-testid="stMetric"] {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    padding: 18px;
    border-radius: 9px;
    min-height: 110px;
}

[data-testid="stMetricLabel"] {
    color: #475569 !important;
}

[data-testid="stMetricValue"] {
    color: #17212B !important;
    overflow-wrap: anywhere;
}

/* AUDIO */

audio {
    width: 100%;
}

/* DOWNLOAD */

.stDownloadButton button {
    width: 100%;
    min-height: 45px;
    background: #0F766E !important;
    color: #FFFFFF !important;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}

.stDownloadButton button p {
    color: #FFFFFF !important;
}

/* FOOTER */

.footer {
    text-align: center;
    color: #64748B !important;
    padding: 25px;
    font-size: 13px;
}

hr {
    border-color: #E2E8F0 !important;
}

/* RESPONSIVE */

@media(max-width: 768px) {
    .main-title {
        font-size: 30px;
    }

    .block-container {
        padding: 1rem;
    }

    .hero-box {
        padding: 22px;
    }

    .result-value {
        font-size: 24px;
    }
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 4. HEADER
# =====================================================

st.markdown("""
<div class="main-title">SignBridge AI</div>
<div class="subtitle">
Intelligent Communication • Accessible Technology
</div>
<div class="hero-box">
<h3>Communication Without Barriers.</h3>
<p>
An AI-powered communication platform that connects
speech, written language and sign language.
</p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# 5. FEATURE NAVIGATION
# =====================================================

st.markdown(
    '<div class="section-heading">Communication Workspace</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Select a communication service to continue.'
    '</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    if st.button(
        "Speech to Text",
        key="speech_button",
        type="primary" if st.session_state.selected_feature == "Speech to Text" else "secondary"
    ):
        st.session_state.selected_feature = "Speech to Text"
        st.rerun()

with col2:
    if st.button(
        "Text to Speech",
        key="text_button",
        type="primary" if st.session_state.selected_feature == "Text to Speech" else "secondary"
    ):
        st.session_state.selected_feature = "Text to Speech"
        st.rerun()

with col3:
    if st.button(
        "Sign Language Recognition",
        key="sign_button",
        type="primary" if st.session_state.selected_feature == "Sign Language to Text" else "secondary"
    ):
        st.session_state.selected_feature = "Sign Language to Text"
        st.rerun()

st.divider()

selected = st.session_state.selected_feature


# =====================================================
# 6. SPEECH TO TEXT
# =====================================================

if selected == "Speech to Text":

    with st.container(border=True):

        st.markdown("## Speech to Text")

        st.write(
            "Convert spoken words into written text using AI-powered speech recognition."
        )

        st.divider()

        st.markdown("### Language Settings")

        language = st.selectbox(
            "Select the language spoken in your audio",
            list(LANGUAGES.keys()),
            key="stt_language"
        )

        st.divider()

        st.markdown("### Audio Input")

        audio_source = st.radio(
            "Choose your audio source",
            ["Record Audio", "Upload Audio"],
            horizontal=True,
            key="audio_source"
        )

        audio_file = None

        if audio_source == "Record Audio":

            st.info("Use the recording control below to record your voice.")

            audio_file = st.audio_input(
                "Record your voice",
                key=f"recording_{st.session_state.recording_version}"
            )

        else:

            st.info("Upload an existing audio file from your device.")

            audio_file = st.file_uploader(
                "Choose audio file",
                type=["wav", "mp3", "m4a", "ogg", "flac", "aac"],
                key=f"uploaded_audio_{st.session_state.upload_version}"
            )

        if audio_file is not None:

            st.markdown("### Audio Preview")
            st.audio(audio_file)

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            convert = st.button(
                "Convert to Text",
                type="primary",
                key="convert_stt"
            )

        with col2:
            clear = st.button(
                "Clear Audio",
                key="clear_recording"
            )

        if clear:

            st.session_state.recording_version += 1
            st.session_state.upload_version += 1
            st.session_state.recognized_text = ""

            st.rerun()

        if convert:

            if not API_KEY:

                st.error("API key missing. Please check your .env file.")

            elif audio_file is None:

                st.warning("Please record or upload an audio file first.")

            else:

                with st.spinner("Processing audio..."):

                    try:

                        headers = {
                            "X-API-Key-ID": API_KEY
                        }

                        files = {
                            "audio_file": (
                                getattr(audio_file, "name", "recording.wav"),
                                audio_file.getvalue(),
                                audio_file.type or "audio/wav"
                            )
                        }

                        data = {
                            "model": "prisma-v3.0",
                            "language_code": LANGUAGES[language]
                        }

                        response = requests.post(
                            STT_URL,
                            headers=headers,
                            files=files,
                            data=data,
                            timeout=120
                        )

                        if response.ok:

                            result = response.json()

                            recognized_text = (
                                result.get("transcript")
                                or result.get("text")
                                or ""
                            )

                            if recognized_text:

                                st.session_state.recognized_text = recognized_text

                                st.success("Speech converted successfully!")

                            else:

                                st.warning("No transcription was found.")
                                st.json(result)

                        else:

                            st.error(f"API Error: {response.status_code}")
                            st.code(response.text)

                    except requests.RequestException as e:

                        st.error(f"Connection Error: {e}")

                    except ValueError:

                        st.error("Invalid API response.")

                    except Exception as e:

                        st.error(f"Error: {e}")

        if st.session_state.recognized_text:

            st.divider()

            st.markdown("### Transcription Result")

            st.text_area(
                "Recognized Text",
                value=st.session_state.recognized_text,
                height=170,
                key="stt_result_display"
            )

            st.download_button(
                "Download Recognized Text",
                data=st.session_state.recognized_text,
                file_name="recognized_text.txt",
                mime="text/plain",
                key="download_stt"
            )


# =====================================================
# 7. TEXT TO SPEECH
# =====================================================

elif selected == "Text to Speech":

    with st.container(border=True):

        st.markdown("## Text to Speech")

        st.write(
            "Transform written messages into natural AI-generated speech."
        )

        st.divider()

        st.markdown("### Language Settings")

        language = st.selectbox(
            "Select speech language",
            list(LANGUAGES.keys()),
            key="tts_language"
        )

        st.divider()

        st.markdown("### Text Input")

        text_input = st.text_area(
            "Enter your message",
            placeholder="Type your message here...",
            height=190,
            key="tts_text"
        )

        if text_input:
            st.caption(f"Characters entered: {len(text_input)}")

        st.divider()

        if st.button(
            "Generate Speech",
            type="primary",
            key="generate_tts"
        ):

            if not API_KEY:

                st.error("API key missing. Please check your .env file.")

            elif not text_input.strip():

                st.warning("Please enter some text first.")

            else:

                try:

                    with st.spinner("Generating speech..."):

                        client = GnaniTTSClient(api_key=API_KEY)

                        audio_bytes = client.synthesize(
                            text_input.strip(),
                            voice="Nalini",
                            model="timbre-v2.5",
                            language=LANGUAGES[language],
                            audio_config=AudioConfig(
                                sample_rate=48000,
                                encoding="linear_pcm",
                                container="wav"
                            )
                        )

                        if audio_bytes and len(audio_bytes) > 0:

                            st.session_state.generated_audio = audio_bytes

                            st.success("Speech generated successfully!")

                        else:

                            st.error("The API returned empty audio.")

                except Exception as e:

                    st.error(f"Speech generation failed: {e}")

        if st.session_state.generated_audio:

            st.divider()

            st.markdown("### Generated Audio")

            st.success("Your audio is ready.")

            st.audio(
                st.session_state.generated_audio,
                format="audio/wav"
            )

            st.download_button(
                "Download Audio",
                data=st.session_state.generated_audio,
                file_name="signbridge_audio.wav",
                mime="audio/wav",
                key="download_tts"
            )


# =====================================================
# 8. SIGN LANGUAGE RECOGNITION
# =====================================================

elif selected == "Sign Language to Text":

    with st.container(border=True):

        st.markdown("## Sign Language Recognition")

        st.write(
            "Recognize hand gestures through your webcam using your trained machine learning model."
        )

        st.divider()

        if not os.path.exists(MODEL_PATH):

            st.error(f"Model file not found: {MODEL_PATH}")

        elif not os.path.exists(TASK_PATH):

            st.error(f"MediaPipe task file not found: {TASK_PATH}")

        else:

            # -----------------------------------------
            # FEATURE EXTRACTION
            # -----------------------------------------

            def extract_features(frames):

                normalized_frames = []

                for frame in frames:

                    normalized = frame.copy()

                    for hand in range(2):

                        points = normalized[hand]

                        if np.any(points):

                            wrist = points[0].copy()

                            points -= wrist

                            scale = np.max(
                                np.linalg.norm(
                                    points[:, :2],
                                    axis=1
                                )
                            )

                            if scale > 0:
                                points /= scale

                    normalized_frames.append(
                        normalized.flatten()
                    )

                features = np.array(normalized_frames)

                return np.concatenate([
                    np.mean(features, axis=0),
                    np.std(features, axis=0)
                ]).reshape(1, -1)

            # -----------------------------------------
            # LOAD MODEL
            # -----------------------------------------

            @st.cache_resource
            def load_model():
                return joblib.load(MODEL_PATH)

            # -----------------------------------------
            # MEDIAPIPE LANDMARKER
            # -----------------------------------------

            def create_landmarker():

                base_options = python.BaseOptions(
                    model_asset_path=TASK_PATH
                )

                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.VIDEO,
                    num_hands=2
                )

                return vision.HandLandmarker.create_from_options(options)

            # -----------------------------------------
            # CAMERA PROCESSOR
            # -----------------------------------------

            class SignProcessor(VideoProcessorBase):

                def __init__(self):

                    self.model = load_model()
                    self.landmarker = create_landmarker()

                    self.frame_buffer = deque(maxlen=30)

                    # Recent raw model predictions
                    self.prediction_buffer = deque(maxlen=12)

                    self.frame_count = 0

                    self.lock = threading.Lock()

                    self.prediction = "Show a sign"
                    self.confidence = 0.0
                    self.status = "Waiting for sign"

                    self.error_message = None

                    self.last_candidate = None
                    self.candidate_streak = 0

                def recv(self, frame):

                    img = frame.to_ndarray(format="bgr24")

                    img = cv2.flip(img, 1)

                    rgb = cv2.cvtColor(
                        img,
                        cv2.COLOR_BGR2RGB
                    )

                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb
                    )

                    try:

                        with self.lock:

                            timestamp = self.frame_count * 33
                            self.frame_count += 1

                            result = self.landmarker.detect_for_video(
                                mp_image,
                                timestamp
                            )

                            detected_hands = result.hand_landmarks

                            # ---------------------------------
                            # DO NOT PREDICT WITHOUT A HAND
                            # ---------------------------------

                            if not detected_hands:

                                self.frame_buffer.clear()
                                self.prediction_buffer.clear()

                                self.last_candidate = None
                                self.candidate_streak = 0

                                self.prediction = "Show a sign"
                                self.confidence = 0.0
                                self.status = "Waiting for hand"

                            else:

                                hand_data = np.zeros(
                                    (2, 21, 3),
                                    dtype=np.float64
                                )

                                for i, hand in enumerate(
                                    detected_hands[:2]
                                ):

                                    for j, landmark in enumerate(hand):

                                        hand_data[i, j] = [
                                            landmark.x,
                                            landmark.y,
                                            landmark.z
                                        ]

                                self.frame_buffer.append(hand_data)

                                # ---------------------------------
                                # PREDICT AFTER 30 FRAMES
                                # ---------------------------------

                                if len(self.frame_buffer) == 30:

                                    features = extract_features(
                                        list(self.frame_buffer)
                                    )

                                    raw_prediction = str(
                                        self.model.predict(features)[0]
                                    )

                                    if hasattr(self.model, "predict_proba"):

                                        probabilities = (
                                            self.model.predict_proba(features)[0]
                                        )

                                        raw_confidence = float(
                                            np.max(probabilities)
                                        )

                                    else:

                                        raw_confidence = 0.0

                                    # ---------------------------------
                                    # STABILITY CHECK
                                    # ---------------------------------

                                    self.prediction_buffer.append(
                                        raw_prediction
                                    )

                                    # Track consecutive candidate
                                    if raw_prediction == self.last_candidate:

                                        self.candidate_streak += 1

                                    else:

                                        self.last_candidate = raw_prediction
                                        self.candidate_streak = 1

                                    counts = Counter(
                                        self.prediction_buffer
                                    )

                                    stable_prediction, votes = (
                                        counts.most_common(1)[0]
                                    )

                                    # Accept only stable predictions
                                    # 10 out of the last 12 predictions
                                    # AND at least 8 consecutive predictions

                                    if (
                                        len(self.prediction_buffer) == 12
                                        and votes >= 10
                                        and self.candidate_streak >= 8
                                    ):

                                        if stable_prediction == raw_prediction:

                                            self.prediction = stable_prediction
                                            self.confidence = raw_confidence
                                            self.status = "Recognized"

                            # ---------------------------------
                            # CAMERA OVERLAY
                            # ---------------------------------

                            if self.prediction != "Show a sign":

                                display_text = (
                                    f"Sign: {self.prediction}"
                                )

                            else:

                                display_text = "Sign: Show a sign"

                            cv2.putText(
                                img,
                                display_text,
                                (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 200, 120),
                                2
                            )

                            cv2.putText(
                                img,
                                f"Confidence: {self.confidence:.2%}",
                                (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (255, 255, 255),
                                2
                            )

                            cv2.putText(
                                img,
                                f"Frames: {len(self.frame_buffer)}/30",
                                (20, 120),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 255, 255),
                                2
                            )

                    except Exception as e:

                        with self.lock:
                            self.error_message = str(e)

                    return av.VideoFrame.from_ndarray(
                        img,
                        format="bgr24"
                    )

            # -----------------------------------------
            # INSTRUCTIONS
            # -----------------------------------------

            st.markdown("### Live Recognition")

            col1, col2 = st.columns(2)

            with col1:

                st.markdown("""
<div class="info-card">
<h4>01 / Enable Camera</h4>
<p>Allow camera permission and start the live video stream.</p>
</div>
""", unsafe_allow_html=True)

            with col2:

                st.markdown("""
<div class="info-card">
<h4>02 / Show Gesture</h4>
<p>Place your hand clearly inside the camera frame.</p>
</div>
""", unsafe_allow_html=True)

            st.write("")

            st.info(
                "Maintain good lighting and show gestures included in your trained model."
            )

            # -----------------------------------------
            # WEBCAM
            # -----------------------------------------

            ctx = webrtc_streamer(
                key="signbridge-camera",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=SignProcessor,
                media_stream_constraints={
                    "video": True,
                    "audio": False
                },
                async_processing=True
            )

            # -----------------------------------------
            # LIVE RECOGNITION RESULT
            # -----------------------------------------

            st.divider()

            st.markdown("### Recognition Result")

            @st.fragment(run_every="500ms")
            def live_recognition_result():

                processor = ctx.video_processor

                if ctx.state.playing and processor is not None:

                    with processor.lock:

                        current_prediction = str(
                            processor.prediction
                        )

                        current_confidence = (
                            processor.confidence
                        )

                        current_status = (
                            processor.status
                        )

                        current_error = (
                            processor.error_message
                        )

                    if current_error:

                        st.error(
                            f"Recognition error: {current_error}"
                        )

                    safe_prediction = html.escape(
                        current_prediction
                    )

                    st.markdown(
                        f"""<div class="result-card">
<div class="result-label">RECOGNIZED SIGN</div>
<div class="result-value">{safe_prediction}</div>
</div>""",
                        unsafe_allow_html=True
                    )

                    col1, col2 = st.columns(2)

                    with col1:

                        st.metric(
                            "Recognition Confidence",
                            f"{current_confidence:.2%}"
                        )

                    with col2:

                        st.metric(
                            "Recognition Status",
                            current_status
                        )

                else:

                    st.markdown(
                        """<div class="result-card">
<div class="result-label">RECOGNIZED SIGN</div>
<div class="result-value">Show a sign</div>
</div>""",
                        unsafe_allow_html=True
                    )

                    col1, col2 = st.columns(2)

                    with col1:

                        st.metric(
                            "Recognition Confidence",
                            "0.00%"
                        )

                    with col2:

                        st.metric(
                            "Recognition Status",
                            "Waiting for sign"
                        )

                    st.info(
                        "Start the webcam to begin live recognition."
                    )

            live_recognition_result()


# =====================================================
# 9. FOOTER
# =====================================================

st.divider()

st.markdown(
    """
<div class="footer">
<strong style="color:#17212B;">SignBridge AI</strong>
<br>
Intelligent Communication • Accessible Technology
<br><br>
Connecting people through Speech, Text and Sign Language.
</div>
""",
    unsafe_allow_html=True
)
