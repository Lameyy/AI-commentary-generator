import os
import uuid
import time
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import google.generativeai as genai
from dotenv import load_dotenv
import ffmpeg  # Use ffmpeg-python instead of moviepy

# Load variables from the .env file into the environment
load_dotenv()

# --- Configuration ---
# NOW, read the credentials from the environment using their correct names
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT_NAME = "gpt-4.1"
API_VERSION = "2025-01-01-preview"
AZURE_OPENAI_SUBSCRIPTION_KEY = os.environ.get("AZURE_OPENAI_SUBSCRIPTION_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing

# --- AI Model Initialization ---

# Model 1: Azure OpenAI for generating stylized commentary
try:
    if not all([AZURE_OPENAI_ENDPOINT, DEPLOYMENT_NAME, API_VERSION, AZURE_OPENAI_SUBSCRIPTION_KEY]):
        raise ValueError("Azure OpenAI environment variables are not fully set. Check your .env file.")
    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=DEPLOYMENT_NAME,
        api_version=API_VERSION,
        azure_openai_api_key=AZURE_OPENAI_SUBSCRIPTION_KEY,
        temperature=0.7,
        max_tokens=256
    )
    print("AzureChatOpenAI initialized successfully.")
except Exception as e:
    print(f"Error initializing AzureChatOpenAI: {e}")
    llm = None

# Model 2: Google Gemini Pro Vision for analyzing video
try:
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable is not set. Check your .env file.")
    genai.configure(api_key=GOOGLE_API_KEY)
    vision_model = genai.GenerativeModel('models/gemini-1.5-flash')
    print("Google Gemini initialized successfully.")
except Exception as e:
    print(f"Error initializing Google Gemini: {e}")
    vision_model = None

# --- In-memory data store ---
commentary_history = []
UPLOAD_FOLDER = 'temp_videos'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- TTS Audio Generation ---
from gtts import gTTS
TTS_FOLDER = 'tts_audio'
os.makedirs(TTS_FOLDER, exist_ok=True)

# --- Helper Functions ---
def analyze_video_with_gemini(video_path):
    """
    Uploads a video to the Gemini API and asks for a description.
    """
    if not vision_model:
        raise ConnectionError("Google Gemini model not initialized.")

    print(f"Uploading file: {video_path}")
    video_file = genai.upload_file(path=video_path)
    
    # Wait for the file to be processed
    while video_file.state.name == "PROCESSING":
        print('.', end='')
        time.sleep(5)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError("Video file processing failed.")

    # Prompt for video analysis
    prompt = "Analyze this sports video clip and provide a concise, factual description of the main action. For example: 'A player in a blue jersey scores a goal with a header.'"
    
    print("Making Gemini API call...")
    response = vision_model.generate_content([prompt, video_file])
    
    # Clean up the uploaded file from Gemini's storage
    genai.delete_file(video_file.name)
    
    return response.text

def merge_audio_with_video(video_path, audio_path, output_path):
    """
    Merges the given audio file with the video file using ffmpeg-python.
    """
    (
        ffmpeg
        .input(video_path)
        .output(output_path, vcodec='copy', an=None)
        .run(overwrite_output=True, quiet=True)
    )
    (
        ffmpeg
        .input(video_path)
        .input(audio_path)
        .output(output_path, vcodec='copy', acodec='aac', strict='experimental', shortest=None)
        .overwrite_output()
        .run(quiet=True)
    )

# --- API Endpoints ---

@app.route('/generate_commentary_from_video', methods=['POST'])
def generate_commentary_from_video():
    """
    Generates sports commentary from an uploaded video file.
    """
    if 'video' not in request.files:
        return jsonify({"error": "No video file part in the request."}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    if not vision_model:
        return jsonify({"error": "Google Gemini model not initialized."}), 500

    # Save the video file temporarily
    filename = f"{uuid.uuid4()}_{file.filename}"
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(video_path)

    try:
        # --- Step 1: Analyze video to get a text description ---
        print("Analyzing video...")
        event_description = analyze_video_with_gemini(video_path)
        print(f"Video Description: {event_description}")

        # --- Step 2: Use Gemini to generate stylized commentary ---
        print("Generating commentary with Gemini...")
        # Get video duration in seconds
        import cv2
        video_capture = cv2.VideoCapture(video_path)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        video_duration = frame_count / fps if fps > 0 else 10
        video_capture.release()
        # Limit duration to a reasonable range (e.g., 5-60 seconds)
        video_duration = max(5, min(video_duration, 60))
        commentary_prompt = f"You are an expert sports commentator. Based on the following event, generate exciting and descriptive commentary that can be spoken in about {int(video_duration)} seconds.\nEvent: {event_description}"
        commentary_response = vision_model.generate_content(commentary_prompt)
        commentary = commentary_response.text

        # --- Generate TTS audio ---
        tts_filename = f"{uuid.uuid4()}_commentary.mp3"
        tts_path = os.path.join(TTS_FOLDER, tts_filename)
        tts = gTTS(text=commentary, lang='en')
        tts.save(tts_path)

        # --- Merge audio with video ---
        commentary_video_filename = f"{uuid.uuid4()}_commentary_video.mp4"
        commentary_video_path = os.path.join(TTS_FOLDER, commentary_video_filename)
        merge_audio_with_video(video_path, tts_path, commentary_video_path)

        # --- Store and return result ---
        commentary_history.append({
            "event": event_description,
            "commentary": commentary,
            "audio_file": tts_filename,
            "video_file": commentary_video_filename
        })
        return jsonify({
            "event_description": event_description,
            "commentary": commentary,
            "audio_url": f"/commentary_audio/{tts_filename}",
            "video_url": f"/commentary_video/{commentary_video_filename}"
        })

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up the temporarily saved video file
        if os.path.exists(video_path):
            os.remove(video_path)

# --- Audio Streaming Endpoint ---
@app.route('/commentary_audio/<filename>', methods=['GET'])
def stream_commentary_audio(filename):
    audio_path = os.path.join(TTS_FOLDER, filename)
    if not os.path.exists(audio_path):
        return jsonify({"error": "Audio file not found."}), 404
    return send_file(audio_path, mimetype='audio/mpeg')

@app.route('/commentary_video/<filename>', methods=['GET'])
def stream_commentary_video(filename):
    video_path = os.path.join(TTS_FOLDER, filename)
    if not os.path.exists(video_path):
        return jsonify({"error": "Video file not found."}), 404
    return send_file(video_path, mimetype='video/mp4')

@app.route('/commentary_history', methods=['GET'])
def get_commentary_history():
    """Returns the history of generated commentaries."""
    return jsonify(commentary_history)

# --- Main Execution ---
if __name__ == '__main__':
    app.run(debug=True, port=5001)