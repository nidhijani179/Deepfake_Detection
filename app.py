
import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'  # Folder for storing frame images

# Load trained model
model = load_model("model.h5")

# Extract frames function with fake detection
def extract_frames_for_prediction(video_path, frame_rate=5):
    frames = []
    suspicious_frames = []  # To store paths of fake frames
    video = cv2.VideoCapture(video_path)
    count = 0
    success = True
    while success:
        success, frame = video.read()
        if success and count % frame_rate == 0:
            frame_resized = cv2.resize(frame, (64, 64))
            frame_normalized = frame_resized.astype(np.float32) / 255.0
            frames.append(frame_normalized)

            # Predict frame authenticity
            prediction = model.predict(np.expand_dims(frame_normalized, axis=0))
            confidence = prediction[0][0] * 100

            # Save suspicious frames if fake confidence is high
            if confidence > 50:  # Adjust threshold as needed
                output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"frame_{count}.jpg")
                cv2.imwrite(output_path, frame)  # Save original resolution frame
                suspicious_frames.append(output_path)

        count += 1
    video.release()
    return np.array(frames), suspicious_frames

@app.route("/", methods=["GET", "POST"])
def upload_video():
    if request.method == "POST":
        if "video" not in request.files:
            return redirect(request.url)
        file = request.files["video"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)
            return redirect(url_for("analyze_video", video_path=file_path))
    return render_template("upload.html")

@app.route("/analyze")
def analyze_video():
    video_path = request.args.get("video_path")
    frames, suspicious_frames = extract_frames_for_prediction(video_path)
    predictions = model.predict(frames)
    fake_confidence = predictions.mean() * 100  # Convert to percentage
    result = "Fake" if fake_confidence > 50 else "Real"
    
    return render_template("result.html", result=result, confidence=fake_confidence, suspicious_frames=suspicious_frames)

# Run Flask app
if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    app.run(debug=True)
# what does it detect name that also
