# train_model.py

import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Define paths to datasets

real_videos_path = "C:/cap_deepfake/uploads/real_videos"
fake_videos_path = "C:/cap_deepfake/uploads/fake_videos"

# Extract frames from each video
def extract_frames(video_path, label, frame_rate=5):
    frames = []
    video = cv2.VideoCapture(video_path)
    count = 0
    success = True
    while success:
        success, frame = video.read()
        if success and count % frame_rate == 0:
            frame = cv2.resize(frame, (64, 64))  # Resize to a smaller dimension
            frame = frame.astype(np.float32) / 255.0  # Normalize for the pixels
            frames.append((frame, label))
        count += 1
    video.release()
    return frames

# Load videos and extract frames
data = []
for label, folder in [('real', real_videos_path), ('fake', fake_videos_path)]:
    for filename in os.listdir(folder):
        video_path = os.path.join(folder, filename)
        data.extend(extract_frames(video_path, label))

# Convert data to numpy arrays
frames, labels = zip(*data)
frames = np.array(frames, dtype=np.float32)
labels = np.array([1 if label == 'fake' else 0 for label in labels])  # Fake=1, Real=0

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(frames, labels, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
EPOCHS = 10
history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_val, y_val), batch_size=16)

# Save the trained model
model.save("model.h5")
print("Model saved as model.h5")

