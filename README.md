<img width="312" height="150" alt="image" src="https://github.com/user-attachments/assets/749f9bd9-6abc-4886-9472-d98e3ac3939e" />
<img width="402" height="191" alt="image" src="https://github.com/user-attachments/assets/ec13e110-20be-4dad-bce9-94273f3aeeeb" />
<img width="316" height="260" alt="image" src="https://github.com/user-attachments/assets/df01893e-1d1c-46b2-93c8-d43b90cc4e4c" />
<img width="319" height="182" alt="image" src="https://github.com/user-attachments/assets/3193368c-aed0-43aa-b013-f1070ad0dc86" />

# Deepfake Detection System

A machine learning-powered web application that detects deepfake videos using Convolutional Neural Networks (CNN). The system analyzes video frames to determine if a video is authentic or artificially generated.

## Features

- **Video Upload Interface**: Simple web interface for uploading video files
- **Real-time Analysis**: Frame-by-frame analysis of uploaded videos
- **Confidence Scoring**: Provides percentage confidence in detection results
- **Suspicious Frame Extraction**: Highlights and saves frames with high fake probability
- **CNN Model**: Custom-trained deep learning model for accurate detection

## Technology Stack

- **Backend**: Flask (Python)
- **Machine Learning**: TensorFlow/Keras
- **Computer Vision**: OpenCV
- **Frontend**: HTML templates
- **Model Architecture**: Convolutional Neural Network (CNN)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Deepfake_Detection-main
```

2. Install required dependencies:
```bash
pip install flask tensorflow opencv-python numpy scikit-learn
```

3. Ensure the following directories exist:
   - `uploads/` - for uploaded videos
   - `static/outputs/` - for extracted suspicious frames
   - `uploads/real_videos/` - for training real videos
   - `uploads/fake_videos/` - for training fake videos

## Usage

### Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to `http://localhost:5000`

3. Upload a video file through the web interface

4. View the analysis results including:
   - Detection result (Real/Fake)
   - Confidence percentage
   - Suspicious frames (if any)

### Training Your Own Model

1. Prepare your dataset:
   - Place real videos in `uploads/real_videos/`
   - Place fake videos in `uploads/fake_videos/`

2. Train the model:
```bash
python train_model.py
```

3. The trained model will be saved as `model.h5`

## Model Architecture

The CNN model consists of:
- 3 Convolutional layers (32, 64, 128 filters)
- MaxPooling layers for dimensionality reduction
- Dropout layer (0.5) for regularization
- Dense layers for classification
- Sigmoid activation for binary classification

**Input**: 64x64x3 RGB frames
**Output**: Binary classification (Real=0, Fake=1)

## File Structure

```
Deepfake_Detection-main/
├── app.py                 # Main Flask application
├── train_model.py         # Model training script
├── model.h5              # Trained model weights
├── plot_training_metrics.py # Training visualization
├── templates/
│   ├── upload.html       # Upload interface
│   └── result.html       # Results display
├── static/outputs/       # Extracted suspicious frames
├── uploads/              # Uploaded videos and training data
└── README.md            # Project documentation
```

## Configuration

- **Frame Rate**: Extracts every 5th frame for analysis (configurable)
- **Detection Threshold**: 50% confidence threshold for fake classification
- **Image Size**: Frames resized to 64x64 pixels for processing
- **Model Format**: Keras HDF5 format (.h5)

## Results Interpretation

- **Real**: Video appears to be authentic (confidence < 50%)
- **Fake**: Video appears to be deepfake (confidence ≥ 50%)
- **Confidence Score**: Percentage indicating model's certainty
- **Suspicious Frames**: Individual frames flagged as potentially fake

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the MIT License.

## Disclaimer

This tool is for educational and research purposes. Detection accuracy may vary depending on the quality and type of deepfake technology used. Always verify results through multiple methods for critical applications.
