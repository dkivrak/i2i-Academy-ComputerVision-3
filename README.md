# i2i Academy Computer Vision Homework

This project is a real-time hand tracking and finger counting application developed with Python, OpenCV, and MediaPipe.

## Project Description

The application captures live video from the webcam, detects a human hand using MediaPipe Hands, extracts hand landmark points, and counts how many fingers are open. The final finger count is displayed directly on the video window in real time.

## Technologies Used

- Python
- OpenCV
- MediaPipe

## Features

- Opens the webcam and reads the video feed frame by frame.
- Detects one hand in real time.
- Extracts hand landmarks/joints using MediaPipe.
- Determines whether each finger is open or closed.
- Displays the number of open fingers on the video window.

## Installation

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the Application

```bash
python main.py
```

(Press q to close the webcam window.)

## Notes

The application uses a pre-trained hand tracking model through MediaPipe, so it does not require training a custom neural network from scratch.
