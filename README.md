# Face Recognition Attendance Management System

An automated attendance system that uses real-time face recognition to identify people from a webcam feed and log their attendance to a CSV file — replacing manual roll-calls with a hands-free, camera-based workflow.

## Features

- 🎥 Real-time face detection and recognition from a live webcam feed
- 🧑‍🤝‍🧑 Recognizes multiple known faces from a folder of reference images
- 📝 Automatically logs each recognized person's name, date, and time to `Attendance.csv`
- 🚫 Marks attendance only once per person per session (no duplicate entries)
- ⚡ Lightweight setup using the `face_recognition` library and OpenCV

## Tech Stack

- **Language:** Python
- **Computer Vision:** OpenCV
- **Face Recognition:** face_recognition (built on dlib)
- **Data Handling:** NumPy, CSV

## How It Works

1. Reference images of known people are stored in the `student_images/` folder (one clear photo per person, named after them).
2. On startup, the system encodes each reference face into a numerical signature.
3. The webcam captures live frames; each detected face is encoded and compared against the known signatures.
4. On a match, the person's name, date, and time are written to `Attendance.csv` (once per session).

## Setup

```bash
# Clone the repo
git clone https://github.com/Tanish002/face-recognition-attendance-management-system.git
cd face-recognition-attendance-management-system

# (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install opencv-python face_recognition numpy
```

> **Note:** `face_recognition` depends on `dlib`. On Windows you may need CMake and Visual Studio C++ build tools installed first; on Linux/Mac, installing `cmake` beforehand usually resolves build issues.

## Usage

1. Add clear, front-facing photos of each person to the `student_images/` folder (the filename is used as the person's name, e.g. `Tanish.jpg`).
2. Run the script:

```bash
python face_recognition.py
```

3. The webcam window opens and begins recognizing faces. Recognized names are logged to `Attendance.csv`. Press `q` to quit.

## Project Structure
