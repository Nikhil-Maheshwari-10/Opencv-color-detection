# OpenCV Color Detection: Real-time Quadrilateral Fill Analyzer

This project is a real-time web application built with Streamlit and OpenCV that detects quadrilateral shapes (rectangles or squares) in your webcam feed and analyzes their fill percentage. It visually displays the detected shape and overlays the fill percentage, making it useful for tasks like liquid level detection, object fill analysis, or educational demonstrations.

## Features
- **Real-time webcam processing**
- **Quadrilateral (rectangle/square) detection**
- **Fill percentage calculation and color-coded display**
- **Streamlit web interface with live video**

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/opencv-color-detection.git
   cd opencv-color-detection
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
   Or, if you use Poetry:
   ```sh
   poetry install
   ```

## Usage

Run the Streamlit app:
```sh
streamlit run app.py
```

- Allow access to your webcam when prompted.
- Show a rectangle or square object to your webcam.
- The app will display the detected shape and its fill percentage in real time.

## File Structure
- `app.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `pyproject.toml` - Poetry project configuration

## Dependencies
- streamlit
- streamlit-webrtc
- opencv-python
- opencv-python-headless
- numpy

## License
This project is licensed under the MIT License.

---

*Created by Nikhil Maheshwari*
