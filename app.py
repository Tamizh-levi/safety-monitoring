# app.py

from flask import Flask, render_template, Response, request
from flask_socketio import SocketIO
import cv2
import threading
from patient_monitor import PatientMonitor, Config

# Initialize Flask app and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_very_secret_key'
socketio = SocketIO(app)

# Initialize the SAFETY monitoring system
config = Config()
# Set the camera index to 0, which was confirmed to work by your test script.
config.VIDEO_SOURCE = 0
# Pass the socketio instance to the monitor so it can send alerts
monitor = PatientMonitor(config, socketio)
monitor.initialize()

# Global variable for the camera
video_capture = None

# A lock to ensure only one thread accesses the camera at a time
video_lock = threading.Lock()


def get_camera():
    """Returns the camera object, ensuring it's initialized once."""
    global video_capture
    with video_lock:
        if video_capture is None:
            print(f"Attempting to open camera with index {config.VIDEO_SOURCE}...")
            video_capture = cv2.VideoCapture(config.VIDEO_SOURCE)
            if not video_capture.isOpened():
                print(f"Error: Cannot open camera source with index {config.VIDEO_SOURCE}. Please check your camera connection or permissions.")
                video_capture = None
            else:
                print(f"Success! Camera opened with index {config.VIDEO_SOURCE}.")
    return video_capture


def generate_frames():
    """
    Generator function that yields the video frames with all processing applied.
    """
    cam = get_camera()
    if cam is None:
        print("Camera is not available. Yielding a placeholder.")
        # Return a black image as a placeholder if the camera is not available.
        placeholder_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ret, buffer = cv2.imencode('.jpg', placeholder_frame)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return

    while True:
        with video_lock:
            success, frame = cam.read()
        if not success:
            print("Failed to grab frame. Is the camera still connected?")
            return
        else:
            processed_frame = monitor.process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Route for the video streaming."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# --- WebSocket Event Handlers ---

@socketio.on('connect')
def handle_connect():
    """Handles a new client connection."""
    print('Client connected successfully.')
    # Send the current state of all features to the new client
    initial_states = {key: val for key, val in monitor.app_state.items() if key.endswith('_active')}
    socketio.emit('initial_states', initial_states)


@socketio.on('disconnect')
def handle_disconnect():
    """Handles a client disconnection."""
    print('Client disconnected.')


@socketio.on('toggle_feature')
def handle_toggle_feature(data):
    """
    Handles requests from the frontend to toggle a monitoring feature on or off.
    """
    feature = data.get('feature')
    status = data.get('status')

    # Map frontend IDs to backend state keys
    feature_map = {
        'cough': 'cough_detection_active',
        'unknown': 'unknown_person_active',
        'unidentified': 'unidentified_person_active',
        'bed-exit': 'bed_exit_active',
        'stroke': 'stroke_detection_active',
        'knife': 'knife_detection_active',
        'crowd': 'crowd_alert_active',
        'gestures': 'gestures_active',
        'voice': 'voice_active'
    }

    state_key = feature_map.get(feature)

    if state_key:
        monitor.app_state[state_key] = status
        print(f"Toggled {state_key} to {'ON' if status else 'OFF'}")

    else:
        print(f"Unknown feature received: {feature}")


if __name__ == '__main__':
    print("Starting Patient Monitoring Server...")
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
