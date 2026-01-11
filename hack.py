import cv2
import os
import numpy as np
import face_recognition
from ultralytics import YOLO
import mediapipe as mp
import threading
import speech_recognition as sr
import pywhatkit
from playsound import playsound
import warnings
import datetime
import time
import logging
from pymongo import MongoClient
import bson
from gtts import gTTS
from typing import List, Tuple, Dict, Any, Optional
import math


# --- Configuration ---
class Config:
    """
    Central configuration class for the Patient Monitoring System.
    This class holds all the static variables, paths, and thresholds
    to make the system easily configurable from one place.
    """
    # --- File and Directory Paths ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    YOLO_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
    # IMPORTANT: Update this path to your trained knife detection model
    YOLO_KNIFE_MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
    LOG_FILE_PATH = os.path.join(BASE_DIR, "patient_monitoring.log")
    REMINDER_AUDIO_DIR = os.path.join(BASE_DIR, "reminder_audio")
    INTRUDER_LOGS_DIR = os.path.join(BASE_DIR, "intruder_logs")

    # --- Alarm Sound Asset Paths ---
    ALARM_SOUNDS_DIR = os.path.join(BASE_DIR, "alarm_sounds")
    ALARM_SOUNDS = {
        "call_nurse": os.path.join(ALARM_SOUNDS_DIR, "call_nurse.mp3"),
        "need_water": os.path.join(ALARM_SOUNDS_DIR, "need_water.mp3"),
        "call_family": os.path.join(ALARM_SOUNDS_DIR, "call_family.mp3"),
        "cancel_request": os.path.join(ALARM_SOUNDS_DIR, "cancel_request.mp3"),
        "unknown_alert": os.path.join(ALARM_SOUNDS_DIR, "unknown_alert.mp3"),
        "unidentified_alert": os.path.join(ALARM_SOUNDS_DIR, "unknown_alert.mp3"),
        "crowd_alert": os.path.join(ALARM_SOUNDS_DIR, "crowd_alert.mp3"),
        "bed_exit_alert": os.path.join(ALARM_SOUNDS_DIR, "fall_alert.mp3"),
        "stroke_alert": os.path.join(ALARM_SOUNDS_DIR, "fall_alert.mp3"),
        "knife_alert": os.path.join(ALARM_SOUNDS_DIR, "knife_alert.mp3"),
        "cough_alert": os.path.join(ALARM_SOUNDS_DIR, "cough_alert.mp3")
    }

    # --- System & Patient Information ---
    # Ensure this URL is accessible from the machine running the script.
    VIDEO_SOURCE = 0
    MONITORED_PATIENT_ID = "PAT001"
    RECEIVER_PHONE = "+919345531046"  # Receiver's phone number for WhatsApp alerts

    # --- Database Configuration ---
    MONGO_URI = "mongodb://localhost:27017/"
    MONGO_DB_NAME = "patient_monitoring"
    ALERTS_COLLECTION_NAME = "alerts"
    PATIENTS_COLLECTION_NAME = "patients"
    SCHEDULES_COLLECTION_NAME = "schedules"

    # --- Detection Thresholds & Timings ---
    YOLO_CONFIDENCE_THRESHOLD = 0.7
    FACE_RECOGNITION_TOLERANCE = 0.5
    KNIFE_HIGH_CONFIDENCE_THRESHOLD = 0.70  # Threshold for an immediate, persistent knife alert
    CROWD_THRESHOLD = 4  # Number of people considered a crowd
    ALERT_CONFIRMATION_SEC = 3  # Seconds to wait before confirming a standard alert
    UNIDENTIFIED_CONFIRMATION_SEC = 4
    LOW_LIGHT_THRESHOLD = 80  # Average pixel intensity to trigger night mode
    BED_EXIT_CONFIRMATION_SEC = 5
    GESTURE_CONFIRMATION_SEC = 2
    STROKE_CONFIRMATION_SEC = 4
    MOUTH_DROOP_THRESHOLD = 0.03  # Facial landmark difference for stroke detection
    KNIFE_CONFIRMATION_SEC = 2
    COUGH_CONFIRMATION_SEC = 2  # Cooldown between individual cough detections
    MOUTH_ASPECT_RATIO_THRESHOLD = 0.3  # For cough detection
    HEAD_FORWARD_THRESHOLD = 0.03  # Z-axis change for cough detection
    COUGH_COUNT_THRESHOLD = 1  # Alert after this many coughs
    COUGH_RESET_SEC = 60  # Reset cough count after this many seconds of no coughing

    # --- Performance Settings ---
    DETECTION_INTERVAL = 5  # Run full detection every N frames to save resources

    # --- Voice Recognition Settings ---
    VOICE_AMBIENT_ADJUST_DUR = 4
    VOICE_PAUSE_THRESHOLD = 1.5
    VOICE_TIMEOUT = 8
    VOICE_PHRASE_LIMIT = 10


class DatabaseManager:
    """Handles all interactions with the MongoDB database."""

    def __init__(self, config: Config):
        self.config = config
        self.db_client = None
        self.alerts_collection = None
        self.patients_collection = None
        self.schedules_collection = None
        self.connect()

    def connect(self):
        """Establishes connection to the MongoDB server and collections."""
        try:
            self.db_client = MongoClient(self.config.MONGO_URI, serverSelectionTimeoutMS=5000)
            self.db_client.admin.command('ismaster')  # Check connection
            db = self.db_client[self.config.MONGO_DB_NAME]
            self.alerts_collection = db[self.config.ALERTS_COLLECTION_NAME]
            self.patients_collection = db[self.config.PATIENTS_COLLECTION_NAME]
            self.schedules_collection = db[self.config.SCHEDULES_COLLECTION_NAME]
            logging.info("Successfully connected to MongoDB.")
            print("Successfully connected to MongoDB.")
        except Exception as e:
            logging.error(f"Could not connect to MongoDB: {e}")
            print(f"Error: Could not connect to MongoDB.")
            self.db_client = None

    def get_patient_details(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Fetches a single SAFETY's document from the database."""
        if self.patients_collection is None: return None
        try:
            return self.patients_collection.find_one({"_id": patient_id})
        except Exception as e:
            print(f"Error fetching SAFETY details for {patient_id}: {e}")
            return None

    def get_all_patients_with_photos(self) -> List[Dict[str, Any]]:
        """Retrieves all patients who have a photo stored for face recognition."""
        if self.patients_collection is None: return []
        try:
            return list(self.patients_collection.find({"photo": {"$exists": True}}))
        except Exception as e:
            print(f"Error fetching patients from DB: {e}")
            return []

    def log_event(self, patient_id: str, patient_name: str, message: str, images: Optional[List[np.ndarray]] = None):
        """Logs an alert or event into the alerts collection."""
        if self.alerts_collection is None: return
        try:
            event_doc = {
                "timestamp": datetime.datetime.utcnow(),
                "patient_id": patient_id,
                "patient_name": patient_name,
                "event_message": message
            }
            if images:
                # Convert images to BSON binary format for storage
                event_doc["image_snapshots"] = [
                    bson.binary.Binary(cv2.imencode('.jpg', img)[1].tobytes()) for img in images
                ]
            self.alerts_collection.insert_one(event_doc)
        except Exception as e:
            logging.error(f"Failed to log event to MongoDB: {e}")

    def get_schedule_for_now(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Checks if there is a scheduled reminder for the current time."""
        if self.schedules_collection is None: return None
        try:
            current_time_str = datetime.datetime.now().strftime("%H:%M")
            return self.schedules_collection.find_one({
                "patient_id": patient_id,
                "time": current_time_str
            })
        except Exception as e:
            print(f"Error checking schedule in DB: {e}")
            return None

    def close(self):
        """Closes the database connection."""
        if self.db_client:
            self.db_client.close()
            print("MongoDB connection closed.")


class AlertManager:
    """Handles triggering alarms, sending notifications, and managing alert states."""

    def __init__(self, config: Config, db_manager: DatabaseManager):
        self.config = config
        self.db_manager = db_manager
        self.patient_id = config.MONITORED_PATIENT_ID
        self.patient_name = "N/A"

        # Timers and flags to manage alerts that require confirmation over time
        self.alert_timers = {
            "unknown": None, "unidentified": None, "crowd": None,
            "bed_exit": None, "stroke": None, "knife": None, "cough": None
        }
        self.alert_sent_flags = {
            "unknown": False, "unidentified": False, "crowd": False,
            "bed_exit": False, "stroke": False, "knife": False, "cough": False
        }

    def set_patient_info(self, name: str):
        """Sets the name of the monitored SAFETY."""
        self.patient_name = name

    def send_whatsapp_alert(self, message: str):
        """Sends an alert message via WhatsApp using pywhatkit."""
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            full_message = (
                f"*** Patient Room Monitoring Alert ***\n\n"
                f"Patient ID: {self.patient_id}\n"
                f"Patient Name: {self.patient_name}\n\n"
                f"Timestamp: {timestamp}\n"
                f"Alert Details: {message}\n\n"
                f"This is an automated alert."
            )
            print(f"Sending WhatsApp message: {message}")
            pywhatkit.sendwhatmsg_instantly(
                self.config.RECEIVER_PHONE,
                full_message,
                wait_time=15,
                tab_close=True
            )
        except Exception as e:
            print(f"Failed to send WhatsApp message: {e}")

    def trigger_alarm(self, message: str, sound_key: Optional[str] = None, images: Optional[List[np.ndarray]] = None):
        """The main alarm function: logs, notifies, saves images, and plays sound."""
        print(f"ALARM: {message}")
        self.db_manager.log_event(self.patient_id, self.patient_name, message, images)
        if images:
            self._save_images(message, images)
        # Run WhatsApp and sound playback in separate threads to avoid blocking the main loop
        threading.Thread(target=self.send_whatsapp_alert, args=(message,), daemon=True).start()
        if sound_key and sound_key in self.config.ALARM_SOUNDS:
            sound_file = self.config.ALARM_SOUNDS[sound_key]
            if os.path.exists(sound_file):
                threading.Thread(target=playsound, args=(sound_file,), daemon=True).start()
            else:
                logging.warning(f"Sound file not found: {sound_file}")

    def _save_images(self, message: str, images: List[np.ndarray]):
        """Saves snapshots associated with an alert to the intruder logs directory."""
        for i, img in enumerate(images):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(
                self.config.INTRUDER_LOGS_DIR,
                f"{message.replace(' ', '_')}_{timestamp}_{i}.jpg"
            )
            cv2.imwrite(filename, img)
            print(f"Saved snapshot: {filename}")

    def check_and_trigger_timed_alert(self, frame: np.ndarray, alert_type: str, condition: bool, conf_sec: int,
                                      message: str, sound_key: str, images: List[np.ndarray], y_pos: int):
        """
        Generic handler for alerts that trigger only after a condition is met for a
        specified confirmation period (conf_sec).
        """
        timer = self.alert_timers.get(alert_type)
        sent_flag = self.alert_sent_flags.get(alert_type)

        if condition:
            # If the condition is met, start or continue the timer
            if timer is None:
                self.alert_timers[alert_type] = time.time()
            elapsed = time.time() - self.alert_timers[alert_type]

            # If the timer exceeds the confirmation time and no alert has been sent, trigger it
            if not sent_flag and elapsed > conf_sec:
                self.trigger_alarm(message, sound_key, images)
                self.alert_sent_flags[alert_type] = True
            elif not sent_flag:
                # Display a countdown on the screen
                countdown = conf_sec - elapsed
                cv2.putText(frame, f"{message} in: {int(countdown) + 1}s",
                            (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # If the condition is no longer met, reset the timer and flag
            self.alert_timers[alert_type] = None
            self.alert_sent_flags[alert_type] = False


class UIManager:
    """Handles drawing UI elements like buttons and processing user input."""

    def __init__(self, app_state: Dict[str, Any]):
        self.app_state = app_state
        self.buttons: Dict[str, Dict[str, Any]] = {}

    def draw_buttons(self, frame: np.ndarray) -> np.ndarray:
        """Draws all interactive toggle buttons on the frame."""
        self.buttons.clear()
        button_definitions = {
            "cough_detection": ("Cough", "left"), "unknown_person": ("Unknown", "left"),
            "unidentified_person": ("Unidentified", "left"), "bed_exit": ("Bed Exit", "left"),
            "stroke_detection": ("Stroke", "left"), "knife_detection": ("Knife", "left"),
            "crowd_alert": ("Crowd", "right"), "gestures": ("Gestures", "right"),
            "voice": ("Voice", "right")
        }
        y_offsets = {"left": frame.shape[0] - 250, "right": frame.shape[0] - 130}

        for key, (name, position) in button_definitions.items():
            is_active = self.app_state.get(f"{key}_active", False)
            text = f"{name}: {'ON' if is_active else 'OFF'}"
            color = (0, 255, 0) if is_active else (0, 165, 255)  # Green for ON, Orange for OFF

            (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            rect_w = text_width + 20
            rect_h = 30

            if position == "left":
                rect_x, rect_y = 10, y_offsets[position]
            else:
                rect_x, rect_y = frame.shape[1] - rect_w - 10, y_offsets[position]

            rect = (rect_x, rect_y, rect_w, rect_h)
            self.buttons[key] = {"rect": rect}

            cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), color, -1)
            cv2.putText(frame, text, (rect_x + 10, rect_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            y_offsets[position] += 40

        return frame

    def handle_click(self, event, x: int, y: int, flags, param):
        """OpenCV mouse callback to toggle application modes based on button clicks."""
        if event == cv2.EVENT_LBUTTONDOWN:
            for key, button in self.buttons.items():
                bx, by, bw, bh = button["rect"]
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    state_key = f"{key}_active"
                    self.app_state[state_key] = not self.app_state.get(state_key, False)
                    print(f"Toggled {key}: {'ON' if self.app_state[state_key] else 'OFF'}")

                    # Special handling for bed exit mode to reset state
                    if key == "bed_exit":
                        if self.app_state[state_key] and self.app_state.get("bed_roi") is None:
                            print("Please select the bed area for exit detection.")
                        elif not self.app_state[state_key]:
                            self.app_state["patient_was_in_bed"] = False
                    break


# --- Main Application Class ---
class PatientMonitor:
    def __init__(self, config: Config):
        self.config = config
        self.setup_logging()
        warnings.filterwarnings("ignore")

        # Central state dictionary to control application flow and features
        self.app_state = {
            "running": True, "bed_roi": None, "cough_detection_active": False,
            "unknown_person_active": False, "unidentified_person_active": False,
            "crowd_alert_active": False, "gestures_active": False, "voice_active": False,
            "bed_exit_active": False, "stroke_detection_active": False,
            "knife_detection_active": False,
            "patient_was_in_bed": False
        }

        # Initialize managers and models
        self.db_manager = DatabaseManager(config)
        self.alert_manager = AlertManager(config, self.db_manager)
        self.ui_manager = UIManager(self.app_state)

        self.yolo_model = YOLO(self.config.YOLO_MODEL_PATH)
        try:
            self.yolo_knife_model = YOLO(self.config.YOLO_KNIFE_MODEL_PATH)
            print("Knife detection model loaded successfully.")
        except Exception as e:
            print(f"Error loading knife detection model: {e}")
            self.yolo_knife_model = None
            self.app_state["knife_detection_active"] = False

        # Initialize MediaPipe models
        self.hands = mp.solutions.hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5,
                                                         min_tracking_confidence=0.5)
        self.drawing_utils = mp.solutions.drawing_utils
        self.drawing_spec = self.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)

        # Initialize Speech Recognition
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()

        # State variables for tracking and recognition
        self.known_face_encodings = []
        self.known_face_names = []
        self.patient_tracker = None  # OpenCV tracker object
        self.patient_bbox = None
        self.other_people_bboxes = []

        # State for cough detection
        self.previous_nose_z = None
        self.previous_shoulder_z = None
        self.cough_count = 0
        self.last_cough_time = 0
        self.cough_alert_sent = False

        # State for persistent knife alert
        self.knife_detected_at_high_conf = False

        # State for reminders and gestures
        self.played_reminders_today = set()
        self.last_reminder_check_date = datetime.date.today()
        self.gesture_actions = {0: ("Call Nurse", "call_nurse"), 2: ("Need Water", "need_water"),
                                3: ("Call Family", "call_family")}
        self.gesture_detected_time = {}

    def setup_logging(self):
        """Configures the logging for the application."""
        logging.basicConfig(filename=self.config.LOG_FILE_PATH, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        print(f"Logging to {self.config.LOG_FILE_PATH}")

    def initialize(self):
        """Loads data from the database and starts background services."""
        self.load_patient_data()
        self.load_known_faces_from_db()
        os.makedirs(self.config.REMINDER_AUDIO_DIR, exist_ok=True)
        os.makedirs(self.config.INTRUDER_LOGS_DIR, exist_ok=True)
        # Start background threads for non-blocking tasks
        threading.Thread(target=self.schedule_checker, daemon=True).start()
        threading.Thread(target=self.listen_for_voice_commands, daemon=True).start()
        print("Background threads (Schedule, Voice) started.")

    def load_patient_data(self):
        """Loads the monitored SAFETY's details from the database."""
        patient_doc = self.db_manager.get_patient_details(self.config.MONITORED_PATIENT_ID)
        if patient_doc and patient_doc.get("name"):
            patient_name = patient_doc["name"]
            self.alert_manager.set_patient_info(patient_name)
            print(f"Monitoring SAFETY: {patient_name} (ID: {self.config.MONITORED_PATIENT_ID})")
        else:
            print(f"Warning: Could not find details for SAFETY ID: {self.config.MONITORED_PATIENT_ID}")

    def load_known_faces_from_db(self):
        """Loads and encodes faces from the database for recognition."""
        print("Loading known faces from database...")
        patients = self.db_manager.get_all_patients_with_photos()
        for patient in patients:
            patient_name, photo_data = patient.get('name'), patient.get('photo')
            if isinstance(photo_data, bytes):
                try:
                    nparr = np.frombuffer(photo_data, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if image is None: continue
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    encodings = face_recognition.face_encodings(rgb_image)
                    if encodings:
                        self.known_face_encodings.append(encodings[0])
                        self.known_face_names.append(patient_name)
                except Exception as e:
                    print(f"Could not process photo for {patient_name}: {e}")
        print(f"Successfully loaded {len(self.known_face_names)} faces from the database.")

    def speak_reminder(self, text: str):
        """Uses Google Text-to-Speech to read a reminder aloud."""
        try:
            print(f"Speaking reminder: {text}")
            tts = gTTS(text=text, lang='en')
            # Save to a temporary file, play it, then delete it
            filename = os.path.join(self.config.REMINDER_AUDIO_DIR, f"reminder_{int(time.time())}.mp3")
            tts.save(filename)
            playsound(filename)
            os.remove(filename)
        except Exception as e:
            print(f"Error in text-to-speech: {e}")

    def schedule_checker(self):
        """Background thread to check for scheduled reminders every 30 seconds."""
        while self.app_state["running"]:
            today = datetime.date.today()
            # Reset the list of played reminders at midnight
            if today != self.last_reminder_check_date:
                self.played_reminders_today.clear()
                self.last_reminder_check_date = today
                print("Resetting daily reminders.")

            reminder = self.db_manager.get_schedule_for_now(self.config.MONITORED_PATIENT_ID)
            if reminder:
                reminder_id = f"{reminder['patient_id']}-{reminder['time']}"
                if reminder_id not in self.played_reminders_today:
                    full_message = f"Hello {self.alert_manager.patient_name}, {reminder.get('message', 'you have a scheduled reminder.')}"
                    threading.Thread(target=self.speak_reminder, args=(full_message,), daemon=True).start()
                    self.played_reminders_today.add(reminder_id)
            time.sleep(30)

    def listen_for_voice_commands(self):
        """Background thread to listen for voice commands from the SAFETY."""
        with self.mic as source:
            print("Adjusting for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source, duration=self.config.VOICE_AMBIENT_ADJUST_DUR)
            self.recognizer.pause_threshold = self.config.VOICE_PAUSE_THRESHOLD
            print("Voice recognition is active.")

        while self.app_state["running"]:
            if self.app_state["voice_active"]:
                try:
                    with self.mic as source:
                        audio = self.recognizer.listen(source, timeout=self.config.VOICE_TIMEOUT,
                                                       phrase_time_limit=self.config.VOICE_PHRASE_LIMIT)
                    command = self.recognizer.recognize_google(audio, language='en-US').lower()

                    # Process recognized commands
                    if any(p in command for p in ["call nurse", "help"]):
                        self.alert_manager.trigger_alarm("Voice Command: Call Nurse", "call_nurse")
                    elif any(p in command for p in ["need water", "thirsty"]):
                        self.alert_manager.trigger_alarm("Voice Command: Need Water", "need_water")
                    elif "call family" in command:
                        self.alert_manager.trigger_alarm("Voice Command: Call Family", "call_family")
                    elif any(p in command for p in ["cancel", "stop"]):
                        self.alert_manager.trigger_alarm("Voice Command: Cancel Request", "cancel_request")
                except sr.WaitTimeoutError:
                    continue  # No speech detected
                except (sr.UnknownValueError, sr.RequestError):
                    time.sleep(2)  # Handle API errors or unintelligible speech
            else:
                time.sleep(0.5)  # Sleep briefly if voice detection is off

    def process_frame_lighting(self, frame: np.ndarray) -> np.ndarray:
        """Enhances the frame if low light is detected (Night Mode)."""
        if np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) < self.config.LOW_LIGHT_THRESHOLD:
            # Use CLAHE (Contrast Limited Adaptive Histogram Equalization) for enhancement
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced_lab = cv2.merge((cl, a, b))
            enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            cv2.putText(enhanced_frame, "Night Mode", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            return enhanced_frame
        return frame

    def _calculate_iou(self, boxA, boxB):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.
        """
        # Determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # Compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # Compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # Compute the intersection over union
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def process_person_detection(self, frame: np.ndarray, rgb_frame: np.ndarray) -> Tuple[
        int, Optional[Tuple[int, int, int, int]]]:
        """
        Detects, tracks, and identifies all persons in the frame.
        This is the core logic for unknown, unidentified, and crowd alerts.
        """
        patient_identified_this_frame = False

        # --- Step 1: Update existing SAFETY tracker if it exists ---
        if self.patient_tracker is not None:
            success, box = self.patient_tracker.update(rgb_frame)
            if success:
                x, y, w, h = map(int, box)
                self.patient_bbox = (x, y, x + w, y + h)
                patient_identified_this_frame = True
            else:
                # Tracker failed, so we need to re-detect
                self.patient_tracker = None
                self.patient_bbox = None

        # --- Step 2: Run YOLO detection to find all people ---
        results = self.yolo_model(rgb_frame, verbose=False, conf=self.config.YOLO_CONFIDENCE_THRESHOLD)
        person_count = 0
        unknown_face_detected = False
        unidentified_person_present = False
        intruder_rois = []
        all_bboxes = []

        for result in results:
            for box in result.boxes:
                if int(box.cls) == 0:  # Class 0 is 'person' in COCO dataset
                    person_count += 1
                    all_bboxes.append(tuple(map(int, box.xyxy[0])))

        # If no people are detected, reset the SAFETY tracker
        if person_count == 0:
            self.patient_tracker = None
            self.patient_bbox = None

        # --- Step 3: Identify each person detected ---
        self.other_people_bboxes.clear()
        temp_patient_bbox = None
        patient_found_by_face = False

        for bbox in all_bboxes:
            x1, y1, x2, y2 = bbox

            # *** LOGIC TO PREVENT DOUBLE-DETECTION ***
            if self.patient_bbox:
                iou = self._calculate_iou(self.patient_bbox, bbox)
                if iou > 0.6:  # If the new box is likely the same person, skip it
                    continue

            person_roi = rgb_frame[y1:y2, x1:x2].copy()
            if person_roi.shape[0] < 20 or person_roi.shape[1] < 20:
                continue

            # --- Step 4: Face Recognition on the detected person ---
            face_locations = face_recognition.face_locations(person_roi)
            if face_locations:
                face_encodings = face_recognition.face_encodings(person_roi, face_locations)
                if not face_encodings: continue

                matches = face_recognition.compare_faces(self.known_face_encodings, face_encodings[0],
                                                         tolerance=self.config.FACE_RECOGNITION_TOLERANCE)
                name = "Unknown"
                if any(matches):
                    name = self.known_face_names[matches.index(True)]

                if name == self.alert_manager.patient_name:
                    temp_patient_bbox = (x1, y1, x2, y2)
                    patient_found_by_face = True
                else:
                    self.other_people_bboxes.append(bbox)
                    unknown_face_detected = True
                    intruder_rois.append(frame[y1:y2, x1:x2])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                unidentified_person_present = True
                self.other_people_bboxes.append(bbox)
                intruder_rois.append(frame[y1:y2, x1:x2])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # --- Step 5: Update the SAFETY tracker ---
        if patient_found_by_face:
            self.patient_bbox = temp_patient_bbox
            self.patient_tracker = cv2.TrackerCSRT_create()
            self.patient_tracker.init(rgb_frame, (self.patient_bbox[0], self.patient_bbox[1],
                                                  self.patient_bbox[2] - self.patient_bbox[0],
                                                  self.patient_bbox[3] - self.patient_bbox[1]))
        elif not patient_identified_this_frame:
            self.patient_bbox = None
            self.patient_tracker = None

        if self.patient_bbox:
            x1, y1, x2, y2 = self.patient_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, self.alert_manager.patient_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)

        # --- Step 6: Trigger timed alerts based on findings ---
        if self.app_state["unknown_person_active"]:
            self.alert_manager.check_and_trigger_timed_alert(frame, "unknown", unknown_face_detected,
                                                             self.config.ALERT_CONFIRMATION_SEC,
                                                             "Other Person Detected",
                                                             "unknown_alert", intruder_rois, frame.shape[0] - 10)
        if self.app_state["unidentified_person_active"]:
            self.alert_manager.check_and_trigger_timed_alert(frame, "unidentified", unidentified_person_present,
                                                             self.config.UNIDENTIFIED_CONFIRMATION_SEC,
                                                             "Unidentified Person", "unidentified_alert", intruder_rois,
                                                             frame.shape[0] - 40)
        if self.app_state["crowd_alert_active"]:
            is_crowd = person_count > self.config.CROWD_THRESHOLD
            self.alert_manager.check_and_trigger_timed_alert(frame, "crowd", is_crowd,
                                                             self.config.ALERT_CONFIRMATION_SEC,
                                                             f"Crowd ({person_count})", "crowd_alert", [frame],
                                                             frame.shape[0] - 70)
        return person_count, self.patient_bbox

    def process_cough_detection(self, frame: np.ndarray, rgb_frame: np.ndarray,
                                patient_bbox: Optional[Tuple[int, int, int, int]]):
        """Detects frequent coughing based on mouth opening and forward body motion."""
        if not self.app_state["cough_detection_active"]:
            self.cough_count = 0
            self.last_cough_time = 0
            self.cough_alert_sent = False
            return

        # Reset cough count if the SAFETY is not visible or after the reset timer expires
        if patient_bbox is None or (
                self.last_cough_time > 0 and time.time() - self.last_cough_time > self.config.COUGH_RESET_SEC):
            self.cough_count = 0
            self.cough_alert_sent = False

        cv2.putText(frame, f"Cough Count: {self.cough_count}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255),
                    2)

        if patient_bbox is None: return

        x1, y1, x2, y2 = patient_bbox
        if x2 <= x1 or y2 <= y1: return

        patient_roi_rgb = rgb_frame[y1:y2, x1:x2]
        patient_roi_bgr = frame[y1:y2, x1:x2]

        if patient_roi_rgb.shape[0] < 1 or patient_roi_rgb.shape[1] < 1:
            return

        pose_results = self.pose.process(patient_roi_rgb)
        face_mesh_results = self.face_mesh.process(patient_roi_rgb)

        mouth_open = False
        head_forward = False
        body_forward = False

        # Check for open mouth using face mesh
        if face_mesh_results.multi_face_landmarks:
            for face_landmarks in face_mesh_results.multi_face_landmarks:
                p_upper_lip = face_landmarks.landmark[13]
                p_lower_lip = face_landmarks.landmark[14]
                p_left_corner = face_landmarks.landmark[61]
                p_right_corner = face_landmarks.landmark[291]
                ver_dist = math.hypot(p_upper_lip.x - p_lower_lip.x, p_upper_lip.y - p_lower_lip.y)
                hor_dist = math.hypot(p_left_corner.x - p_right_corner.x, p_left_corner.y - p_right_corner.y)

                if hor_dist > 0:
                    mar = ver_dist / hor_dist  # Mouth Aspect Ratio
                    if mar > self.config.MOUTH_ASPECT_RATIO_THRESHOLD:
                        mouth_open = True

        # Check for forward head and body motion using pose estimation
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            try:
                current_nose_z = landmarks[self.mp_pose.PoseLandmark.NOSE.value].z
                if self.previous_nose_z is not None and self.previous_nose_z - current_nose_z > self.config.HEAD_FORWARD_THRESHOLD:
                    head_forward = True
                self.previous_nose_z = current_nose_z

                left_shoulder_z = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].z
                right_shoulder_z = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z
                current_shoulder_z = (left_shoulder_z + right_shoulder_z) / 2
                if self.previous_shoulder_z is not None and self.previous_shoulder_z - current_shoulder_z > self.config.HEAD_FORWARD_THRESHOLD:
                    body_forward = True
                self.previous_shoulder_z = current_shoulder_z
            except IndexError:
                self.previous_nose_z, self.previous_shoulder_z = None, None
        else:
            self.previous_nose_z, self.previous_shoulder_z = None, None

        # A cough is a combination of these movements
        if mouth_open and head_forward and body_forward:
            current_time = time.time()
            # Debounce: only count a new cough after a short cooldown
            if current_time - self.last_cough_time > self.config.COUGH_CONFIRMATION_SEC:
                self.cough_count += 1
                self.last_cough_time = current_time
                print(f"Cough detected. Count: {self.cough_count}")

        # If the cough count reaches the threshold, send an alert
        if self.cough_count >= self.config.COUGH_COUNT_THRESHOLD and not self.cough_alert_sent:
            alert_message = f"Frequent Coughing Detected ({self.cough_count} coughs)"
            self.alert_manager.trigger_alarm(alert_message, "cough_alert", [frame])
            self.cough_alert_sent = True  # Prevent re-sending for this batch
            self.cough_count = 0  # Reset after alerting

    def process_gestures(self, frame: np.ndarray, rgb_frame: np.ndarray):
        """Detects hand gestures for calling for help."""
        if not self.app_state["gestures_active"]: return
        results_hands = self.hands.process(rgb_frame)
        if results_hands.multi_hand_landmarks:
            for hand_landmarks, handedness_info in zip(results_hands.multi_hand_landmarks,
                                                       results_hands.multi_handedness):
                fingers_up = self._count_fingers(hand_landmarks, handedness_info)
                cv2.putText(frame, f"Fingers: {fingers_up}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                            2)

                if fingers_up in self.gesture_actions:
                    label, sound_key = self.gesture_actions[fingers_up]
                    cv2.putText(frame, f"Gesture: {label}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    # Use a timed confirmation for gestures to avoid false positives
                    current_time = time.time()
                    if fingers_up not in self.gesture_detected_time:
                        self.gesture_detected_time[fingers_up] = current_time
                    elapsed = current_time - self.gesture_detected_time[fingers_up]

                    if elapsed > self.config.GESTURE_CONFIRMATION_SEC:
                        self.alert_manager.trigger_alarm(f"Gesture: {label}", sound_key, images=[frame])
                        self.gesture_detected_time.pop(fingers_up, None)  # Reset after triggering
                    else:
                        countdown = self.config.GESTURE_CONFIRMATION_SEC - elapsed
                        cv2.putText(frame, f"in {int(countdown) + 1}s", (150, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 255, 255), 2)
                else:
                    self.gesture_detected_time.clear()
        else:
            self.gesture_detected_time.clear()

    def _count_fingers(self, hand_landmarks, handedness_info) -> int:
        """Helper function to count the number of extended fingers."""
        finger_count = 0
        hand_label = handedness_info.classification[0].label
        tip_ids = [4, 8, 12, 16, 20]  # Landmark IDs for finger tips

        # Count four fingers (index to pinky)
        for i in range(1, 5):
            if hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[tip_ids[i] - 2].y:
                finger_count += 1

        # Count thumb (based on x-axis position relative to wrist)
        if (hand_label == "Right" and hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[
            tip_ids[0] - 1].x) or \
                (hand_label == "Left" and hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[
                    tip_ids[0] - 1].x):
            finger_count += 1

        return finger_count

    def process_bed_exit(self, frame: np.ndarray, patient_bbox: Optional[Tuple[int, int, int, int]]):
        """Detects if the SAFETY has left a predefined bed area."""
        bed_roi = self.app_state.get("bed_roi")
        if not self.app_state["bed_exit_active"] or bed_roi is None:
            return

        x, y, w, h = bed_roi
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        patient_in_bed = False
        if patient_bbox:
            px1, py1, px2, py2 = patient_bbox
            person_center_x = (px1 + px2) // 2
            person_center_y = (py1 + py2) // 2
            # Check if the center of the SAFETY is inside the bed ROI
            if x < person_center_x < (x + w) and y < person_center_y < (y + h):
                patient_in_bed = True

        if patient_in_bed:
            self.app_state["patient_was_in_bed"] = True

        # Alert condition: SAFETY was previously in bed but is not anymore
        patient_out_of_bed = self.app_state["patient_was_in_bed"] and not patient_in_bed

        self.alert_manager.check_and_trigger_timed_alert(
            frame, "bed_exit", patient_out_of_bed,
            self.config.BED_EXIT_CONFIRMATION_SEC, "Patient Left Bed", "bed_exit_alert",
            [frame] if patient_out_of_bed else [],
            frame.shape[0] - 130
        )

    def process_stroke_detection_mediapipe(self, frame: np.ndarray, rgb_frame: np.ndarray,
                                           patient_bbox: Optional[Tuple[int, int, int, int]]):
        """Detects potential stroke symptoms based on facial asymmetry (mouth droop)."""
        if not self.app_state["stroke_detection_active"] or patient_bbox is None:
            self.alert_manager.alert_timers["stroke"] = None
            self.alert_manager.alert_sent_flags["stroke"] = False
            return

        x1, y1, x2, y2 = patient_bbox
        if x2 <= x1 or y2 <= y1: return

        patient_face_roi = rgb_frame[y1:y2, x1:x2].copy()

        if patient_face_roi.shape[0] < 1 or patient_face_roi.shape[1] < 1:
            return

        results = self.face_mesh.process(patient_face_roi)

        stroke_detected_this_frame = False
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw landmarks for visualization
                self.drawing_utils.draw_landmarks(
                    image=frame[y1:y2, x1:x2],
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=self.drawing_spec,
                    connection_drawing_spec=self.drawing_spec)

                left_mouth_corner = face_landmarks.landmark[61]
                right_mouth_corner = face_landmarks.landmark[291]

                # Check for vertical difference between mouth corners
                droop_difference = abs(left_mouth_corner.y - right_mouth_corner.y)
                if droop_difference > self.config.MOUTH_DROOP_THRESHOLD:
                    stroke_detected_this_frame = True
                    label = f"Stroke: Yes ({droop_difference:.3f})"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        self.alert_manager.check_and_trigger_timed_alert(
            frame, "stroke", stroke_detected_this_frame,
            self.config.STROKE_CONFIRMATION_SEC, "Potential Stroke", "stroke_alert",
            [frame[y1:y2, x1:x2]] if stroke_detected_this_frame else [],
            frame.shape[0] - 160
        )

    def process_knife_detection(self, frame: np.ndarray, rgb_frame: np.ndarray):
        """
        Detects knives in the frame. Implements a persistent alert for high-confidence
        detections to ensure the threat is continuously highlighted.
        """
        if not self.app_state["knife_detection_active"] or self.yolo_knife_model is None:
            self.knife_detected_at_high_conf = False  # Reset state if turned off
            return

        knife_detected_this_frame = False
        knife_images = []

        results = self.yolo_knife_model(rgb_frame, verbose=False, conf=self.config.YOLO_CONFIDENCE_THRESHOLD)
        for r in results:
            for box in r.boxes:
                class_name = self.yolo_knife_model.names[int(box.cls[0])]
                conf = float(box.conf[0])

                if 'knife' in class_name.lower():
                    knife_detected_this_frame = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # If confidence is high, trigger an immediate, persistent alert
                    if conf > self.config.KNIFE_HIGH_CONFIDENCE_THRESHOLD and not self.knife_detected_at_high_conf:
                        self.knife_detected_at_high_conf = True
                        message = f"DANGER: High-confidence Knife Detected! ({conf:.2f})"
                        print(message)
                        self.alert_manager.trigger_alarm(message, "knife_alert", [frame[y1:y2, x1:x2]])

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Handle the persistent alert state
        if self.knife_detected_at_high_conf:
            if not knife_detected_this_frame:
                # If knife is no longer seen, clear the persistent state
                self.knife_detected_at_high_conf = False
                print("Knife no longer detected. Persistent alert state cleared.")
            else:
                # Display a continuous warning on screen while the persistent alert is active
                cv2.putText(frame, "DANGER ALERT ACTIVE",
                            (10, frame.shape[0] - 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # For lower-confidence detections, use the standard timed alert
            self.alert_manager.check_and_trigger_timed_alert(
                frame, "knife", knife_detected_this_frame,
                self.config.KNIFE_CONFIRMATION_SEC, "Knife Detected", "knife_alert",
                knife_images if knife_images else [frame],
                frame.shape[0] - 190
            )

    def run(self):
        """The main application loop."""
        self.initialize()
        cap = cv2.VideoCapture(self.config.VIDEO_SOURCE)
        if not cap.isOpened():
            print(f"Error: Cannot open camera source: {self.config.VIDEO_SOURCE}")
            return

        window_name = "Patient Room Monitoring"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.ui_manager.handle_click)

        frame_count = 0
        person_count = 0
        patient_bbox = None

        try:
            while self.app_state["running"] and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("End of video stream or camera error.")
                    break

                processed_frame = cv2.resize(frame, (640, 480))

                # Handle bed area selection if needed
                if self.app_state["bed_exit_active"] and self.app_state["bed_roi"] is None:
                    roi = cv2.selectROI("Select Bed Area", processed_frame, fromCenter=False, showCrosshair=True)
                    if roi[2] > 0 and roi[3] > 0:
                        self.app_state["bed_roi"] = roi
                        print(f"Bed area selected at: {self.app_state['bed_roi']}")
                    else:
                        print("Bed area selection cancelled. Turning off Bed Exit mode.")
                        self.app_state["bed_exit_active"] = False
                    cv2.destroyWindow("Select Bed Area")

                processed_frame = self.process_frame_lighting(processed_frame)
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

                # --- Track-by-Detect Strategy ---
                # Run full, heavy detection periodically
                if frame_count % self.config.DETECTION_INTERVAL == 0:
                    person_count, patient_bbox = self.process_person_detection(processed_frame, rgb_frame)
                # Use the fast tracker for in-between frames
                else:
                    if self.patient_tracker:
                        success, box = self.patient_tracker.update(rgb_frame)
                        if success:
                            x, y, w, h = map(int, box)
                            patient_bbox = (x, y, x + w, y + h)
                            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(processed_frame, self.alert_manager.patient_name, (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        else:
                            self.patient_tracker = None
                            patient_bbox = None
                    # Redraw boxes for other people
                    for bbox in self.other_people_bboxes:
                        cv2.rectangle(processed_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 165, 255), 2)

                # --- Call all other detection modules on every frame ---
                self.process_cough_detection(processed_frame, rgb_frame, patient_bbox)
                self.process_gestures(processed_frame, rgb_frame)
                self.process_bed_exit(processed_frame, patient_bbox)
                self.process_stroke_detection_mediapipe(processed_frame, rgb_frame, patient_bbox)
                self.process_knife_detection(processed_frame, rgb_frame)

                # Draw UI elements
                processed_frame = self.ui_manager.draw_buttons(processed_frame)
                cv2.putText(processed_frame, f"People: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)
                cv2.imshow(window_name, processed_frame)

                frame_count += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.app_state["running"] = False
                    break
        finally:
            # Cleanup resources
            print("Shutting down...")
            self.app_state["running"] = False
            cap.release()
            cv2.destroyAllWindows()
            self.db_manager.close()


if __name__ == "__main__":
    config = Config()
    monitor = PatientMonitor(config)
    monitor.run()

