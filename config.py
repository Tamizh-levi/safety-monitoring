import os
import warnings
import logging

# Suppress all warnings for a cleaner console output
warnings.filterwarnings("ignore")


# --- Configuration ---
class Config:
    """
    Central configuration class for the Patient Monitoring System.
    This class holds all the static variables, paths, and thresholds
    to make the system easily configurable from one place.
    """
    # --- File and Directory Paths ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Use relative paths for portability. Place models in a 'models' directory.
    YOLO_MODEL_PATH = os.path.join(BASE_DIR, "yolo26n.pt")
    # Note: Use raw strings (r"...") for paths with backslashes on Windows
    # os.path.join with an absolute path as the second argument uses the absolute path directly
    YOLO_KNIFE_MODEL_PATH = os.path.join(BASE_DIR,
                                         r'C:\Users\sadik\knife detect\knife-detection.v4i.yolov8\runs\detect\train2\weights\best.pt')
    YOLO_GUN_MODEL_PATH = os.path.join(BASE_DIR, r"C:\Users\sadik\Downloads\gun-detection.v1i.yolov8\runs\detect\train\weights\best.pt")
    YOLO_FALL_MODEL_PATH = os.path.join(BASE_DIR, r"C:\Users\sadik\data\runs\train\fall_detection_model\weights\fall.pt")

    # --- MISSING PATHS ADDED HERE ---
    # Leave empty or set to None if you don't have the model file yet to avoid loading errors
    YOLO_SAFETY_MODEL_PATH = os.path.join(BASE_DIR, r"C:\Users\sadik\vest\runs\detect\train\weights\best.pt")
    YOLO_FIRE_MODEL_PATH = os.path.join(BASE_DIR, r"C:\Users\sadik\Downloads\fire.v1i.yolov8\runs\detect\fire_detection2\weights\best.pt")

    LOG_FILE_PATH = os.path.join(BASE_DIR, "patient_monitoring.log")
    REMINDER_AUDIO_DIR = os.path.join(BASE_DIR, "reminder_audio")
    INTRUDER_LOGS_DIR = os.path.join(BASE_DIR, "intruder_logs")

    # --- Alarm Sound Asset Paths ---
    ALARM_SOUNDS_DIR = os.path.join(BASE_DIR, "alarm_sounds")
    ALARM_SOUNDS = {
        "call_manager": os.path.join(ALARM_SOUNDS_DIR, "call_nurse.mp3"),
        "need_water": os.path.join(ALARM_SOUNDS_DIR, "need_water.mp3"),
        "call_family": os.path.join(ALARM_SOUNDS_DIR, "call_family.mp3"),
        "cancel_request": os.path.join(ALARM_SOUNDS_DIR, "cancel_request.mp3"),
        "unknown_alert": os.path.join(ALARM_SOUNDS_DIR, "unknown_alert.mp3"),
        "unidentified_alert": os.path.join(ALARM_SOUNDS_DIR, "unknown_alert.mp3"),
        "crowd_alert": os.path.join(ALARM_SOUNDS_DIR, "crowd_alert.mp3"),
        "bed_exit_alert": os.path.join(ALARM_SOUNDS_DIR, "fall_alert.mp3"),
        "stroke_alert": os.path.join(ALARM_SOUNDS_DIR, "fall_alert.mp3"),
        "fall_alert": os.path.join(ALARM_SOUNDS_DIR, "fall_alert.mp3"),
        "knife_alert": os.path.join(ALARM_SOUNDS_DIR, "knife_alert.mp3"),
        "gun_alert": os.path.join(ALARM_SOUNDS_DIR, "knife_alert.mp3"),
        "cough_alert": os.path.join(ALARM_SOUNDS_DIR, "cough_alert.mp3"),
        "drowsiness_alert": os.path.join(ALARM_SOUNDS_DIR, "fall_alert.mp3"),
        "pain_alert": os.path.join(ALARM_SOUNDS_DIR, "fall_alert.mp3"),
        "happiness_alert": os.path.join(ALARM_SOUNDS_DIR, "positive_alert.mp3"),
        "sadness_alert": os.path.join(ALARM_SOUNDS_DIR, "negative_alert.mp3"),
        "help_request": os.path.join(ALARM_SOUNDS_DIR, "call_nurse.mp3"),
        "thank_you_response": os.path.join(ALARM_SOUNDS_DIR, "positive_alert.mp3"),
        "safety_alert": os.path.join(ALARM_SOUNDS_DIR, "negative_alert.mp3"),  # Added
        "fire_alert": os.path.join(ALARM_SOUNDS_DIR, "negative_alert.mp3")  # Added
    }

    # --- System & Patient Information ---
    VIDEO_SOURCE = 0
    MONITORED_PATIENT_ID = "PAT001"

    # --- Worker/Manager Information ---
    MONITORED_WORKER_ID = "WRK888"
    MONITORED_WORKER_NAME = "Supervisor Smith"

    # --- Phone numbers for alerts ---
    RECEIVER_PHONE_WHATSAPP = "+919345531046"
    RECEIVER_PHONE_SMS = "+919345531046"

    # --- Twilio Configuration ---
    TWILIO_ACCOUNT_SID = "C79862b8222770870cce96bf9f2b8c84f"
    TWILIO_AUTH_TOKEN = "b4585fa73e2af436da4d4c46ef8e312f"
    TWILIO_PHONE_NUMBER = "+16282655983"

    # --- Database Configuration ---
    MONGO_URI = "mongodb://localhost:27017/"
    MONGO_DB_NAME = "patient_monitoring"
    ALERTS_COLLECTION_NAME = "alerts"
    PATIENTS_COLLECTION_NAME = "patients"
    SCHEDULES_COLLECTION_NAME = "schedules"

    # --- Detection Thresholds & Timings ---
    YOLO_CONFIDENCE_THRESHOLD = 0.7
    FACE_RECOGNITION_TOLERANCE = 0.5
    KNIFE_HIGH_CONFIDENCE_THRESHOLD = 1.0
    GUN_HIGH_CONFIDENCE_THRESHOLD = 1.0
    FALL_CONFIDENCE_THRESHOLD = 0.70
    CROWD_THRESHOLD = 4
    ALERT_CONFIRMATION_SEC = 3
    UNIDENTIFIED_CONFIRMATION_SEC = 2
    LOW_LIGHT_THRESHOLD = 80
    BED_EXIT_CONFIRMATION_SEC = 5
    FALL_CONFIRMATION_SEC = 2
    GESTURE_CONFIRMATION_SEC = 2
    STROKE_CONFIRMATION_SEC = 4
    MOUTH_DROOP_THRESHOLD = 0.03
    KNIFE_CONFIRMATION_SEC = 2
    GUN_CONFIRMATION_SEC = 2
    COUGH_CONFIRMATION_SEC = 2
    MOUTH_ASPECT_RATIO_THRESHOLD = 0.2

    # --- Safety Gear Settings (ADDED) ---
    # List of classes your safety model detects (must match the model's class names)
    YOLO_SAFETY_CLASSES = ["helmet", "vest", "goggles", "mask"]

    # --- Fire Detection Settings (ADDED) ---
    FIRE_CONFIDENCE_THRESHOLD = 0.4

    FIRE_CLASSES = ["fire"]
    FIRE_INSTANT_ALERT = True
    FIRE_CONFIRMATION_SEC = 10

    HEAD_FORWARD_THRESHOLD = 0.03
    COUGH_COUNT_THRESHOLD = 1
    COUGH_RESET_SEC = 60

    DASHBOARD_URL = "http://10.58.95.32:5000/api/alerts"# Set to your dashboard URL
    ENABLE_PORT_NOTIFICATIONS = True
    NOTIFICATION_HOST = '127.0.0.1'
    NOTIFICATION_PORT = 2000

    # --- Drowsiness/Fatigue Detection Settings ---
    EYE_ASPECT_RATIO_THRESHOLD = 0.25
    INITIAL_DROWSINESS_SEC = 4
    GESTURE_CONFIRMATION_TIMEOUT = 15
    DROWSINESS_COOLDOWN_SEC = 8
    SLEEP_START_TIME = "22:00"
    SLEEP_END_TIME = "06:00"

    # --- Pain and Discomfort Assessment Settings ---
    PAIN_CONFIRMATION_SEC = 3
    FACIAL_ASYMMETRY_THRESHOLD = 0.02
    PAIN_INTENSITY_THRESHOLD = 0.1
    PAIN_INSTANT_ALERT = True  # Added

    EYE_SQUEEZE_THRESHOLD = 0.03
    MOUTH_ASYMMETRY_THRESHOLD = 0.035
    EYEBROW_MOVEMENT_DEVIATION = 0.03
    CHEEK_RAISE_DISTANCE_THRESHOLD = 0.04

    # --- Happiness/Sadness Detection Settings ---
    EMOTION_DETECTION_CONFIRMATION_SEC = 4
    HAPPINESS_THRESHOLD = 0.17054
    SADNESS_THRESHOLD = 0.001111

    # --- Emotional Companion Settings ---
    COMPANION_TIMEOUT_SEC = 45
    COMPANION_COOLDOWN_SEC = 120
    COMPANION_PROACTIVE_THRESHOLD = 0.15
    COMPANION_TRIGGER_WORDS = ["medimind", "companion", "hello"]

    # --- Ollama Configuration ---
    OLLAMA_API_URL = 'http://localhost:11434/api/generate'
    OLLAMA_MODEL_NAME = 'llama3'

    # --- Performance Settings ---
    DETECTION_INTERVAL = 3


    # --- Voice Recognition Settings ---
    VOICE_AMBIENT_ADJUST_DUR = 4
    VOICE_PAUSE_THRESHOLD = 1.5
    VOICE_TIMEOUT = 8
    VOICE_PHRASE_LIMIT = 10