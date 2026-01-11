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
    YOLO_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
    YOLO_KNIFE_MODEL_PATH = os.path.join(BASE_DIR,
                                         r'C:\Users\sadik\knife detect\knife-detection.v4i.yolov8\runs\detect\train2\weights\best.pt')
    YOLO_GUN_MODEL_PATH = os.path.join(BASE_DIR, "gun.pt")
    YOLO_FALL_MODEL_PATH = os.path.join(BASE_DIR, r"C:\fall detection.v1i.yolov8\runs\detect\train\weights\fall.pt")

    # --- FIRE/SMOKE DETECTION CONFIGURATION (NEW) ---
    YOLO_FIRE_MODEL_PATH = os.path.join(BASE_DIR, r'C:\Users\sadik\Downloads\fire.v1i.yolov8 (1)\test\runs\detect\fire_smoke_detect4\weights\best.pt')
    FIRE_CLASSES = ['fire']
    FIRE_CONFIRMATION_SEC = 1 # Immediate alert for fire/smoke
    FIRE_CONFIDENCE_THRESHOLD = 0.10 # NEW: Specific threshold for fire detection
    FIRE_INSTANT_ALERT = False # New flag for instant fire alert
    # --- END FIRE/SMOKE CONFIGURATION ---

    # --- NEW SAFETY GEAR DETECTION CONFIGURATION ---
    YOLO_SAFETY_MODEL_PATH = os.path.join(BASE_DIR, r"C:\Users\sadik\Downloads\fire.v1i.yolov8\civi\helmet.pt")
    YOLO_SAFETY_CLASSES = ['Safety-Helmet', 'Reflective-Jacket']
    # --- END NEW CONFIGURATION ---

    LOG_FILE_PATH = os.path.join(BASE_DIR, "patient_monitoring.log")
    REMINDER_AUDIO_DIR = os.path.join(BASE_DIR, "reminder_audio")
    INTRUDER_LOGS_DIR = os.path.join(BASE_DIR, "intruder_logs")

    # --- Alarm Sound Asset Paths ---
    ALARM_SOUNDS_DIR = os.path.join(BASE_DIR, "alarm_sounds")
    ALARM_SOUNDS = {
        "call_manager": os.path.join(ALARM_SOUNDS_DIR, "call_nurse.mp3"),  # UPDATED: Renamed key to call_manager
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
        "gun_alert": os.path.join(ALARM_SOUNDS_DIR, "knife_alert.mp3"),  # Reusing knife sound
        "cough_alert": os.path.join(ALARM_SOUNDS_DIR, "cough_alert.mp3"),
        "drowsiness_alert": os.path.join(ALARM_SOUNDS_DIR, "fall_alert.mp3"),  # Reusing fall alert for drowsiness
        "pain_alert": os.path.join(ALARM_SOUNDS_DIR, "fall_alert.mp3"),  # Pain alert (reusing fall alert sound)
        "happiness_alert": os.path.join(ALARM_SOUNDS_DIR, "positive_alert.mp3"),  # ADDED: Happiness alert
        "sadness_alert": os.path.join(ALARM_SOUNDS_DIR, "negative_alert.mp3"),  # ADDED: Sadness alert
        "safety_alert": os.path.join(ALARM_SOUNDS_DIR, "knife_alert.mp3"), # Reusing knife alert for safety violation
        "fire_alert": os.path.join(ALARM_SOUNDS_DIR, "knife_alert.mp3"), # ADDED: Using knife alert sound for fire/smoke
        "help_request": os.path.join(ALARM_SOUNDS_DIR, "call_nurse.mp3"),  # NEW: Reusing nurse sound for help
        "thank_you_response": os.path.join(ALARM_SOUNDS_DIR, "positive_alert.mp3")
        # NEW: Reusing positive sound for thanks
    }

    # --- System & Patient Information ---
    VIDEO_SOURCE = 0  # Use 0 for default webcam
    MONITORED_PATIENT_ID = "PAT001"

    # --- Worker/Manager Information (NEW) ---
    MONITORED_WORKER_ID = "WRK888"  # Mock worker ID for "help" requests
    MONITORED_WORKER_NAME = "Supervisor Smith"  # Mock worker name

    # --- DASHBOARD CONFIGURATION (NEW) ---
    # UPDATED: Use the root path only, which the debug Flask app handles with a POST method.
    DASHBOARD_URL = ("http://10.84.156.32:5000/api/fire_alert")

    # --- Phone numbers for alerts (E.164 format, e.g., +1234567890) ---
    RECEIVER_PHONE_WHATSAPP = "+919345531046"
    RECEIVER_PHONE_SMS = "+917448917940"

    # --- Twilio Configuration (Add your credentials here) ---
    TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
    TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
    TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

    # --- Port Notification Configuration (NEW) ---
    NOTIFICATION_HOST = "127.0.0.1"  # IP address of the listener
    NOTIFICATION_PORT = 2000         # Port to send notifications to
    ENABLE_PORT_NOTIFICATIONS = True # Master switch for this feature

    # --- Database Configuration ---
    MONGO_URI = "mongodb://localhost:27017/"
    MONGO_DB_NAME = "patient_monitoring"
    ALERTS_COLLECTION_NAME = "alerts"
    PATIENTS_COLLECTION_NAME = "patients"
    SCHEDULES_COLLECTION_NAME = "schedules"

    # --- Detection Thresholds & Timings ---
    YOLO_CONFIDENCE_THRESHOLD = 0.7
    FACE_RECOGNITION_TOLERANCE = 0.5
    KNIFE_HIGH_CONFIDENCE_THRESHOLD = 0.70  # Immediate, persistent knife alert
    GUN_HIGH_CONFIDENCE_THRESHOLD = 0.70  # Immediate, persistent gun alert
    FALL_CONFIDENCE_THRESHOLD = 0.50
    CROWD_THRESHOLD = 4  # Number of people considered a crowd
    ALERT_CONFIRMATION_SEC = 3
    UNIDENTIFIED_CONFIRMATION_SEC = 2
    LOW_LIGHT_THRESHOLD = 80
    BED_EXIT_CONFIRMATION_SEC = 5
    FALL_CONFIRMATION_SEC = 5
    GESTURE_CONFIRMATION_SEC = 2
    STROKE_CONFIRMATION_SEC = 4
    MOUTH_DROOP_THRESHOLD = 0.03
    KNIFE_CONFIRMATION_SEC = 2
    GUN_CONFIRMATION_SEC = 2
    COUGH_CONFIRMATION_SEC = 2
    MOUTH_ASPECT_RATIO_THRESHOLD = 0.2

    HEAD_FORWARD_THRESHOLD = 0.03
    COUGH_COUNT_THRESHOLD = 1
    COUGH_RESET_SEC = 60

    # --- Drowsiness/Fatigue Detection Settings ---
    EYE_ASPECT_RATIO_THRESHOLD = 0.25  # Threshold for eye closure
    INITIAL_DROWSINESS_SEC = 4  # Duration eyes must be closed before prompt
    GESTURE_CONFIRMATION_TIMEOUT = 15  # Time patient has to respond to voice prompt
    DROWSINESS_COOLDOWN_SEC = 8  # Time to suppress re-triggering after manual cancellation
    SLEEP_START_TIME = "22:00"  # Start of the "safe sleep" window (24hr format)
    SLEEP_END_TIME = "06:00"  # End of the "safe sleep" window

    # --- Pain and Discomfort Assessment Settings (Emotion AI) ---
    PAIN_CONFIRMATION_SEC = 1  # MODIFIED: Duration facial discomfort must be active before alerting (1 sec)
    PAIN_INSTANT_ALERT = True # NEW: If True, t
    # riggers alarm instantly if score > 0.3
    FACIAL_ASYMMETRY_THRESHOLD = 0.02  # Threshold for vertical difference between key landmarks (e.g., mouth/eye)
    PAIN_INTENSITY_THRESHOLD = 0.1
    # Overall intensity threshold (will be higher now due to more metrics)

    # NEW PAIN METRICS THRESHOLDS
    # Lower than this value indicates significant eyebrow squeeze/furrowing
    EYEBROW_SQUEEZE_THRESHOLD = 0.03
    # Higher than this value indicates significant mouth asymmetry/grimace
    MOUTH_ASYMMETRY_THRESHOLD = 0.035
    # Higher deviation from baseline indicates significant eyebrow movement (raising/furrowing)
    EYEBROW_MOVEMENT_DEVIATION = 0.03
    # Lower than this value indicates significant cheek raise (tightening)
    CHEEK_RAISE_DISTANCE_THRESHOLD = 0.04

    # --- Happiness/Sadness Detection Settings ---
    EMOTION_DETECTION_CONFIRMATION_SEC = 4  # Duration emotion must be held to trigger alert/log
    HAPPINESS_THRESHOLD = 0.17054  # ADJUSTED: Drastically lowered for sensitivity
    SADNESS_THRESHOLD = 0.001111

    # Heuristic score for a significant frown/downturned corners (higher is sadder)

    # --- Emotional Companion Settings --- (NEW)
    COMPANION_TIMEOUT_SEC = 45  # Max duration for a single conversation
    COMPANION_COOLDOWN_SEC = 120  # Minimum time between proactive engagements
    # Normalized pain score threshold (0.0 to 1.0) to trigger proactive companion engagement
    # Set below PAIN_INTENSITY_THRESHOLD (0.04) for mild distress/mood detection
    COMPANION_PROACTIVE_THRESHOLD = 0.15
    COMPANION_TRIGGER_WORDS = ["medimind", "companion", "hello"]  # Words patient can use to start conversation

    # --- Ollama Configuration (NEW) ---
    OLLAMA_API_URL = 'http://localhost:11434/api/generate'
    OLLAMA_MODEL_NAME = 'llama3'

    # --- Performance Settings ---
    DETECTION_INTERVAL = 15  # Run full detection every N frames

    # --- Voice Recognition Settings ---
    VOICE_AMBIENT_ADJUST_DUR = 4
    VOICE_PAUSE_THRESHOLD = 1.5
    VOICE_TIMEOUT = 8
    VOICE_PHRASE_LIMIT = 10