import mediapipe as mp
from ultralytics import YOLO
from typing import Dict, Any, Optional

# Assumes other modules are in the same directory
from config import Config
from alerts import AlertManager

# Import the 6 functional mixins
from core_utils import CoreUtilsMixin
from tracking_processor import TrackingMixin
from facial_processor import FacialMixin
from fatigue_processor import FatigueMixin
from gesture_processor import GestureMixin
from hazard_processor import HazardMixin


class PatientMonitorCore(CoreUtilsMixin, TrackingMixin, FacialMixin, FatigueMixin, GestureMixin, HazardMixin):
    """
    Handles all frame-by-frame processing and detection logic.
    This class is instantiated and used by the main application loop.
    Inherits cleanly separated functionality from 6 Mixins to prevent a monolithic file.
    """

    def __init__(self, config: Config, app_state: Dict[str, Any], alert_manager: AlertManager):
        self.config = config
        self.app_state = app_state
        self.alert_manager = alert_manager

        # Initialize MediaPipe and YOLO models
        self.hands = mp.solutions.hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()

        # UPDATED: refine_landmarks=True gives better eye/iris details for drowsiness
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.drawing_utils = mp.solutions.drawing_utils
        self.drawing_spec = self.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)

        # Load YOLO models (assuming they are already validated by the main app)
        self.yolo_model = YOLO(self.config.YOLO_MODEL_PATH) if self.config.YOLO_MODEL_PATH and self.app_state.get(
            'yolo_model_loaded') else None
        self.yolo_knife_model = YOLO(
            self.config.YOLO_KNIFE_MODEL_PATH) if self.config.YOLO_KNIFE_MODEL_PATH and self.app_state.get(
            'knife_detection_active') else None
        self.yolo_gun_model = YOLO(
            self.config.YOLO_GUN_MODEL_PATH) if self.config.YOLO_GUN_MODEL_PATH and self.app_state.get(
            'gun_detection_active') else None
        self.yolo_fall_model = YOLO(
            self.config.YOLO_FALL_MODEL_PATH) if self.config.YOLO_FALL_MODEL_PATH and self.app_state.get(
            'fall_detection_active') else None
        self.yolo_fire_model = YOLO(
            self.config.YOLO_FIRE_MODEL_PATH) if self.config.YOLO_FIRE_MODEL_PATH and self.app_state.get(
            'fire_detection_active') else None
        self.yolo_safety_model = YOLO(
            self.config.YOLO_SAFETY_MODEL_PATH) if self.config.YOLO_SAFETY_MODEL_PATH and self.app_state.get(
            'safety_detection_active') else None

        # State for cough detection
        self.previous_nose_z = None
        self.previous_shoulder_z = None
        self.cough_count = 0
        self.last_cough_time = 0
        self.cough_alert_sent = False

        # State variables for persistent display (Cough)
        self.last_face_landmarks = None
        self.last_mar_value = 0.0
        self.last_patient_roi_dims = None

        # State variables for Drowsiness detection
        self.last_ear_value = 0.0
        self.drowsiness_initial_timer = None
        self.drowsiness_confirmation_start_time = None
        self.drowsiness_prompt_spoken = False
        self.last_cancellation_time = 0.0

        # State for Pain Detection (Emotion AI)
        self.last_pain_score = 0.0
        self.pain_alert_started_time = None
        self.pain_landmarks = None
        self.pain_roi_dims = None

        # State for Emotion Detection (Happiness/Sadness)
        self.last_happiness_score = 0.0
        self.last_sadness_score = 0.0
        self.emotion_landmarks = None
        self.emotion_roi_dims = None
        self.last_sadness_trigger_time = 0.0
        self.is_companion_active = False

        # State for generic alert tracking (for persistent display across all timed alerts)
        self.generic_alert_status: Dict[str, Optional[Dict[str, Any]]] = {
            "unknown": None, "unidentified": None, "crowd": None,
            "bed_exit": None, "stroke": None, "knife": None, "gun": None,
            "fall_detection": None, "pain": None,
            "safety_violation": None, "fire": None,
            "happiness": None, "sadness": None
        }

        # State for persistent alerts
        self.knife_detected_at_high_conf = False
        self.gun_detected_at_high_conf = False

        # State for gestures
        self.gesture_actions = {
            0: ("Call Manager", "call_manager"),
            2: ("Need Water", "need_water"),
            3: ("Call Family", "call_family")
        }
        self.gesture_detected_time = {}

        # Tracking state from the main application
        self.known_face_encodings = self.app_state.get("known_face_encodings", [])
        self.known_face_names = self.app_state.get("known_face_names", [])
        self.patient_tracker = None
        self.patient_bbox = None
        self.other_people_bboxes = []
        self.last_detected_person_bbox = None
        self.tracking_start_time = 0