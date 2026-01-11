import cv2
import numpy as np
import face_recognition
from ultralytics import YOLO
import mediapipe as mp
import time
import math
import threading
from typing import List, Tuple, Dict, Any, Optional

# Assumes other modules are in the same directory
from config import Config
from alerts import AlertManager


class PatientMonitorCore:
    """
    Handles all frame-by-frame processing and detection logic.
    This class is instantiated and used by the main application loop.
    """

    def __init__(self, config: Config, app_state: Dict[str, Any], alert_manager: AlertManager):
        self.config = config
        self.app_state = app_state
        self.alert_manager = alert_manager

        # Initialize MediaPipe and YOLO models
        self.hands = mp.solutions.hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5,
                                                         min_tracking_confidence=0.5)
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
        # NEW: Load Fire Model
        self.yolo_fire_model = YOLO(
            self.config.YOLO_FIRE_MODEL_PATH) if self.config.YOLO_FIRE_MODEL_PATH and self.app_state.get(
            'fire_detection_active') else None
        # Load Safety Gear Model (previously added)
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
        self.last_patient_roi_dims = None  # To store (x1, y1, x2, y2) for persistent drawing

        # New state variables for Drowsiness detection (Stage 1: Initial Detection & Stage 2: Confirmation)
        self.last_ear_value = 0.0
        self.drowsiness_initial_timer = None  # Timer for the first 4 seconds of closure (Stage 1)
        self.drowsiness_confirmation_start_time = None  # Time when the voice prompt starts (Stage 2)
        self.drowsiness_prompt_spoken = False  # Flag to ensure voice prompt only plays once
        self.last_cancellation_time = 0.0  # Tracks when the drowsiness prompt was successfully cancelled

        # State for Pain Detection (Emotion AI)
        self.last_pain_score = 0.0
        self.pain_alert_started_time = None
        self.pain_landmarks = None  # ADDED: Stores the face landmarks for persistent drawing
        self.pain_roi_dims = None  # ADDED: Stores (x1, y1, x2, y2) for persistent drawing

        # State for Emotion Detection (Happiness/Sadness)
        self.last_happiness_score = 0.0
        self.last_sadness_score = 0.0
        self.emotion_landmarks = None  # ADDED: Stores the face landmarks for persistent drawing
        self.emotion_roi_dims = None  # ADDED: Stores (x1, y1, x2, y2) for persistent drawing
        self.last_sadness_trigger_time = 0.0  # ADDED: Tracks last time sadness triggered chat
        self.is_companion_active = False  # ADDED: Flag to suppress new chats if one is ongoing (THIS IS THE FLAG USED)

        # State for generic alert tracking (for persistent display across all timed alerts)
        # Structure: {'alert_type': {'condition': bool, 'timestamp': float, 'message': str, 'conf_sec': int, 'y_pos': int}}
        # ADDED: "fire" and "safety_violation"
        self.generic_alert_status: Dict[str, Optional[Dict[str, Any]]] = {
            "unknown": None, "unidentified": None, "crowd": None,
            "bed_exit": None, "stroke": None, "knife": None, "gun": None,
            "fall_detection": None, "pain": None,
            "safety_violation": None, "fire": None,  # ADDED: fire
            "happiness": None, "sadness": None
        }

        # State for persistent alerts
        self.knife_detected_at_high_conf = False
        self.gun_detected_at_high_conf = False

        # State for gestures
        # UPDATED: Changed the alert message and sound key for gesture 0
        self.gesture_actions = {
            0: ("Call Manager", "call_manager"),  # UPDATED: Message and key changed from "Call Nurse"
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

    def _calculate_ear(self, face_landmarks) -> float:
        """Calculates the Eye Aspect Ratio (EAR) for both eyes."""
        # Indices for the left eye (MediaPipe indices)
        # P1=33(outer corner), P2=133(inner corner)
        P1 = face_landmarks.landmark[33]
        P2 = face_landmarks.landmark[133]
        # P3, P4, P5, P6 are vertical points
        P3 = face_landmarks.landmark[160]
        P4 = face_landmarks.landmark[158]
        P5 = face_landmarks.landmark[144]
        P6 = face_landmarks.landmark[153]

        # Indices for the right eye (MediaPipe indices)
        P7 = face_landmarks.landmark[362]
        P8 = face_landmarks.landmark[263]
        P9 = face_landmarks.landmark[387]
        P10 = face_landmarks.landmark[385]
        P11 = face_landmarks.landmark[373]
        P12 = face_landmarks.landmark[380]

        # Calculate distances for left eye: (vertical) / (horizontal)
        left_vertical_dist = math.hypot(P3.x - P5.x, P3.y - P5.y) + math.hypot(P4.x - P6.x, P4.y - P6.y)
        left_horizontal_dist = math.hypot(P1.x - P2.x, P1.y - P2.y)
        left_ear = (left_vertical_dist) / (2.0 * left_horizontal_dist) if left_horizontal_dist > 0 else 0.0

        # Calculate distances for right eye: (vertical) / (horizontal)
        right_vertical_dist = math.hypot(P9.x - P11.x, P9.y - P11.y) + math.hypot(P10.x - P12.x, P10.y - P12.y)
        right_horizontal_dist = math.hypot(P7.x - P8.x, P7.y - P8.y)
        right_ear = (right_vertical_dist) / (2.0 * right_horizontal_dist) if right_horizontal_dist > 0 else 0.0

        # Average the two EARs
        return (left_ear + right_ear) / 2.0

    def _calculate_emotion_metrics(self, face_landmarks) -> Tuple[float, float]:
        """
        Calculates normalized metrics for basic emotion detection (Happiness/Sadness).

        Returns: (Normalized Happiness Score, Normalized Sadness Score)
        """

        p_upper_lip = face_landmarks.landmark[13]
        p_lower_lip = face_landmarks.landmark[14]
        p_left_corner = face_landmarks.landmark[61]
        p_right_corner = face_landmarks.landmark[291]

        # Use midpoint of the upper/lower lip centers as a stable horizontal reference for sadness
        p_mouth_y_ref = (p_upper_lip.y + p_lower_lip.y) / 2

        p_ref = face_landmarks.landmark[2]  # Reference point near nose/cheek

        y_coords = [lm.y for lm in face_landmarks.landmark]
        face_height = max(y_coords) - min(y_coords) if max(y_coords) > min(y_coords) else 1.0

        if face_height == 0 or face_height < 0.01:
            return 0.0, 0.0

        # --- FIX: Initialize scores before calculation to prevent UnboundLocalError ---
        happiness_score = 0.0
        sadness_score = 0.0
        # --------------------------------------------------------------------------

        # --- 1. Happiness Score (Smile Heuristic) ---
        # Measure how much the mouth corners are pulled up relative to the nose/cheek reference point (p_ref).

        # Calculate the average vertical distance between the reference point (p_ref) and the mouth corners.
        left_corner_vertical_movement = (p_ref.y - p_left_corner.y) / face_height
        right_corner_vertical_movement = (p_ref.y - p_right_corner.y) / face_height

        avg_vertical_movement = (left_corner_vertical_movement + right_corner_vertical_movement) / 2.0

        # FIXED: Use the absolute value of the vertical movement (as the negative/positive sign was inconsistent/inverted)
        happiness_score = abs(avg_vertical_movement)

        # --- 2. Sadness Score (Frown/Droop Heuristic) ---
        # Measure how much the mouth corners are pulled DOWN relative to the central horizontal line of the mouth (p_mouth_y_ref).

        # Calculate the vertical distance the corner is BELOW the horizontal mouth center line.
        # (Y is higher for points lower on the screen)
        left_corner_droop = (p_left_corner.y - p_mouth_y_ref) / face_height
        right_corner_droop = (p_right_corner.y - p_mouth_y_ref) / face_height

        # Droop is positive if the corner is below the mouth line, which indicates a frown.
        sadness_score = max(left_corner_droop, right_corner_droop)  # Use max droop for sadness

        return happiness_score, sadness_score

    def _calculate_pain_metrics(self, face_landmarks) -> Tuple[float, float]:
        """
        Calculates normalized metrics for pain detection (Eyebrow Squeeze and Mouth Asymmetry).
        """
        # Landmarks for Eyebrow Squeeze (Vertical distance from eyebrow inner point to eye upper lid center)
        # Left: Eyebrow(70) to Eye Top(159)
        # Right: Eyebrow(300) to Eye Top(386)

        # Calculate Eyebrow Squeeze (Avg of Left and Right)
        left_eye_top = face_landmarks.landmark[159]
        left_eyebrow_inner = face_landmarks.landmark[70]

        right_eye_top = face_landmarks.landmark[386]
        right_eyebrow_inner = face_landmarks.landmark[300]

        # Vertical distance (y_eyebrow - y_eye). In screen coords, low Y is up. High distance = open eyes.
        left_squeeze_dist = abs(left_eyebrow_inner.y - left_eye_top.y)
        right_squeeze_dist = abs(right_eyebrow_inner.y - right_eye_top.y)

        avg_squeeze_dist = (left_squeeze_dist + right_squeeze_dist) / 2.0

        # Calculate Mouth Asymmetry (Vertical difference between the two mouth corners)
        # Left Corner: 61, Right Corner: 291
        left_mouth_corner = face_landmarks.landmark[61]
        right_mouth_corner = face_landmarks.landmark[291]

        # Vertical difference (absolute y difference)
        mouth_asymmetry_y = abs(left_mouth_corner.y - right_mouth_corner.y)

        # Normalization Factor (Vertical size of the face)
        y_coords = [lm.y for lm in face_landmarks.landmark]
        face_height = max(y_coords) - min(y_coords)

        if face_height == 0 or face_height < 0.01:  # Avoid division by zero and extremely small faces
            return 0.0, 0.0

        # Normalized values (0 to 1, relative to face height)
        normalized_squeeze = avg_squeeze_dist / face_height
        normalized_asymmetry = mouth_asymmetry_y / face_height

        return normalized_squeeze, normalized_asymmetry

    def process_frame_lighting(self, frame: np.ndarray) -> np.ndarray:
        """Enhances the frame if low light is detected (Night Mode)."""
        if np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) < self.config.LOW_LIGHT_THRESHOLD:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced_lab = cv2.merge((cl, a, b))
            enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            cv2.putText(enhanced_frame, "Night Mode", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            return enhanced_frame
        return frame

    def process_person_detection(self, frame: np.ndarray, rgb_frame: np.ndarray) -> Tuple[
        int, Optional[Tuple[int, int, int, int]]]:
        """
        Detects, tracks, and identifies all persons in the frame.
        This is the core logic for unknown, unidentified, and crowd alerts.
        """
        if self.yolo_model is None:
            return 0, None

        patient_identified_this_frame = False
        all_bboxes = []
        patient_bbox = None

        # Run YOLO detection for all objects first
        results = self.yolo_model(rgb_frame, verbose=False, conf=self.config.YOLO_CONFIDENCE_THRESHOLD)

        person_count = 0
        unknown_face_detected = False
        unidentified_person_present = False
        intruder_rois = []

        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == 0:  # Class 0 is 'person'
                    person_count += 1
                    bbox = tuple(map(int, box.xyxy[0]))
                    all_bboxes.append(bbox)

        if person_count == 0:
            self.patient_tracker = None
            self.patient_bbox = None
            self.last_detected_person_bbox = None

        if self.patient_tracker is not None:
            success, box = self.patient_tracker.update(rgb_frame)
            if success:
                x, y, w, h = map(int, box)
                self.patient_bbox = (x, y, x + w, y + h)
                patient_identified_this_frame = True
            else:
                self.patient_tracker = None
                self.patient_bbox = None

        self.other_people_bboxes.clear()
        temp_patient_bbox = None
        patient_found_by_face = False

        for bbox in all_bboxes:
            x1, y1, x2, y2 = bbox
            person_roi = rgb_frame[y1:y2, x1:x2].copy()

            if person_roi.shape[0] < 20 or person_roi.shape[1] < 20: continue

            face_locations = face_recognition.face_locations(person_roi)
            if face_locations:
                face_encodings = face_recognition.face_encodings(person_roi, face_locations)
                if not face_encodings:
                    unidentified_person_present = True
                    self.other_people_bboxes.append(bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    intruder_rois.append(frame[y1:y2, x1:x2])
                    continue

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

        if patient_found_by_face:
            self.patient_bbox = temp_patient_bbox
            self.last_detected_person_bbox = temp_patient_bbox
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

        # Update generic alert status based on this detection frame
        self._update_generic_alert_status(frame, "unknown", unknown_face_detected, self.config.ALERT_CONFIRMATION_SEC,
                                          "Other Person Detected", "unknown_alert", intruder_rois, frame.shape[0] - 10)
        self._update_generic_alert_status(frame, "unidentified", unidentified_person_present,
                                          self.config.UNIDENTIFIED_CONFIRMATION_SEC,
                                          "Unidentified Person", "unidentified_alert", intruder_rois,
                                          frame.shape[0] - 40)

        is_crowd = person_count > self.config.CROWD_THRESHOLD
        self._update_generic_alert_status(frame, "crowd", is_crowd, self.config.ALERT_CONFIRMATION_SEC,
                                          f"Crowd ({person_count})", "crowd_alert", [frame], frame.shape[0] - 70)

        # Original call to check_and_trigger_timed_alert is now replaced by _update_generic_alert_status and draw_generic_alert_overlays
        # We still need to call the alert manager for the final trigger, but drawing happens later.
        if self.app_state["unknown_person_active"]:
            self.alert_manager.check_and_trigger_timed_alert(frame, "unknown", unknown_face_detected,
                                                             self.config.ALERT_CONFIRMATION_SEC,
                                                             "Other Person Detected",
                                                             "unknown_alert", intruder_rois, 0)  # y_pos is ignored
        if self.app_state["unidentified_person_active"]:
            self.alert_manager.check_and_trigger_timed_alert(frame, "unidentified", unidentified_person_present,
                                                             self.config.UNIDENTIFIED_CONFIRMATION_SEC,
                                                             "Unidentified Person", "unidentified_alert", intruder_rois,
                                                             0)  # y_pos is ignored
        if self.app_state["crowd_alert_active"]:
            self.alert_manager.check_and_trigger_timed_alert(frame, "crowd", is_crowd,
                                                             self.config.ALERT_CONFIRMATION_SEC,
                                                             f"Crowd ({person_count})", "crowd_alert", [frame],
                                                             0)  # y_pos is ignored

        return person_count, self.patient_bbox

    def _update_generic_alert_status(self, frame: np.ndarray, alert_type: str, condition: bool, conf_sec: int,
                                     message: str, sound_key: str, images: Optional[List[np.ndarray]], y_pos: int):
        """Updates internal state for persistent display of generic timed alerts."""
        if not self.app_state.get(f"{alert_type}_active", False) and alert_type not in ["happiness", "sadness",
                                                                                        "safety_violation", "fire"]:
            self.generic_alert_status[alert_type] = None
            return

        # Special handling for happiness/sadness as they are part of a single toggle 'emotion_detection'
        if alert_type in ["happiness", "sadness"] and not self.app_state.get("emotion_detection_active", False):
            self.generic_alert_status[alert_type] = None
            return

        # Special handling for fire detection if it's not active
        if alert_type == "fire" and not self.app_state.get("fire_detection_active", False):
            self.generic_alert_status[alert_type] = None
            return

        # Special handling for safety detection if it's not active
        if alert_type == "safety_violation" and not self.app_state.get("safety_detection_active", False):
            self.generic_alert_status[alert_type] = None
            return

        current_time = time.time()
        status = self.generic_alert_status.get(alert_type)

        if condition:
            if status is None:
                # Start the timer and store metadata
                self.generic_alert_status[alert_type] = {
                    "timestamp": current_time,
                    "message": message,
                    "conf_sec": conf_sec,
                    "y_pos": y_pos
                }
            # The core logic for triggering the final alert is still handled by the alert manager,
            # but we track the status here for drawing.
        else:
            # Condition broken, reset status
            self.generic_alert_status[alert_type] = None

    def draw_generic_alert_overlays(self, frame: np.ndarray):
        """LIGHT DRAWING: Draws all active generic alert countdowns on every frame."""
        for alert_type, status in self.generic_alert_status.items():
            if status is not None and alert_type not in ["happiness", "sadness"]:  # Filter out explicit emotion alerts
                # Check if the alert is actually active (not yet triggered) by checking the alert manager state
                is_sent = self.alert_manager.alert_sent_flags.get(alert_type, True)

                if not is_sent:
                    elapsed = time.time() - status["timestamp"]
                    countdown = status["conf_sec"] - elapsed

                    # Only draw if still counting down
                    if countdown > 0:
                        cv2.putText(frame, f"{status['message']} in: {int(countdown) + 1}s",
                                    (10, status['y_pos']), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def process_fire_detection(self, frame: np.ndarray, rgb_frame: np.ndarray):
        """Detects fire or smoke in the frame using a dedicated YOLO model."""
        if not self.app_state["fire_detection_active"] or self.yolo_fire_model is None:
            self.generic_alert_status["fire"] = None
            self.alert_manager.alert_timers.pop("fire", None)
            self.alert_manager.alert_sent_flags.pop("fire", None)
            return

        fire_detected_this_frame = False
        fire_images = []

        # Run YOLO detection using the configurable FIRE_CONFIDENCE_THRESHOLD
        results = self.yolo_fire_model.predict(
            rgb_frame,
            imgsz=640,
            conf=self.config.FIRE_CONFIDENCE_THRESHOLD,  # Use the specific confidence threshold
            verbose=False
        )

        # Convert configured FIRE_CLASSES to lowercase for case-insensitive matching
        lower_fire_classes = [c.lower() for c in self.config.FIRE_CLASSES]

        for result in results:
            for box in result.boxes:
                # Extract class ID, confidence, and bounding box coordinates
                class_id = int(box.cls[0].item())
                conf = box.conf[0].item()

                # Check if class ID is valid for the model's names list
                if class_id < len(self.yolo_fire_model.names):
                    label = self.yolo_fire_model.names[class_id]
                else:
                    label = "Unknown Class"

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Check if detected label matches configured FIRE_CLASSES (case-insensitive)
                if label.lower() in lower_fire_classes:
                    fire_detected_this_frame = True
                    color = (0, 0, 255)  # Red box for fire/smoke
                else:
                    # Draw other detected classes (if any, in green/default)
                    color = (0, 255, 0)

                # Draw detection box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label_text = f"{label} {conf:.2f}"
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if fire_detected_this_frame:
                    fire_images.append(frame[y1:y2, x1:x2].copy())

        # Logic for INSTANT ALERT (added in a previous step)
        if self.app_state.get("fire_detection_active", False) and fire_detected_this_frame and getattr(self.config,
                                                                                                       'FIRE_INSTANT_ALERT',
                                                                                                       False):
            # Check if alert hasn't been sent recently (using alert_sent_flags for a quick check)
            if not self.alert_manager.alert_sent_flags.get("fire", False):
                self.alert_manager.trigger_alarm(
                    "CRITICAL: Fire/Smoke Detected! Immediate Action Required (INSTANT ALERT).",
                    "fire_alert", fire_images if fire_images else [frame],
                    is_dashboard_alert=True, alert_type="fire_alert"
                )
                self.alert_manager.alert_sent_flags["fire"] = True
                # Set a timer to automatically reset the flag after a short period (cooldown)
                threading.Timer(self.config.ALERT_CONFIRMATION_SEC, self.alert_manager.alert_sent_flags.pop,
                                args=("fire", None)).start()

            # Bypass the timed alert status update below if instant alert is active
            return

        # Update generic alert status for persistent display
        self._update_generic_alert_status(
            frame, "fire", fire_detected_this_frame, self.config.FIRE_CONFIRMATION_SEC,
            "FIRE/SMOKE DETECTED!", "fire_alert", fire_images if fire_images else [frame], frame.shape[0] - 250
        )

        # Original alert trigger (y_pos ignored)
        self.alert_manager.check_and_trigger_timed_alert(
            frame, "fire", fire_detected_this_frame, self.config.FIRE_CONFIRMATION_SEC,
            "CRITICAL: Fire/Smoke Detected! Immediate Action Required.", "fire_alert",
            fire_images if fire_images else [frame], 0
        )

    def process_drowsiness_detection(self, frame: np.ndarray, rgb_frame: np.ndarray,
                                     patient_bbox: Optional[Tuple[int, int, int, int]]):
        """
        HEAVY DETECTION: Manages the 3-stage drowsiness detection process.
        """
        is_drowsy_this_frame = False

        # --- Stage 0: Check for Off State ---
        if not self.app_state.get("drowsiness_detection_active", False):
            self._reset_drowsiness_state()
            return

        if patient_bbox is None:
            self._reset_drowsiness_state()
            return

        x1, y1, x2, y2 = patient_bbox
        if x2 <= x1 and y2 <= y1:  # Corrected from x2 <= x1 or y2 <= y1 to handle non-detection better
            self._reset_drowsiness_state()
            return

        patient_roi_rgb = rgb_frame[y1:y2, x1:x2]
        if patient_roi_rgb.shape[0] < 1 or patient_roi_rgb.shape[1] < 1:
            self._reset_drowsiness_state()
            return

        # --- Stage 1: Initial Drowsiness Detection (EAR Check) ---
        face_mesh_results = self.face_mesh.process(patient_roi_rgb)

        if face_mesh_results.multi_face_landmarks:
            face_landmarks = face_mesh_results.multi_face_landmarks[0]
            ear = self._calculate_ear(face_landmarks)
            self.last_ear_value = ear

            if ear < self.config.EYE_ASPECT_RATIO_THRESHOLD:
                is_drowsy_this_frame = True
        else:
            self.last_ear_value = 0.0  # Clear EAR if face is lost

        # If in safe sleep window, we calculate EAR but suppress the alert process
        if self.app_state.get("is_patient_in_safe_sleep_window", False):
            self.drowsiness_initial_timer = None
            self.drowsiness_confirmation_start_time = None
            self.drowsiness_prompt_spoken = False
            return

        current_time = time.time()

        # NEW COOLDOWN CHECK: If recently cancelled, suppress re-triggering for a short period
        cooldown_duration = getattr(self.config, 'DROWSINESS_COOLDOWN_SEC', 8)
        if current_time - self.last_cancellation_time < cooldown_duration:
            # Display a status message instead of restarting the timer
            self.drowsiness_initial_timer = None
            return  # Skip the rest of the detection logic

        if is_drowsy_this_frame:
            # Stage 1: Initial Detection (4 seconds countdown)
            if not self.drowsiness_prompt_spoken:
                # If confirmation phase hasn't started yet, we are still in Stage 1 or just starting it.
                if self.drowsiness_initial_timer is None:
                    self.drowsiness_initial_timer = current_time
                    print(f"[Drowsiness Debug] Stage 1 Started. Timer: {self.drowsiness_initial_timer:.2f}")

                elapsed_initial = current_time - self.drowsiness_initial_timer

                # --- Transition to Stage 2: Confirmation Prompt ---
                if elapsed_initial >= self.config.INITIAL_DROWSINESS_SEC:
                    print(f"[Drowsiness Debug] Stage 1 Expired ({elapsed_initial:.2f}s). Triggering Stage 2 Prompt.")

                    # Speak the prompt
                    prompt_message = f"Hello {self.alert_manager.patient_name}, are you sleeping? If you are okay, please show one finger to cancel the alarm."
                    # We need to use the alert manager's speech function here
                    threading.Thread(target=self.alert_manager.speak_reminder, args=(prompt_message,),
                                     daemon=True).start()

                    # Set flags for Stage 2
                    self.drowsiness_prompt_spoken = True
                    # Reset initial timer to None to stop its countdown display
                    self.drowsiness_initial_timer = None
                    # ONLY SET START TIME HERE TO PREVENT RE-TRIGGERS
                    self.drowsiness_confirmation_start_time = current_time
                    print(
                        f"[Drowsiness Debug] Stage 2 Confirmation Started. Prompt Spoken: {self.drowsiness_confirmation_start_time:.2f}")

            # Stage 2: Confirmation Phase (Check Gesture Timeout)
            if self.drowsiness_prompt_spoken:
                elapsed_confirmation = current_time - self.drowsiness_confirmation_start_time

                # --- Transition to Stage 3: Trigger Final Alert ---
                if elapsed_confirmation >= self.config.GESTURE_CONFIRMATION_TIMEOUT:
                    print(f"[Drowsiness Debug] Stage 2 Timeout. Triggering FINAL ALERT!")
                    alert_message = f"Drowsiness Confirmed: Patient did not respond to prompt within {self.config.GESTURE_CONFIRMATION_TIMEOUT}s."
                    self.alert_manager.trigger_alarm(alert_message, "drowsiness_alert", [frame])
                    self._reset_drowsiness_state()

        else:
            # Drowsiness condition is broken, reset all state
            if self.drowsiness_initial_timer is not None or self.drowsiness_prompt_spoken:
                print("[Drowsiness Debug] Condition broken. Resetting all states.")
            self._reset_drowsiness_state()
            return

    def check_drowsiness_gesture(self, frame: np.ndarray, rgb_frame: np.ndarray) -> bool:
        """
        LIGHT DETECTION: Checks for the 'one finger' gesture only during the drowsiness confirmation phase.
        Returns True if confirmation gesture is made.
        """
        # Only run gesture check if we are in the confirmation stage
        if not self.drowsiness_prompt_spoken:
            return False

        results_hands = self.hands.process(rgb_frame)
        if results_hands.multi_hand_landmarks:
            for hand_landmarks, handedness_info in zip(results_hands.multi_hand_landmarks,
                                                       results_hands.multi_handedness):
                fingers_up = self._count_fingers(hand_landmarks, handedness_info)

                # Check for one finger (Value 1)
                if fingers_up == 1:
                    # Draw a confirmation box
                    cv2.putText(frame, "CONFIRMATION GESTURE DETECTED!", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 255, 0), 2)
                    print("[Drowsiness Debug] Confirmation Gesture (1 finger) detected! Resetting state.")
                    # Set the cancellation time here
                    self.last_cancellation_time = time.time()
                    return True
        return False

    def _reset_drowsiness_state(self):
        """Resets all internal state variables for the drowsiness detection process."""
        self.drowsiness_initial_timer = None
        self.drowsiness_confirmation_start_time = None
        self.drowsiness_prompt_spoken = False
        # IMPORTANT: Do NOT reset self.last_ear_value here, so it persists for the overlay.

    def draw_drowsiness_overlay(self, frame: np.ndarray):
        """
        LIGHT DRAWING: Draws the persistent UI elements for drowsiness detection on every frame.
        NOTE: EAR value is displayed even in the safe sleep window.
        """
        if not self.app_state.get("drowsiness_detection_active", False):
            return

        ear = self.last_ear_value
        ear_color = (0, 255, 0)  # Green (OK)
        # Start Y position for Drowsiness display (just below People/Night Mode)
        y_pos = 60

        # Get cooldown duration (using default 8s if setting is missing)
        cooldown_duration = getattr(self.config, 'DROWSINESS_COOLDOWN_SEC', 8)
        current_time = time.time()

        # --- Display EAR Value ---
        if ear > 0.0:
            if ear < self.config.EYE_ASPECT_RATIO_THRESHOLD:
                ear_color = (0, 0, 255)  # Red (Closed/Drowsy)

            cv2.putText(frame, f"EAR: {ear:.3f}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ear_color, 2)
            y_pos += 30

        # --- Check for Sleep Window Suppression ---
        if self.app_state.get("is_patient_in_safe_sleep_window", False):
            cv2.putText(frame, "Drowsiness OFF (Scheduled Sleep)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 165, 0), 2)
            y_pos += 30

        # --- Display Cooldown Status ---
        if current_time - self.last_cancellation_time < cooldown_duration:
            cooldown_remaining = int(cooldown_duration - (current_time - self.last_cancellation_time)) + 1
            cv2.putText(frame, f"COOLDOWN: {cooldown_remaining}s", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 0), 2)
            y_pos += 30

        # --- Display Initial Detection Countdown (Stage 1) ---
        # Draw this ONLY if the timer is active AND the prompt has NOT been spoken yet.
        if self.drowsiness_initial_timer is not None and not self.drowsiness_prompt_spoken:
            elapsed_initial = time.time() - self.drowsiness_initial_timer
            countdown = self.config.INITIAL_DROWSINESS_SEC - elapsed_initial

            if countdown > 0:
                # Use the next available y_pos after EAR display
                cv2.putText(frame, f"Drowsiness Check: {int(countdown) + 1}s",
                            (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                y_pos += 30

        # --- Display Confirmation Phase Status (Stage 2) ---
        # Draw this ONLY if the prompt HAS been spoken.
        if self.drowsiness_prompt_spoken and self.drowsiness_confirmation_start_time is not None:
            elapsed_confirmation = time.time() - self.drowsiness_confirmation_start_time
            countdown = self.config.GESTURE_CONFIRMATION_TIMEOUT - elapsed_confirmation
            prompt_color = (0, 165, 255)  # Orange for Confirmation state

            if countdown > 0:  # Ensure we stop drawing after timeout/trigger
                cv2.putText(frame, "AWAITING GESTURE RESPONSE", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            prompt_color, 2)
                y_pos += 30
                cv2.putText(frame, f"Alert in: {int(countdown) + 1}s",
                            (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, prompt_color, 2)

    def process_pain_detection(self, frame: np.ndarray, rgb_frame: np.ndarray,
                               patient_bbox: Optional[Tuple[int, int, int, int]]):
        """
        HEAVY DETECTION: Detects facial signs of pain (grimace, eye squeeze) using MediaPipe.
        """
        if not self.app_state["pain_detection_active"] or patient_bbox is None:
            self.pain_alert_started_time = None
            self.last_pain_score = 0.0
            self.generic_alert_status["pain"] = None
            self.alert_manager.alert_timers.pop("pain", None)
            self.alert_manager.alert_sent_flags.pop("pain", None)
            self.pain_landmarks = None
            self.pain_roi_dims = None
            return

        x1, y1, x2, y2 = patient_bbox
        # Save the current ROI dimensions for persistent drawing in draw_pain_detection_overlay
        self.pain_roi_dims = (x1, y1, x2, y2)

        patient_roi_rgb = rgb_frame[y1:y2, x1:x2]

        if patient_roi_rgb.shape[0] < 1 or patient_roi_rgb.shape[1] < 1:
            self.pain_landmarks = None
            return

        face_mesh_results = self.face_mesh.process(patient_roi_rgb)
        pain_detected_this_frame = False
        current_time = time.time()

        self.pain_landmarks = None  # Assume no landmarks unless detected below

        if face_mesh_results.multi_face_landmarks:
            face_landmarks = face_mesh_results.multi_face_landmarks[0]

            # STORE LANDMARKS FOR PERSISTENT DRAWING
            self.pain_landmarks = face_landmarks

            # Calculate metrics
            eye_squeeze_norm, mouth_asymmetry_norm = self._calculate_pain_metrics(face_landmarks)

            # --- START UPDATED PAIN SCORE LOGIC ---

            PAIN_TRIGGER_THRESHOLD = 0.25  # Slightly higher for clearer detection

            # Convert normalized values to standard Python floats for numpy usage
            # We assume numpy is imported (as np) since it's used elsewhere in this class.

            # ----------------------------
            # 1. EYE SQUEEZE CONTRIBUTION (Normalized: 0.010 - 0.060)
            # ----------------------------
            # Smaller distance = tighter eyes = pain
            SQUEEZE_MIN = 0.010  # extreme squeeze (eyes shut)
            SQUEEZE_MAX = 0.060  # relaxed eyes

            # Ensure the difference is not zero before dividing
            squeeze_range_diff = SQUEEZE_MAX - SQUEEZE_MIN
            if squeeze_range_diff > 0:
                # Invert the relationship: High input (relaxed) means low output score.
                eye_s_norm = np.clip((SQUEEZE_MAX - eye_squeeze_norm) / squeeze_range_diff, 0, 1)
            else:
                eye_s_norm = 0.0

            eye_s_score = eye_s_norm * 0.5  # max 0.5 contribution

            # ----------------------------
            # 2. MOUTH ASYMMETRY CONTRIBUTION (Normalized: 0.015 - 0.080)
            # ----------------------------
            # Larger asymmetry = more pain (grimace)
            ASYM_MIN = 0.015  # neutral face
            ASYM_MAX = 0.080  # high asymmetry (grimace)

            # Ensure the difference is not zero before dividing
            asym_range_diff = ASYM_MAX - ASYM_MIN
            if asym_range_diff > 0:
                # Direct relationship: High input (asymmetry) means high output score.
                mouth_a_norm = np.clip((mouth_asymmetry_norm - ASYM_MIN) / asym_range_diff, 0, 1)
            else:
                mouth_a_norm = 0.0

            mouth_a_score = mouth_a_norm * 0.5  # max 0.5 contribution

            # ----------------------------
            # 3. COMBINED PAIN SCORE
            # ----------------------------
            combined_pain_score = min(1.0, eye_s_score + mouth_a_score)
            self.last_pain_score = combined_pain_score

            # Trigger condition
            is_pain_alert_condition = combined_pain_score > PAIN_TRIGGER_THRESHOLD

            # --- END UPDATED PAIN SCORE LOGIC ---

            if is_pain_alert_condition:
                # Draw red box around person if pain is detected
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        else:
            self.last_pain_score = 0.0
            is_pain_alert_condition = False
            self.pain_landmarks = None  # Clear landmarks if face is lost

        # --- INSTANT ALERT LOGIC (NOW USES THE NEW SCORE AND THRESHOLD) ---
        if self.app_state["pain_detection_active"] and is_pain_alert_condition and getattr(self.config,
                                                                                           'PAIN_INSTANT_ALERT', False):
            # Check if alert hasn't been sent recently
            if not self.alert_manager.alert_sent_flags.get("pain", False):
                self.alert_manager.trigger_alarm(
                    f"CRITICAL: High Pain Detected (INSTANT SCORE: {self.last_pain_score:.2f})",
                    "pain_alert", [frame] if patient_bbox else [],
                    is_dashboard_alert=True, alert_type="pain_alert"
                )
                self.alert_manager.alert_sent_flags["pain"] = True
                # Use the configured PAIN_CONFIRMATION_SEC (now 1s in config) as a simple cooldown duration
                threading.Timer(self.config.PAIN_CONFIRMATION_SEC, self.alert_manager.alert_sent_flags.pop,
                                args=("pain", None)).start()
            return  # Skip timed alert process if instant alert is active
        # --- END INSTANT ALERT LOGIC ---

        # --- Trigger Logic (Timed - used if PAIN_INSTANT_ALERT is False) ---
        self._update_generic_alert_status(
            frame, "pain", is_pain_alert_condition, self.config.PAIN_CONFIRMATION_SEC,
            f"Discomfort ({self.last_pain_score:.2f})", "pain_alert", [frame[y1:y2, x1:x2]] if patient_bbox else [],
            frame.shape[0] - 250  # Display position
        )

        # Original alert trigger (y_pos ignored)
        self.alert_manager.check_and_trigger_timed_alert(
            frame, "pain", is_pain_alert_condition, self.config.PAIN_CONFIRMATION_SEC,
            f"High Discomfort/Pain Detected (Score: {self.last_pain_score:.2f})", "pain_alert",
            [frame] if patient_bbox else [], 0
        )

    def draw_pain_detection_overlay(self, frame: np.ndarray):
        """
        LIGHT DRAWING: Draws the persistent UI elements for Pain Detection on every frame,
        including the facial landmarks if available.
        """
        if not self.app_state.get("pain_detection_active", False):
            return

        # Y position for Pain Score (Top of the frame, below FPS)
        y_pos = 180  # This position aligns with the previous placement below drowsiness/cough.

        score = self.last_pain_score
        score_color = (0, 255, 0)  # Green (OK)

        # Color coding the score based on intensity
        PAIN_TRIGGER_THRESHOLD = 0.25  # Define locally for coloring clarity

        if score > PAIN_TRIGGER_THRESHOLD:
            score_color = (0, 255, 255)  # Yellow/Orange (Low Discomfort)
        if score > 0.5:
            score_color = (0, 165, 255)  # Orange (High Discomfort)
        if score > 0.7:
            score_color = (0, 0, 255)  # Red (Severe Pain)

        # --- Display Pain Score at the TOP LEFT ---
        cv2.putText(frame, f"Pain Score: {score:.2f}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2)
        y_pos += 30

        # --- Draw Face Mesh Persistence for Pain Detection ---
        if self.pain_landmarks and self.pain_roi_dims:
            x1, y1, x2, y2 = self.pain_roi_dims
            # Ensure ROI is valid and draw the mesh
            if x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0 and x2 <= frame.shape[1] and y2 <= frame.shape[0]:
                patient_roi_bgr = frame[y1:y2, x1:x2]
                self.drawing_utils.draw_landmarks(image=patient_roi_bgr, landmark_list=self.pain_landmarks,
                                                  connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                                                  landmark_drawing_spec=self.drawing_spec,
                                                  connection_drawing_spec=self.drawing_spec)
        # --- End Face Mesh Drawing ---

    def trigger_companion_chat_if_needed(self, frame: np.ndarray, sadness_detected: bool, sadness_score: float):
        """
        Custom logic to trigger the Ollama emotional companion if sadness is confirmed
        and the system is not in a cooldown period or already chatting.
        """
        current_time = time.time()

        # 1. Check if the companion feature is globally enabled (assuming it's tied to emotion_detection for now)
        if not self.app_state.get("emotion_detection_active", False):
            self.last_sadness_trigger_time = 0.0  # Reset on disable
            return

        # *** FIX: If companion chat is already active, stop checking for sadness immediately ***
        if self.is_companion_active:
            # Only update local flag if the global state says chat is over
            if not self.app_state.get("is_companion_chat_active", False):
                self.is_companion_active = False
            return

        # 2. Check Cooldown
        if current_time - self.last_sadness_trigger_time < self.config.COMPANION_COOLDOWN_SEC:
            # Still in cooldown, do nothing
            return

        # 3. Check if sadness is confirmed (we use the alert manager's internal timer for confirmation)
        is_sadness_confirmed = self.alert_manager.check_confirmation_status("sadness", sadness_detected,
                                                                            self.config.EMOTION_DETECTION_CONFIRMATION_SEC)

        if is_sadness_confirmed:
            # We don't want to re-trigger the chat if one is currently active via voice commands or a previous sadness trigger
            if not self.app_state.get("is_companion_chat_active", False):
                # Update the core's companion flag and the app state flag (used by MainApplication)
                self.is_companion_active = True
                self.app_state["is_companion_chat_active"] = True
                self.last_sadness_trigger_time = current_time

                print(f"[COMPANION] Sadness confirmed (Score: {sadness_score:.3f}). Initiating companion chat.")

                # The actual chat initialization and voice output must happen non-blockingly
                # We assume AlertManager has a method for this.
                prompt = f"The SAFETY, {self.alert_manager.patient_name}, has been detected showing signs of sadness with an emotional intensity score of {sadness_score:.3f}. Initiate a proactive, supportive, and gentle conversation to check in on their emotional state. Start with a non-alarming, empathetic greeting."

                # Use threading to call the AlertManager's chat function to avoid blocking the video loop
                threading.Thread(target=self.alert_manager.start_companion_chat,
                                 args=(prompt, self.config.OLLAMA_MODEL_NAME, self.config.COMPANION_TIMEOUT_SEC),
                                 daemon=True).start()

                # Suppress the normal alarm sound for sadness, as the companion chat takes over
                self.alert_manager.alert_sent_flags["sadness"] = True

        elif not sadness_detected:
            # Condition is broken, reset the alert manager's internal timer
            self.alert_manager.alert_timers.pop("sadness", None)

    def process_emotion_detection(self, frame: np.ndarray, rgb_frame: np.ndarray,
                                  patient_bbox: Optional[Tuple[int, int, int, int]]):
        """
        HEAVY DETECTION: Detects happiness and sadness based on facial landmarks (simple heuristic).
        This function calculates scores and stores state; drawing is done in draw_emotion_detection_overlay.
        """
        # *** FIX: Stop detection if companion chat is active ***
        if self.is_companion_active:
            # Only update local flag if the global state says chat is over
            if not self.app_state.get("is_companion_chat_active", False):
                self.is_companion_active = False  # Reset flag and proceed to normal flow for drawing/updates

            if self.is_companion_active:
                # If chat is still active, bypass all detection and debugging prints
                # Retain the last calculated scores and landmarks for persistent drawing
                return
        # *******************************************************

        if not self.app_state.get("emotion_detection_active", False) or patient_bbox is None:
            self.last_happiness_score = 0.0
            self.last_sadness_score = 0.0
            self.emotion_landmarks = None  # CLEAR state when feature is off or SAFETY is gone
            self.emotion_roi_dims = None
            self.generic_alert_status["happiness"] = None
            self.generic_alert_status["sadness"] = None
            self.alert_manager.alert_timers.pop("happiness", None)
            self.alert_manager.alert_sent_flags.pop("happiness", None)
            self.alert_manager.alert_timers.pop("sadness", None)
            self.alert_manager.alert_sent_flags.pop("sadness", None)
            self.is_companion_active = False  # Clear active chat state
            return

        x1, y1, x2, y2 = patient_bbox
        patient_roi_rgb = rgb_frame[y1:y2, x1:x2]

        # Save the current ROI dimensions for persistent drawing in draw_emotion_detection_overlay
        self.emotion_roi_dims = (x1, y1, x2, y2)

        if patient_roi_rgb.shape[0] < 1 or patient_roi_rgb.shape[1] < 1:
            self.emotion_landmarks = None
            return

        face_mesh_results = self.face_mesh.process(patient_roi_rgb)

        happiness_detected_this_frame = False
        sadness_detected_this_frame = False

        if face_mesh_results.multi_face_landmarks:
            face_landmarks = face_mesh_results.multi_face_landmarks[0]

            # STORE LANDMARKS FOR PERSISTENT DRAWING (LIGHTWEIGHT FUNCTION)
            self.emotion_landmarks = face_landmarks

            # Calculate metrics
            happiness_score, sadness_score = self._calculate_emotion_metrics(face_landmarks)

            self.last_happiness_score = happiness_score
            self.last_sadness_score = sadness_score

            # --- DEBUGGING OUTPUT (Keep this for threshold tuning) ---
            print(
                f"DEBUG EMOTION: Happy Score={happiness_score:.5f}, Sad Score={sadness_score:.5f}, Threshold={self.config.HAPPINESS_THRESHOLD}")
            # --------------------------

            # Determine detection based on configured thresholds
            if happiness_score > self.config.HAPPINESS_THRESHOLD:
                happiness_detected_this_frame = True

            if sadness_score > self.config.SADNESS_THRESHOLD:
                sadness_detected_this_frame = True

            # Reset companion activity if happiness is detected while sad chat is active (SAFETY intervention)
            if happiness_detected_this_frame and self.app_state.get("is_companion_chat_active", False):
                self.is_companion_active = False
                self.app_state["is_companion_chat_active"] = False
                print("[COMPANION] Happiness detected. Companion chat cancelled.")
                self.alert_manager.cancel_companion_chat(
                    "happiness_alert")  # Assuming AlertManager handles cancellation


        else:
            self.last_happiness_score = 0.0
            self.last_sadness_score = 0.0
            self.emotion_landmarks = None
            self.emotion_roi_dims = None  # Clear ROI if face is lost

        # --- Sadness Companion Logic ---
        # NOTE: This overrides the default check_and_trigger_timed_alert for sadness
        self.trigger_companion_chat_if_needed(frame, sadness_detected_this_frame, self.last_sadness_score)

        # --- Trigger Logic for Happiness (Standard Alarm) ---
        self._update_generic_alert_status(
            frame, "happiness", happiness_detected_this_frame, self.config.EMOTION_DETECTION_CONFIRMATION_SEC,
            f"Happy ({self.last_happiness_score:.2f})", "happiness_alert", [frame] if patient_bbox else [],
            frame.shape[0] - 280  # Display position
        )
        self.alert_manager.check_and_trigger_timed_alert(
            frame, "happiness", happiness_detected_this_frame, self.config.EMOTION_DETECTION_CONFIRMATION_SEC,
            f"Patient is Happy ({self.last_happiness_score:.2f})", "happiness_alert", [frame] if patient_bbox else [], 0
        )

        # --- Trigger Logic for Sadness (Standard Alarm - Only if companion is NOT used/active) ---
        # The sadness alarm only serves as a visual indicator now. The actual trigger is handled by trigger_companion_chat_if_needed.
        # We ensure the alarm is only tracked here for visual countdown, but the sent flag is managed by the companion logic.
        self._update_generic_alert_status(
            frame, "sadness", sadness_detected_this_frame, self.config.EMOTION_DETECTION_CONFIRMATION_SEC,
            f"Sad ({self.last_sadness_score:.2f})", "sadness_alert", [frame] if patient_bbox else [],
            frame.shape[0] - 310  # Display position
        )

    def draw_emotion_detection_overlay(self, frame: np.ndarray):
        """
        LIGHT DRAWING: Draws the persistent UI elements for Emotion Detection on every frame.
        This function runs constantly to prevent flickering.
        """
        if not self.app_state.get("emotion_detection_active", False):
            return

        # Y position for Emotion Score (below Pain Score - adjust to clear line 210)
        y_pos = 210

        h_score = self.last_happiness_score
        s_score = self.last_sadness_score

        h_color = (0, 255, 0) if h_score > self.config.HAPPINESS_THRESHOLD else (255, 255, 255)
        s_color = (0, 0, 255) if s_score > self.config.SADNESS_THRESHOLD else (255, 255, 255)

        # 1. Draw Score Persistence
        cv2.putText(frame, f"Happy: {h_score:.3f}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, h_color, 2)
        y_pos += 30
        cv2.putText(frame, f"Sad: {s_score:.3f}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, s_color, 2)

        # Companion Status Overlay
        if self.app_state.get("is_companion_chat_active", False):
            cv2.putText(frame, "COMPANION ACTIVE", (10, y_pos + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

        # 2. Draw Face Mesh Persistence (NEW: Uses stored landmarks/ROI)
        if self.emotion_landmarks and self.emotion_roi_dims:
            x1, y1, x2, y2 = self.emotion_roi_dims
            # Ensure ROI is valid and draw the mesh
            if x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0 and x2 <= frame.shape[1] and y2 <= frame.shape[0]:
                patient_roi_bgr = frame[y1:y2, x1:x2]
                self.drawing_utils.draw_landmarks(image=patient_roi_bgr, landmark_list=self.emotion_landmarks,
                                                  connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                                                  landmark_drawing_spec=self.drawing_spec,
                                                  connection_drawing_spec=self.drawing_spec)

        # 3. Draw Countdown Persistence (already existed)

        # Happiness Countdown (Redraw if active)
        h_status = self.generic_alert_status.get("happiness")
        if h_status is not None and not self.alert_manager.alert_sent_flags.get("happiness", True):
            elapsed = time.time() - h_status["timestamp"]
            countdown = h_status["conf_sec"] - elapsed
            if countdown > 0:
                cv2.putText(frame, f"Happy Alert in: {int(countdown) + 1}s", (10, h_status['y_pos']),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Sadness Countdown (Redraw if active)
        s_status = self.generic_alert_status.get("sadness")
        if s_status is not None and not self.alert_manager.alert_sent_flags.get("sadness", True):
            elapsed = time.time() - s_status["timestamp"]
            countdown = s_status["conf_sec"] - elapsed
            if countdown > 0:
                # NOTE: Using orange/yellow here to indicate this is leading to the COMPANION chat, not a standard alarm
                cv2.putText(frame, f"Companion Check in: {int(countdown) + 1}s", (10, s_status['y_pos']),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    def process_cough_detection(self, frame: np.ndarray, rgb_frame: np.ndarray,
                                patient_bbox: Optional[Tuple[int, int, int, int]]):
        """
        HEAVY DETECTION: Detects mouth opening and body motion for cough, and updates internal state.
        The drawing is handled by draw_cough_detection_overlay().
        """
        if not self.app_state["cough_detection_active"]:
            self.cough_count = 0
            self.last_cough_time = 0
            self.cough_alert_sent = False
            self.last_face_landmarks = None
            self.last_mar_value = 0.0
            self.last_patient_roi_dims = None
            return

        # Reset cough count if SAFETY leaves frame or timeout occurs
        if patient_bbox is None or (
                self.last_cough_time > 0 and time.time() - self.last_cough_time > self.config.COUGH_RESET_SEC):
            self.cough_count = 0
            self.cough_alert_sent = False

        if patient_bbox is None: return

        x1, y1, x2, y2 = patient_bbox
        if x2 <= x1 or y2 <= y1: return
        patient_roi_rgb = rgb_frame[y1:y2, x1:x2]

        if patient_roi_rgb.shape[0] < 1 or patient_roi_rgb.shape[1] < 1: return

        # Update ROI dimensions for the persistent drawing function
        self.last_patient_roi_dims = (x1, y1, x2, y2)

        pose_results = self.pose.process(patient_roi_rgb)
        face_mesh_results = self.face_mesh.process(patient_roi_rgb)

        mouth_open = False
        head_forward = False
        body_forward = False
        mar = 0.0

        # --- Face Mesh Detection and MAR Calculation ---
        if face_mesh_results.multi_face_landmarks:
            face_landmarks = face_mesh_results.multi_face_landmarks[0]
            self.last_face_landmarks = face_landmarks

            p_upper_lip = face_landmarks.landmark[13]
            p_lower_lip = face_landmarks.landmark[14]
            p_left_corner = face_landmarks.landmark[61]
            p_right_corner = face_landmarks.landmark[291]

            ver_dist = math.hypot(p_upper_lip.x - p_lower_lip.x, p_upper_lip.y - p_lower_lip.y)
            hor_dist = math.hypot(p_left_corner.x - p_right_corner.x, p_left_corner.y - p_right_corner.y)

            if hor_dist > 0:
                mar = ver_dist / hor_dist
                self.last_mar_value = mar
                if mar > self.config.MOUTH_ASPECT_RATIO_THRESHOLD:
                    mouth_open = True
        else:
            self.last_face_landmarks = None
            self.last_mar_value = 0.0

        # --- Pose Detection (Body and Head Motion) ---
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            try:
                # Head movement (Nose Z-axis)
                current_nose_z = landmarks[self.mp_pose.PoseLandmark.NOSE.value].z
                if self.previous_nose_z is not None and self.previous_nose_z - current_nose_z > self.config.HEAD_FORWARD_THRESHOLD:
                    head_forward = True
                self.previous_nose_z = current_nose_z

                # Body movement (Shoulder Z-axis)
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

        # --- Cough Logic ---
        # Note: mouth_open relies on fresh MAR calculation, which only happens on this frame.
        if mouth_open and head_forward and body_forward:
            current_time = time.time()
            if current_time - self.last_cough_time > self.config.COUGH_CONFIRMATION_SEC:
                self.cough_count += 1
                self.last_cough_time = current_time
                print(f"Cough detected. Count: {self.cough_count}")

        if self.cough_count >= self.config.COUGH_COUNT_THRESHOLD and not self.cough_alert_sent:
            alert_message = f"Frequent Coughing Detected ({self.cough_count} coughs)"
            self.alert_manager.trigger_alarm(alert_message, "cough_alert", [frame])
            self.cough_alert_sent = True
            self.cough_count = 0

    def draw_cough_detection_overlay(self, frame: np.ndarray):
        """LIGHT DRAWING: Draws the persistent UI elements for cough detection on every frame."""
        if not self.app_state["cough_detection_active"]:
            return

        # Y=90 is the expected position for Cough Count (shifted down to clear Drowsiness)
        y_pos = 90
        cv2.putText(frame, f"Cough Count: {self.cough_count}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255),
                    2)

        if self.last_face_landmarks and self.last_patient_roi_dims:
            x1, y1, x2, y2 = self.last_patient_roi_dims
            patient_roi_bgr = frame[y1:y2, x1:x2]

            # Draw the last detected face mesh on the current frame's ROI
            self.drawing_utils.draw_landmarks(image=patient_roi_bgr, landmark_list=self.last_face_landmarks,
                                              connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                                              landmark_drawing_spec=self.drawing_spec,
                                              connection_drawing_spec=self.drawing_spec)

            mar = self.last_mar_value
            mouth_open = mar > self.config.MOUTH_ASPECT_RATIO_THRESHOLD

            # Y=120 is the expected position for MAR
            cv2.putText(frame, f"MAR: {mar:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255) if mouth_open else (255, 255, 255), 2)

    def process_gestures(self, frame: np.ndarray, rgb_frame: np.ndarray):
        """Detects hand gestures for calling for help."""
        if not self.app_state["gestures_active"]: return
        results_hands = self.hands.process(rgb_frame)
        if results_hands.multi_hand_landmarks:
            for hand_landmarks, handedness_info in zip(results_hands.multi_hand_landmarks,
                                                       results_hands.multi_handedness):
                fingers_up = self._count_fingers(hand_landmarks, handedness_info)
                cv2.putText(frame, f"Fingers: {fingers_up}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                            2)  # Y=210 (Shifted lower to clear Drowsiness/Cough)
            if fingers_up in self.gesture_actions:
                label, sound_key = self.gesture_actions[fingers_up]
                cv2.putText(frame, f"Gesture: {label}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255),
                            2)  # Y=240
                current_time = time.time()
                if fingers_up not in self.gesture_detected_time: self.gesture_detected_time[fingers_up] = current_time
                elapsed = current_time - self.gesture_detected_time[fingers_up]
                if elapsed > self.config.GESTURE_CONFIRMATION_SEC:

                    # --- CORRECTION APPLIED HERE to trigger dashboard alert ---
                    is_dash_alert = True

                    self.alert_manager.trigger_alarm(
                        f"Gesture: {label}",
                        sound_key,
                        images=[frame],
                        # Pass dashboard flag and alert type (sound_key acts as alert_type)
                        is_dashboard_alert=is_dash_alert,
                        alert_type=sound_key
                    )
                    # --- END CORRECTION ---

                    self.gesture_detected_time.pop(fingers_up, None)
                else:
                    countdown = self.config.GESTURE_CONFIRMATION_SEC - elapsed
                    cv2.putText(frame, f"in {int(countdown) + 1}s", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 255), 2)
            else:
                self.gesture_detected_time.clear()
        else:
            self.gesture_detected_time.clear()

    def _count_fingers(self, hand_landmarks, handedness_info) -> int:
        """Helper function to count the number of extended fingers."""
        finger_count = 0
        hand_label = handedness_info.classification[0].label
        tip_ids = [4, 8, 12, 16, 20]
        for i in range(1, 5):
            if hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[tip_ids[i] - 2].y:
                finger_count += 1
        if (hand_label == "Right" and hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[
            tip_ids[0] - 1].x) or (
                hand_label == "Left" and hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[
            tip_ids[0] - 1].x):
            finger_count += 1
        return finger_count

    def process_bed_exit(self, frame: np.ndarray, patient_bbox: Optional[Tuple[int, int, int, int]]):
        """
        Detects if the SAFETY has left a predefined bed area by tracking their
        bounding box center point.
        """
        bed_roi = self.app_state.get("bed_roi")
        if not self.app_state["bed_exit_active"] or bed_roi is None: return
        x, y, w, h = bed_roi
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        patient_in_bed = False
        if patient_bbox:
            px1, py1, px2, py2 = patient_bbox
            person_center_x = (px1 + px2) // 2
            person_center_y = (py1 + py2) // 2
            if x < person_center_x < (x + w) and y < person_center_y < (y + h): patient_in_bed = True
            cv2.circle(frame, (person_center_x, person_center_y), 5, (0, 255, 255), -1)

        if patient_in_bed: self.app_state["patient_was_in_bed"] = True
        patient_out_of_bed = self.app_state["patient_was_in_bed"] and not patient_in_bed

        self._update_generic_alert_status(
            frame, "bed_exit", patient_out_of_bed, self.config.BED_EXIT_CONFIRMATION_SEC,
            "Patient Left Bed", "bed_exit_alert", [frame] if patient_out_of_bed else [], frame.shape[0] - 130
        )
        # Original alert trigger (y_pos ignored)
        self.alert_manager.check_and_trigger_timed_alert(
            frame, "bed_exit", patient_out_of_bed, self.config.BED_EXIT_CONFIRMATION_SEC,
            "Patient Left Bed", "bed_exit_alert", [frame] if patient_out_of_bed else [], 0
        )

    def process_fall_detection(self, frame: np.ndarray, rgb_frame: np.ndarray):
        """
        Detects if a person has fallen using a custom YOLO model.
        UPDATED to use the user-provided logic for clearer structure.
        """
        if not self.app_state["fall_detection_active"] or self.yolo_fall_model is None:
            # Clear any timers if feature is disabled
            self.alert_manager.alert_timers.pop("fall_detection", None)
            self.alert_manager.alert_sent_flags.pop("fall_detection", None)
            self.generic_alert_status["fall_detection"] = None
            return

        fall_detected_this_frame = False
        fall_images = []

        # Run inference using the predict method
        results = self.yolo_fall_model.predict(
            rgb_frame,
            imgsz=640,
            conf=self.config.YOLO_CONFIDENCE_THRESHOLD,  # Use the global confidence threshold
            verbose=False
        )

        for result in results:
            if len(result.boxes) > 0:
                # We assume the model is trained to detect 'fall' as the primary class
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    # Assuming class ID 0 is 'fall' based on common YOLO dataset structure
                    FALL_CLASS_ID = 0

                    if class_id == FALL_CLASS_ID:
                        fall_detected_this_frame = True

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        label = f"Fall {conf:.2f}"

                        # Draw bounding box and label (using Red for critical alert)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                        fall_images.append(frame[y1:y2, x1:x2].copy())

        # Update generic alert status for persistent display
        self._update_generic_alert_status(
            frame, "fall_detection", fall_detected_this_frame, self.config.FALL_CONFIRMATION_SEC,
            "Fall Detected", "fall_alert", fall_images if fall_images else [frame], frame.shape[0] - 160
        )

        # Trigger the full alert chain after confirmation time
        self.alert_manager.check_and_trigger_timed_alert(
            frame, "fall_detection", fall_detected_this_frame, self.config.FALL_CONFIRMATION_SEC,
            "CRITICAL: Fall Detected! Immediate Action Required.", "fall_alert",
            fall_images if fall_images else [frame], 0
        )

    def process_stroke_detection_mediapipe(self, frame: np.ndarray, rgb_frame: np.ndarray,
                                           patient_bbox: Optional[Tuple[int, int, int, int]]):
        """Detects potential stroke symptoms based on facial asymmetry (mouth droop)."""
        if not self.app_state["stroke_detection_active"] or patient_bbox is None:
            self.alert_manager.alert_timers["stroke"] = None
            self.alert_manager.alert_sent_flags["stroke"] = False
            self.generic_alert_status["stroke"] = None
            return

        x1, y1, x2, y2 = patient_bbox
        if x2 <= x1 or y2 <= y1: return
        patient_face_roi = rgb_frame[y1:y2, x1:x2].copy()
        if patient_face_roi.shape[0] < 1 or patient_face_roi.shape[1] < 1: return
        results = self.face_mesh.process(patient_face_roi)
        stroke_detected_this_frame = False
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.drawing_utils.draw_landmarks(image=frame[y1:y2, x1:x2], landmark_list=face_landmarks,
                                                  connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                                                  landmark_drawing_spec=self.drawing_spec,
                                                  connection_drawing_spec=self.drawing_spec)
                left_mouth_corner = face_landmarks.landmark[61]
                right_mouth_corner = face_landmarks.landmark[291]
                droop_difference = abs(left_mouth_corner.y - right_mouth_corner.y)
                if droop_difference > self.config.MOUTH_DROOP_THRESHOLD:
                    stroke_detected_this_frame = True
                    label = f"Stroke: Yes ({droop_difference:.3f})"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        self._update_generic_alert_status(
            frame, "stroke", stroke_detected_this_frame, self.config.STROKE_CONFIRMATION_SEC,
            "Potential Stroke", "stroke_alert", [frame[y1:y2, x1:x2]] if stroke_detected_this_frame else [],
            frame.shape[0] - 160
        )
        # Original alert trigger (y_pos ignored)
        self.alert_manager.check_and_trigger_timed_alert(
            frame, "stroke", stroke_detected_this_frame, self.config.STROKE_CONFIRMATION_SEC,
            "Potential Stroke", "stroke_alert", [frame[y1:y2, x1:x2]] if stroke_detected_this_frame else [],
            0
        )

    def process_knife_detection(self, frame: np.ndarray, rgb_frame: np.ndarray):
        """Detects knives in the frame using a dedicated YOLO model."""
        if not self.app_state["knife_detection_active"] or self.yolo_knife_model is None:
            self.knife_detected_at_high_conf = False
            self.generic_alert_status["knife"] = None
            return
        knife_detected_this_frame = False
        knife_images = []
        results = self.yolo_knife_model(rgb_frame, verbose=False, conf=self.config.YOLO_CONFIDENCE_THRESHOLD)
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                if class_id == 0:
                    knife_detected_this_frame = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if conf > self.config.KNIFE_HIGH_CONFIDENCE_THRESHOLD and not self.knife_detected_at_high_conf:
                        self.knife_detected_at_high_conf = True
                        message = f"DANGER: High-confidence Knife Detected! ({conf:.2f})"
                        self.alert_manager.trigger_alarm(message, "knife_alert", [frame[y1:y2, x1:x2]])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = f"Knife {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    knife_images.append(frame[y1:y2, x1:x2])
        if self.knife_detected_at_high_conf:
            if not knife_detected_this_frame:
                self.knife_detected_at_high_conf = False
            else:
                cv2.putText(frame, "DANGER: KNIFE ALERT ACTIVE", (10, frame.shape[0] - 190), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)
        else:
            self._update_generic_alert_status(
                frame, "knife", knife_detected_this_frame, self.config.KNIFE_CONFIRMATION_SEC,
                "Knife Detected", "knife_alert", knife_images, frame.shape[0] - 190
            )
            # Original alert trigger (y_pos ignored)
            self.alert_manager.check_and_trigger_timed_alert(
                frame, "knife", knife_detected_this_frame, self.config.KNIFE_CONFIRMATION_SEC,
                "Knife Detected", "knife_alert", knife_images, 0
            )

    def process_gun_detection(self, frame: np.ndarray, rgb_frame: np.ndarray):
        """Detects guns in the frame using a dedicated YOLO model."""
        if not self.app_state["gun_detection_active"] or self.yolo_gun_model is None:
            self.gun_detected_at_high_conf = False
            self.generic_alert_status["gun"] = None
            return

        gun_detected_this_frame = False
        gun_images = []
        results = self.yolo_gun_model(rgb_frame, verbose=False, conf=self.config.YOLO_CONFIDENCE_THRESHOLD)
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                if class_id == 0:
                    gun_detected_this_frame = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if conf > self.config.GUN_HIGH_CONFIDENCE_THRESHOLD and not self.gun_detected_at_high_conf:
                        self.gun_detected_at_high_conf = True
                        message = f"DANGER: High-confidence Gun Detected! ({conf:.2f})"
                        self.alert_manager.trigger_alarm(message, "gun_alert", [frame[y1:y2, x1:x2]])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = f"Gun {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    gun_images.append(frame[y1:y2, x1:x2])
        if self.gun_detected_at_high_conf:
            if not gun_detected_this_frame:
                self.gun_detected_at_high_conf = False
            else:
                cv2.putText(frame, "DANGER: GUN ALERT ACTIVE", (10, frame.shape[0] - 220), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)
        else:
            self._update_generic_alert_status(
                frame, "gun", gun_detected_this_frame, self.config.GUN_CONFIRMATION_SEC,
                "Gun Detected", "gun_alert", gun_images, frame.shape[0] - 220
            )
            # Original alert trigger (y_pos ignored)
            self.alert_manager.check_and_trigger_timed_alert(
                frame, "gun", gun_detected_this_frame, self.config.GUN_CONFIRMATION_SEC,
                "Gun Detected", "gun_alert", gun_images, 0
            )

    def process_safety_gear_detection(self, frame: np.ndarray, rgb_frame: np.ndarray):
        """Detects if workers are wearing required safety gear using a YOLO model."""
        if not self.app_state.get("safety_detection_active", False) or self.yolo_safety_model is None:
            self.generic_alert_status["safety_violation"] = None
            self.alert_manager.alert_timers.pop("safety_violation", None)
            self.alert_manager.alert_sent_flags.pop("safety_violation", None)
            return

        # Run inference
        results = self.yolo_safety_model.predict(rgb_frame, conf=self.config.YOLO_CONFIDENCE_THRESHOLD, verbose=False)

        # Track what was detected in this frame
        detected_gear = {cls: False for cls in self.config.YOLO_SAFETY_CLASSES}

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                label = self.yolo_safety_model.names[class_id]
                conf = box.conf[0].item()

                if label in self.config.YOLO_SAFETY_CLASSES:
                    detected_gear[label] = True
                    # Draw a bounding box for the detected gear
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)  # Pink color
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 0, 255), 2)

        # Check if the required gear is missing
        missing_gear = [gear for gear, detected in detected_gear.items() if not detected]
        is_violation = bool(missing_gear)

        if is_violation:
            # Check for a person in the frame to ensure a person is present to be monitored
            if self.patient_bbox or self.other_people_bboxes:
                message = f"SAFETY VIOLATION: Missing {', '.join(missing_gear)}"
                # Use a specific, high-priority alert type for this
                self.alert_manager.check_and_trigger_timed_alert(
                    frame, "safety_violation", True, self.config.ALERT_CONFIRMATION_SEC,
                    message, "safety_alert", [frame], 0
                )

        self._update_generic_alert_status(
            frame, "safety_violation", is_violation, self.config.ALERT_CONFIRMATION_SEC,
            f"Safety Violation: Missing {', '.join(missing_gear)}", "safety_alert", [frame] if is_violation else [],
            frame.shape[0] - 280
        )

        if not is_violation:
            self.alert_manager.check_and_trigger_timed_alert(frame, "safety_violation", False, 0, "", "", [], 0)





