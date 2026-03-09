import cv2
import numpy as np
import mediapipe as mp
import time
import threading
from typing import List, Tuple, Dict, Any, Optional


class FacialMixin:
    """Handles Emotion, Pain, and Stroke Detection using MediaPipe FaceMesh."""

    def _calculate_emotion_metrics(self, face_landmarks) -> Tuple[float, float]:
        """Calculates normalized metrics for basic emotion detection (Happiness/Sadness)."""
        p_upper_lip = face_landmarks.landmark[13]
        p_lower_lip = face_landmarks.landmark[14]
        p_left_corner = face_landmarks.landmark[61]
        p_right_corner = face_landmarks.landmark[291]
        p_mouth_y_ref = (p_upper_lip.y + p_lower_lip.y) / 2
        p_ref = face_landmarks.landmark[2]

        y_coords = [lm.y for lm in face_landmarks.landmark]
        face_height = max(y_coords) - min(y_coords) if max(y_coords) > min(y_coords) else 1.0

        if face_height == 0 or face_height < 0.01:
            return 0.0, 0.0

        happiness_score = 0.0
        sadness_score = 0.0

        left_corner_vertical_movement = (p_ref.y - p_left_corner.y) / face_height
        right_corner_vertical_movement = (p_ref.y - p_right_corner.y) / face_height
        avg_vertical_movement = (left_corner_vertical_movement + right_corner_vertical_movement) / 2.0
        happiness_score = abs(avg_vertical_movement)

        left_corner_droop = (p_left_corner.y - p_mouth_y_ref) / face_height
        right_corner_droop = (p_right_corner.y - p_mouth_y_ref) / face_height
        sadness_score = max(left_corner_droop, right_corner_droop)

        return happiness_score, sadness_score

    def _calculate_pain_metrics(self, face_landmarks) -> Tuple[float, float]:
        """Calculates normalized metrics for pain detection (Eyebrow Squeeze and Mouth Asymmetry)."""
        left_eye_top = face_landmarks.landmark[159]
        left_eyebrow_inner = face_landmarks.landmark[70]
        right_eye_top = face_landmarks.landmark[386]
        right_eyebrow_inner = face_landmarks.landmark[300]

        left_squeeze_dist = abs(left_eyebrow_inner.y - left_eye_top.y)
        right_squeeze_dist = abs(right_eyebrow_inner.y - right_eye_top.y)
        avg_squeeze_dist = (left_squeeze_dist + right_squeeze_dist) / 2.0

        left_mouth_corner = face_landmarks.landmark[61]
        right_mouth_corner = face_landmarks.landmark[291]
        mouth_asymmetry_y = abs(left_mouth_corner.y - right_mouth_corner.y)

        y_coords = [lm.y for lm in face_landmarks.landmark]
        face_height = max(y_coords) - min(y_coords)

        if face_height == 0 or face_height < 0.01:
            return 0.0, 0.0

        normalized_squeeze = avg_squeeze_dist / face_height
        normalized_asymmetry = mouth_asymmetry_y / face_height

        return normalized_squeeze, normalized_asymmetry

    def process_pain_detection(self, frame: np.ndarray, rgb_frame: np.ndarray,
                               patient_bbox: Optional[Tuple[int, int, int, int]]):
        """HEAVY DETECTION: Detects facial signs of pain (grimace, eye squeeze) using MediaPipe."""
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
        self.pain_roi_dims = (x1, y1, x2, y2)
        patient_roi_rgb = np.ascontiguousarray(rgb_frame[y1:y2, x1:x2])

        if patient_roi_rgb.shape[0] < 1 or patient_roi_rgb.shape[1] < 1:
            self.pain_landmarks = None
            return

        face_mesh_results = self.face_mesh.process(patient_roi_rgb)
        pain_detected_this_frame = False
        self.pain_landmarks = None

        if face_mesh_results.multi_face_landmarks:
            face_landmarks = face_mesh_results.multi_face_landmarks[0]
            self.pain_landmarks = face_landmarks
            eye_squeeze_norm, mouth_asymmetry_norm = self._calculate_pain_metrics(face_landmarks)

            PAIN_TRIGGER_THRESHOLD = 0.25
            SQUEEZE_MIN = 0.010
            SQUEEZE_MAX = 0.060

            squeeze_range_diff = SQUEEZE_MAX - SQUEEZE_MIN
            if squeeze_range_diff > 0:
                eye_s_norm = np.clip((SQUEEZE_MAX - eye_squeeze_norm) / squeeze_range_diff, 0, 1)
            else:
                eye_s_norm = 0.0

            eye_s_score = eye_s_norm * 0.5

            ASYM_MIN = 0.015
            ASYM_MAX = 0.080
            asym_range_diff = ASYM_MAX - ASYM_MIN
            if asym_range_diff > 0:
                mouth_a_norm = np.clip((mouth_asymmetry_norm - ASYM_MIN) / asym_range_diff, 0, 1)
            else:
                mouth_a_norm = 0.0

            mouth_a_score = mouth_a_norm * 0.5
            combined_pain_score = min(1.0, eye_s_score + mouth_a_score)
            self.last_pain_score = combined_pain_score

            is_pain_alert_condition = combined_pain_score > PAIN_TRIGGER_THRESHOLD

            if is_pain_alert_condition:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        else:
            self.last_pain_score = 0.0
            is_pain_alert_condition = False
            self.pain_landmarks = None

        if self.app_state["pain_detection_active"] and is_pain_alert_condition and getattr(self.config,
                                                                                           'PAIN_INSTANT_ALERT', False):
            if not self.alert_manager.alert_sent_flags.get("pain", False):
                self.alert_manager.trigger_alarm(
                    f"CRITICAL: High Pain Detected (INSTANT SCORE: {self.last_pain_score:.2f})",
                    "pain_alert", [frame] if patient_bbox else [],
                    is_dashboard_alert=True, alert_type="pain_alert"
                )
                self.alert_manager.alert_sent_flags["pain"] = True
                threading.Timer(self.config.PAIN_CONFIRMATION_SEC, self.alert_manager.alert_sent_flags.pop,
                                args=("pain", None)).start()
            return

        self._update_generic_alert_status(
            frame, "pain", is_pain_alert_condition, self.config.PAIN_CONFIRMATION_SEC,
            f"Discomfort ({self.last_pain_score:.2f})", "pain_alert", [frame[y1:y2, x1:x2]] if patient_bbox else [],
            frame.shape[0] - 250
        )
        self.alert_manager.check_and_trigger_timed_alert(
            frame, "pain", is_pain_alert_condition, self.config.PAIN_CONFIRMATION_SEC,
            f"High Discomfort/Pain Detected (Score: {self.last_pain_score:.2f})", "pain_alert",
            [frame] if patient_bbox else [], 0
        )

    def draw_pain_detection_overlay(self, frame: np.ndarray):
        """LIGHT DRAWING: Draws the persistent UI elements for Pain Detection on every frame."""
        if not self.app_state.get("pain_detection_active", False):
            return

        y_pos = 180
        score = self.last_pain_score
        score_color = (0, 255, 0)
        PAIN_TRIGGER_THRESHOLD = 0.25

        if score > PAIN_TRIGGER_THRESHOLD:
            score_color = (0, 255, 255)
        if score > 0.5:
            score_color = (0, 165, 255)
        if score > 0.7:
            score_color = (0, 0, 255)

        cv2.putText(frame, f"Pain Score: {score:.2f}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2)
        y_pos += 30

        if self.pain_landmarks and self.pain_roi_dims:
            x1, y1, x2, y2 = self.pain_roi_dims
            if x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0 and x2 <= frame.shape[1] and y2 <= frame.shape[0]:
                patient_roi_bgr = frame[y1:y2, x1:x2]
                self.drawing_utils.draw_landmarks(image=patient_roi_bgr, landmark_list=self.pain_landmarks,
                                                  connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                                                  landmark_drawing_spec=self.drawing_spec,
                                                  connection_drawing_spec=self.drawing_spec)

    def trigger_companion_chat_if_needed(self, frame: np.ndarray, sadness_detected: bool, sadness_score: float):
        """Logic to trigger the Ollama emotional companion if sadness is confirmed."""
        current_time = time.time()

        if not self.app_state.get("emotion_detection_active", False):
            self.last_sadness_trigger_time = 0.0
            return

        if self.is_companion_active:
            if not self.app_state.get("is_companion_chat_active", False):
                self.is_companion_active = False
            return

        if current_time - self.last_sadness_trigger_time < self.config.COMPANION_COOLDOWN_SEC:
            return

        is_sadness_confirmed = self.alert_manager.check_confirmation_status("sadness", sadness_detected,
                                                                            self.config.EMOTION_DETECTION_CONFIRMATION_SEC)

        if is_sadness_confirmed:
            if not self.app_state.get("is_companion_chat_active", False):
                self.is_companion_active = True
                self.app_state["is_companion_chat_active"] = True
                self.last_sadness_trigger_time = current_time

                print(f"[COMPANION] Sadness confirmed (Score: {sadness_score:.3f}). Initiating companion chat.")

                prompt = f"The SAFETY, {self.alert_manager.patient_name}, has been detected showing signs of sadness with an emotional intensity score of {sadness_score:.3f}. Initiate a proactive, supportive, and gentle conversation to check in on their emotional state. Start with a non-alarming, empathetic greeting."
                threading.Thread(target=self.alert_manager.start_companion_chat,
                                 args=(prompt, self.config.OLLAMA_MODEL_NAME, self.config.COMPANION_TIMEOUT_SEC),
                                 daemon=True).start()
                self.alert_manager.alert_sent_flags["sadness"] = True

        elif not sadness_detected:
            self.alert_manager.alert_timers.pop("sadness", None)

    def process_emotion_detection(self, frame: np.ndarray, rgb_frame: np.ndarray,
                                  patient_bbox: Optional[Tuple[int, int, int, int]]):
        """HEAVY DETECTION: Detects happiness and sadness based on facial landmarks."""
        if self.is_companion_active:
            if not self.app_state.get("is_companion_chat_active", False):
                self.is_companion_active = False

            if self.is_companion_active:
                return

        if not self.app_state.get("emotion_detection_active", False) or patient_bbox is None:
            self.last_happiness_score = 0.0
            self.last_sadness_score = 0.0
            self.emotion_landmarks = None
            self.emotion_roi_dims = None
            self.generic_alert_status["happiness"] = None
            self.generic_alert_status["sadness"] = None
            self.alert_manager.alert_timers.pop("happiness", None)
            self.alert_manager.alert_sent_flags.pop("happiness", None)
            self.alert_manager.alert_timers.pop("sadness", None)
            self.alert_manager.alert_sent_flags.pop("sadness", None)
            self.is_companion_active = False
            return

        x1, y1, x2, y2 = patient_bbox
        patient_roi_rgb = np.ascontiguousarray(rgb_frame[y1:y2, x1:x2])
        self.emotion_roi_dims = (x1, y1, x2, y2)

        if patient_roi_rgb.shape[0] < 1 or patient_roi_rgb.shape[1] < 1:
            self.emotion_landmarks = None
            return

        face_mesh_results = self.face_mesh.process(patient_roi_rgb)
        happiness_detected_this_frame = False
        sadness_detected_this_frame = False

        if face_mesh_results.multi_face_landmarks:
            face_landmarks = face_mesh_results.multi_face_landmarks[0]
            self.emotion_landmarks = face_landmarks
            happiness_score, sadness_score = self._calculate_emotion_metrics(face_landmarks)

            self.last_happiness_score = happiness_score
            self.last_sadness_score = sadness_score

            print(
                f"DEBUG EMOTION: Happy Score={happiness_score:.5f}, Sad Score={sadness_score:.5f}, Threshold={self.config.HAPPINESS_THRESHOLD}")

            if happiness_score > self.config.HAPPINESS_THRESHOLD:
                happiness_detected_this_frame = True

            if sadness_score > self.config.SADNESS_THRESHOLD:
                sadness_detected_this_frame = True

            if happiness_detected_this_frame and self.app_state.get("is_companion_chat_active", False):
                self.is_companion_active = False
                self.app_state["is_companion_chat_active"] = False
                print("[COMPANION] Happiness detected. Companion chat cancelled.")
                self.alert_manager.cancel_companion_chat("happiness_alert")

        else:
            self.last_happiness_score = 0.0
            self.last_sadness_score = 0.0
            self.emotion_landmarks = None
            self.emotion_roi_dims = None

        self.trigger_companion_chat_if_needed(frame, sadness_detected_this_frame, self.last_sadness_score)

        self._update_generic_alert_status(
            frame, "happiness", happiness_detected_this_frame, self.config.EMOTION_DETECTION_CONFIRMATION_SEC,
            f"Happy ({self.last_happiness_score:.2f})", "happiness_alert", [frame] if patient_bbox else [],
            frame.shape[0] - 280
        )
        self.alert_manager.check_and_trigger_timed_alert(
            frame, "happiness", happiness_detected_this_frame, self.config.EMOTION_DETECTION_CONFIRMATION_SEC,
            f"Patient is Happy ({self.last_happiness_score:.2f})", "happiness_alert", [frame] if patient_bbox else [], 0
        )

        self._update_generic_alert_status(
            frame, "sadness", sadness_detected_this_frame, self.config.EMOTION_DETECTION_CONFIRMATION_SEC,
            f"Sad ({self.last_sadness_score:.2f})", "sadness_alert", [frame] if patient_bbox else [],
            frame.shape[0] - 310
        )

    def draw_emotion_detection_overlay(self, frame: np.ndarray):
        """LIGHT DRAWING: Draws the persistent UI elements for Emotion Detection on every frame."""
        if not self.app_state.get("emotion_detection_active", False):
            return

        y_pos = 210
        h_score = self.last_happiness_score
        s_score = self.last_sadness_score

        h_color = (0, 255, 0) if h_score > self.config.HAPPINESS_THRESHOLD else (255, 255, 255)
        s_color = (0, 0, 255) if s_score > self.config.SADNESS_THRESHOLD else (255, 255, 255)

        cv2.putText(frame, f"Happy: {h_score:.3f}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, h_color, 2)
        y_pos += 30
        cv2.putText(frame, f"Sad: {s_score:.3f}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, s_color, 2)

        if self.app_state.get("is_companion_chat_active", False):
            cv2.putText(frame, "COMPANION ACTIVE", (10, y_pos + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

        if self.emotion_landmarks and self.emotion_roi_dims:
            x1, y1, x2, y2 = self.emotion_roi_dims
            if x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0 and x2 <= frame.shape[1] and y2 <= frame.shape[0]:
                patient_roi_bgr = frame[y1:y2, x1:x2]
                self.drawing_utils.draw_landmarks(image=patient_roi_bgr, landmark_list=self.emotion_landmarks,
                                                  connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                                                  landmark_drawing_spec=self.drawing_spec,
                                                  connection_drawing_spec=self.drawing_spec)

        h_status = self.generic_alert_status.get("happiness")
        if h_status is not None and not self.alert_manager.alert_sent_flags.get("happiness", True):
            elapsed = time.time() - h_status["timestamp"]
            countdown = h_status["conf_sec"] - elapsed
            if countdown > 0:
                cv2.putText(frame, f"Happy Alert in: {int(countdown) + 1}s", (10, h_status['y_pos']),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        s_status = self.generic_alert_status.get("sadness")
        if s_status is not None and not self.alert_manager.alert_sent_flags.get("sadness", True):
            elapsed = time.time() - s_status["timestamp"]
            countdown = s_status["conf_sec"] - elapsed
            if countdown > 0:
                cv2.putText(frame, f"Companion Check in: {int(countdown) + 1}s", (10, s_status['y_pos']),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

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

        patient_face_roi = np.ascontiguousarray(rgb_frame[y1:y2, x1:x2])
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
        self.alert_manager.check_and_trigger_timed_alert(
            frame, "stroke", stroke_detected_this_frame, self.config.STROKE_CONFIRMATION_SEC,
            "Potential Stroke", "stroke_alert", [frame[y1:y2, x1:x2]] if stroke_detected_this_frame else [],
            0
        )