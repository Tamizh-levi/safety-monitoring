import cv2
import numpy as np
import mediapipe as mp
import time
import math
import threading
from typing import List, Tuple, Dict, Any, Optional


class FatigueMixin:
    """Handles Drowsiness (EAR) and Cough Detection (MAR + Pose)."""

    def _calculate_ear(self, face_landmarks) -> float:
        """Calculates the Eye Aspect Ratio (EAR) for both eyes."""
        P1 = face_landmarks.landmark[33]
        P2 = face_landmarks.landmark[133]
        P3 = face_landmarks.landmark[160]
        P4 = face_landmarks.landmark[158]
        P5 = face_landmarks.landmark[144]
        P6 = face_landmarks.landmark[153]

        P7 = face_landmarks.landmark[362]
        P8 = face_landmarks.landmark[263]
        P9 = face_landmarks.landmark[387]
        P10 = face_landmarks.landmark[385]
        P11 = face_landmarks.landmark[373]
        P12 = face_landmarks.landmark[380]

        left_vertical_dist = math.hypot(P3.x - P5.x, P3.y - P5.y) + math.hypot(P4.x - P6.x, P4.y - P6.y)
        left_horizontal_dist = math.hypot(P1.x - P2.x, P1.y - P2.y)
        left_ear = (left_vertical_dist) / (2.0 * left_horizontal_dist) if left_horizontal_dist > 0 else 0.0

        right_vertical_dist = math.hypot(P9.x - P11.x, P9.y - P11.y) + math.hypot(P10.x - P12.x, P10.y - P12.y)
        right_horizontal_dist = math.hypot(P7.x - P8.x, P7.y - P8.y)
        right_ear = (right_vertical_dist) / (2.0 * right_horizontal_dist) if right_horizontal_dist > 0 else 0.0

        return (left_ear + right_ear) / 2.0

    def process_drowsiness_detection(self, frame: np.ndarray, rgb_frame: np.ndarray,
                                     patient_bbox: Optional[Tuple[int, int, int, int]]):
        """HEAVY DETECTION: Manages the 3-stage drowsiness detection process."""
        is_drowsy_this_frame = False

        if not self.app_state.get("drowsiness_detection_active", False) or patient_bbox is None:
            self._reset_drowsiness_state()
            return

        x1, y1, x2, y2 = patient_bbox
        if x2 <= x1 or y2 <= y1:
            self._reset_drowsiness_state()
            return

        patient_roi_rgb = np.ascontiguousarray(rgb_frame[y1:y2, x1:x2])

        if patient_roi_rgb.shape[0] < 1 or patient_roi_rgb.shape[1] < 1:
            self._reset_drowsiness_state()
            return

        face_mesh_results = self.face_mesh.process(patient_roi_rgb)

        if face_mesh_results.multi_face_landmarks:
            face_landmarks = face_mesh_results.multi_face_landmarks[0]
            ear = self._calculate_ear(face_landmarks)
            self.last_ear_value = ear

            if ear < self.config.EYE_ASPECT_RATIO_THRESHOLD:
                is_drowsy_this_frame = True
        else:
            self.last_ear_value = 0.0

        if self.app_state.get("is_patient_in_safe_sleep_window", False):
            self.drowsiness_initial_timer = None
            self.drowsiness_confirmation_start_time = None
            self.drowsiness_prompt_spoken = False
            return

        current_time = time.time()
        cooldown_duration = getattr(self.config, 'DROWSINESS_COOLDOWN_SEC', 8)
        if current_time - self.last_cancellation_time < cooldown_duration:
            self.drowsiness_initial_timer = None
            return

        if is_drowsy_this_frame:
            if not self.drowsiness_prompt_spoken:
                if self.drowsiness_initial_timer is None:
                    self.drowsiness_initial_timer = current_time
                    print(f"[Drowsiness Debug] Stage 1 Started. Timer: {self.drowsiness_initial_timer:.2f}")

                elapsed_initial = current_time - self.drowsiness_initial_timer

                if elapsed_initial >= self.config.INITIAL_DROWSINESS_SEC:
                    print(f"[Drowsiness Debug] Stage 1 Expired ({elapsed_initial:.2f}s). Triggering Stage 2 Prompt.")
                    prompt_message = f"Hello {self.alert_manager.patient_name}, are you sleeping? If you are okay, please show one finger to cancel the alarm."
                    threading.Thread(target=self.alert_manager.speak_reminder, args=(prompt_message,),
                                     daemon=True).start()

                    self.drowsiness_prompt_spoken = True
                    self.drowsiness_initial_timer = None
                    self.drowsiness_confirmation_start_time = current_time
                    print(
                        f"[Drowsiness Debug] Stage 2 Confirmation Started. Prompt Spoken: {self.drowsiness_confirmation_start_time:.2f}")

            if self.drowsiness_prompt_spoken:
                elapsed_confirmation = current_time - self.drowsiness_confirmation_start_time

                if elapsed_confirmation >= self.config.GESTURE_CONFIRMATION_TIMEOUT:
                    print(f"[Drowsiness Debug] Stage 2 Timeout. Triggering FINAL ALERT!")
                    alert_message = f"Drowsiness Confirmed: Patient did not respond to prompt within {self.config.GESTURE_CONFIRMATION_TIMEOUT}s."
                    self.alert_manager.trigger_alarm(alert_message, "drowsiness_alert", [frame])
                    self._reset_drowsiness_state()
        else:
            if self.drowsiness_initial_timer is not None or self.drowsiness_prompt_spoken:
                print("[Drowsiness Debug] Condition broken. Resetting all states.")
            self._reset_drowsiness_state()
            return

    def _reset_drowsiness_state(self):
        """Resets all internal state variables for the drowsiness detection process."""
        self.drowsiness_initial_timer = None
        self.drowsiness_confirmation_start_time = None
        self.drowsiness_prompt_spoken = False

    def draw_drowsiness_overlay(self, frame: np.ndarray):
        """LIGHT DRAWING: Draws the persistent UI elements for drowsiness detection on every frame."""
        if not self.app_state.get("drowsiness_detection_active", False):
            return

        ear = self.last_ear_value
        ear_color = (0, 255, 0)
        y_pos = 60

        cooldown_duration = getattr(self.config, 'DROWSINESS_COOLDOWN_SEC', 8)
        current_time = time.time()

        if ear > 0.0:
            if ear < self.config.EYE_ASPECT_RATIO_THRESHOLD:
                ear_color = (0, 0, 255)

            cv2.putText(frame, f"EAR: {ear:.3f}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ear_color, 2)
            y_pos += 30

        if self.app_state.get("is_patient_in_safe_sleep_window", False):
            cv2.putText(frame, "Drowsiness OFF (Scheduled Sleep)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 165, 0), 2)
            y_pos += 30

        if current_time - self.last_cancellation_time < cooldown_duration:
            cooldown_remaining = int(cooldown_duration - (current_time - self.last_cancellation_time)) + 1
            cv2.putText(frame, f"COOLDOWN: {cooldown_remaining}s", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 0), 2)
            y_pos += 30

        if self.drowsiness_initial_timer is not None and not self.drowsiness_prompt_spoken:
            elapsed_initial = time.time() - self.drowsiness_initial_timer
            countdown = self.config.INITIAL_DROWSINESS_SEC - elapsed_initial

            if countdown > 0:
                cv2.putText(frame, f"Drowsiness Check: {int(countdown) + 1}s",
                            (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                y_pos += 30

        if self.drowsiness_prompt_spoken and self.drowsiness_confirmation_start_time is not None:
            elapsed_confirmation = time.time() - self.drowsiness_confirmation_start_time
            countdown = self.config.GESTURE_CONFIRMATION_TIMEOUT - elapsed_confirmation
            prompt_color = (0, 165, 255)

            if countdown > 0:
                cv2.putText(frame, "AWAITING GESTURE RESPONSE", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            prompt_color, 2)
                y_pos += 30
                cv2.putText(frame, f"Alert in: {int(countdown) + 1}s",
                            (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, prompt_color, 2)

    def process_cough_detection(self, frame: np.ndarray, rgb_frame: np.ndarray,
                                patient_bbox: Optional[Tuple[int, int, int, int]]):
        """HEAVY DETECTION: Detects mouth opening and body motion for cough, and updates internal state."""
        if not self.app_state["cough_detection_active"]:
            self.cough_count = 0
            self.last_cough_time = 0
            self.cough_alert_sent = False
            self.last_face_landmarks = None
            self.last_mar_value = 0.0
            self.last_patient_roi_dims = None
            return

        if patient_bbox is None or (
                self.last_cough_time > 0 and time.time() - self.last_cough_time > self.config.COUGH_RESET_SEC):
            self.cough_count = 0
            self.cough_alert_sent = False

        if patient_bbox is None: return

        x1, y1, x2, y2 = patient_bbox
        if x2 <= x1 or y2 <= y1: return

        patient_roi_rgb = np.ascontiguousarray(rgb_frame[y1:y2, x1:x2])

        if patient_roi_rgb.shape[0] < 1 or patient_roi_rgb.shape[1] < 1: return

        self.last_patient_roi_dims = (x1, y1, x2, y2)

        pose_results = self.pose.process(patient_roi_rgb)
        face_mesh_results = self.face_mesh.process(patient_roi_rgb)

        mouth_open = False
        head_forward = False
        body_forward = False
        mar = 0.0

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

        y_pos = 90
        cv2.putText(frame, f"Cough Count: {self.cough_count}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255),
                    2)

        if self.last_face_landmarks and self.last_patient_roi_dims:
            x1, y1, x2, y2 = self.last_patient_roi_dims
            patient_roi_bgr = frame[y1:y2, x1:x2]

            self.drawing_utils.draw_landmarks(image=patient_roi_bgr, landmark_list=self.last_face_landmarks,
                                              connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                                              landmark_drawing_spec=self.drawing_spec,
                                              connection_drawing_spec=self.drawing_spec)

            mar = self.last_mar_value
            mouth_open = mar > self.config.MOUTH_ASPECT_RATIO_THRESHOLD

            cv2.putText(frame, f"MAR: {mar:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255) if mouth_open else (255, 255, 255), 2)