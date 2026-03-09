import cv2
import numpy as np
import mediapipe as mp
import time
from typing import List, Tuple, Dict, Any, Optional


class GestureMixin:
    """Handles Hand gestures for manual help and Drowsiness Cancellation."""

    def process_gestures(self, frame: np.ndarray, rgb_frame: np.ndarray):
        """Detects hand gestures for calling for help."""
        if not self.app_state["gestures_active"]: return
        results_hands = self.hands.process(rgb_frame)
        if results_hands.multi_hand_landmarks:
            for hand_landmarks, handedness_info in zip(results_hands.multi_hand_landmarks,
                                                       results_hands.multi_handedness):
                fingers_up = self._count_fingers(hand_landmarks, handedness_info)
                cv2.putText(frame, f"Fingers: {fingers_up}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                            2)
            if fingers_up in self.gesture_actions:
                label, sound_key = self.gesture_actions[fingers_up]
                cv2.putText(frame, f"Gesture: {label}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255),
                            2)
                current_time = time.time()
                if fingers_up not in self.gesture_detected_time: self.gesture_detected_time[fingers_up] = current_time
                elapsed = current_time - self.gesture_detected_time[fingers_up]
                if elapsed > self.config.GESTURE_CONFIRMATION_SEC:
                    is_dash_alert = True
                    self.alert_manager.trigger_alarm(
                        f"Gesture: {label}",
                        sound_key,
                        images=[frame],
                        is_dashboard_alert=is_dash_alert,
                        alert_type=sound_key
                    )
                    self.gesture_detected_time.pop(fingers_up, None)
                else:
                    countdown = self.config.GESTURE_CONFIRMATION_SEC - elapsed
                    cv2.putText(frame, f"in {int(countdown) + 1}s", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 255), 2)
            else:
                self.gesture_detected_time.clear()
        else:
            self.gesture_detected_time.clear()

    def check_drowsiness_gesture(self, frame: np.ndarray, rgb_frame: np.ndarray) -> bool:
        """
        LIGHT DETECTION: Checks for the 'one finger' gesture only during the drowsiness confirmation phase.
        Returns True if confirmation gesture is made.
        """
        if not self.drowsiness_prompt_spoken:
            return False

        results_hands = self.hands.process(rgb_frame)
        if results_hands.multi_hand_landmarks:
            for hand_landmarks, handedness_info in zip(results_hands.multi_hand_landmarks,
                                                       results_hands.multi_handedness):
                fingers_up = self._count_fingers(hand_landmarks, handedness_info)

                if fingers_up == 1:
                    cv2.putText(frame, "CONFIRMATION GESTURE DETECTED!", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 255, 0), 2)
                    print("[Drowsiness Debug] Confirmation Gesture (1 finger) detected! Resetting state.")
                    self.last_cancellation_time = time.time()
                    return True
        return False

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
