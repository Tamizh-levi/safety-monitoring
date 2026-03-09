import cv2
import numpy as np
import time
from typing import List, Tuple, Dict, Any, Optional


class CoreUtilsMixin:
    """Contains utility methods shared across the PatientMonitorCore."""

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