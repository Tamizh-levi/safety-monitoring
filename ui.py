import cv2
import numpy as np
from typing import Dict, Any, Optional
import sys


# Assume app_state is a shared dictionary
class UIManager:
    """
    Handles drawing UI elements like buttons and processing user input.
    """

    def __init__(self, app_state: Dict[str, Any]):
        self.app_state = app_state
        self.buttons: Dict[str, Dict[str, Any]] = {}

    def draw_buttons(self, frame: np.ndarray) -> np.ndarray:
        """Draws all interactive toggle buttons on the frame with correct alignment."""
        self.buttons.clear()

        # FULL BUTTON LIST based on your requirements
        button_definitions = {
            # --- LEFT COLUMN ---
            "unidentified_person": ("Unidentified", "left"),
            "unknown_person": ("Unknown Face", "left"),
            "bed_exit": ("ZONE EXIT", "left"),
            "fall_detection": ("Fall", "left"),
            "drowsiness_detection": ("Drowsiness", "left"),
            "pain_detection": ("Pain", "left"),
            "stroke_detection": ("Stroke", "left"),
            "emotion_detection": ("Emotion", "left"),
            "safety_detection": ("Safety Gear", "left"),

            # --- RIGHT COLUMN ---
            "fire_detection": ("Fire/Smoke", "right"),
            "crowd_alert": ("Crowd", "right"),
            "gestures": ("Gestures", "right"),
            "voice": ("Voice", "right"),
            "cough_detection": ("Cough", "right"),
            "gun_detection": ("Gun Detect", "right"),  # Added
            "knife_detection": ("Knife Detect", "right")  # Added
        }

        # Calculate layout
        # We have about 9 buttons on the left. 9 * 35 = 315px.
        # Screen height is typically 480. So we start around y=100 or so to fit them?
        # Let's align them starting from the bottom up to ensure they don't overlap with header text.
        # Or top-down below the header.

        # Let's use a fixed start_y that ensures they fit.
        # If we have many buttons, we might need to adjust the y_step or font size.
        start_y = 60  # Start below the header info
        y_step = 35  # 30px button + 5px gap

        y_offsets = {"left": start_y, "right": start_y}

        # Iterate through all buttons to draw them
        for key, (name, position) in button_definitions.items():
            is_active = self.app_state.get(f"{key}_active", False)
            text = f"{name}: {'ON' if is_active else 'OFF'}"
            # Green for ON, Orange for OFF
            color = (0, 255, 0) if is_active else (0, 165, 255)

            (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            rect_w = text_width + 10  # slightly tighter padding
            rect_h = 25  # slightly shorter buttons

            if position == "left":
                rect_x, rect_y = 10, y_offsets[position]
            else:
                rect_x, rect_y = frame.shape[1] - rect_w - 10, y_offsets[position]

            rect = (rect_x, rect_y, rect_w, rect_h)
            self.buttons[key] = {"rect": rect}

            # Draw rectangle (fill)
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), color, -1)
            # Draw text
            cv2.putText(frame, text, (rect_x + 5, rect_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Increment the offset for the next button in this column
            y_offsets[position] += (rect_h + 5)

        return frame

    def handle_click(self, event, x: int, y: int, flags, param):
        """OpenCV mouse callback to toggle application modes based on button clicks."""
        if event == cv2.EVENT_LBUTTONDOWN:
            for key, button in self.buttons.items():
                bx, by, bw, bh = button["rect"]
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    state_key = f"{key}_active"
                    # Toggle the state
                    self.app_state[state_key] = not self.app_state.get(state_key, False)
                    print(f"Toggled {key}: {'ON' if self.app_state[state_key] else 'OFF'}")

                    # Special handling for bed exit mode
                    if key == "bed_exit":
                        if self.app_state[state_key] and self.app_state.get("bed_roi") is None:
                            print("Please select the bed area for exit detection.")
                        elif not self.app_state[state_key]:
                            self.app_state["patient_was_in_bed"] = False

                    # Reset active alerts if feature is turned OFF
                    if not self.app_state[state_key]:
                        self._reset_alerts_for_feature(key, param["alert_manager"])

                    break

    def _reset_alerts_for_feature(self, feature_key: str, alert_manager):
        """Helper to clear timers and flags when a feature is disabled."""
        alert_type_mapping = {
            "unknown_person": "unknown",
            "unidentified_person": "unidentified",
            "bed_exit": "bed_exit",
            "stroke_detection": "stroke",
            "knife_detection": "knife",
            "gun_detection": "gun",
            "fall_detection": "fall_detection",
            "cough_detection": "cough",
            "crowd_alert": "crowd",
            "drowsiness_detection": "drowsiness",
            "pain_detection": "pain",
            "safety_detection": "safety_violation",
            "fire_detection": "fire",
            "emotion_detection": "happiness"  # clears sadness too
        }

        alert_key = alert_type_mapping.get(feature_key)

        if feature_key == "emotion_detection":
            alert_manager.alert_timers.pop("happiness", None)
            alert_manager.alert_sent_flags.pop("happiness", None)
            alert_manager.alert_timers.pop("sadness", None)
            alert_manager.alert_sent_flags.pop("sadness", None)
        elif alert_key:
            alert_manager.alert_timers.pop(alert_key, None)
            alert_manager.alert_sent_flags.pop(alert_key, None)