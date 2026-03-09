import cv2
import numpy as np
import time
import threading
from typing import List, Tuple, Dict, Any, Optional


class HazardMixin:
    """Handles object detection based hazards (Fire, Fall, Weapons, Safety Gear)."""

    def process_fire_detection(self, frame: np.ndarray, rgb_frame: np.ndarray):
        """Detects fire or smoke in the frame using a dedicated YOLO model."""
        if not self.app_state["fire_detection_active"] or self.yolo_fire_model is None:
            self.generic_alert_status["fire"] = None
            self.alert_manager.alert_timers.pop("fire", None)
            self.alert_manager.alert_sent_flags.pop("fire", None)
            return

        fire_detected_this_frame = False
        fire_images = []

        results = self.yolo_fire_model.predict(
            rgb_frame,
            imgsz=640,
            conf=self.config.FIRE_CONFIDENCE_THRESHOLD,
            verbose=False
        )

        lower_fire_classes = [c.lower() for c in self.config.FIRE_CLASSES]

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                conf = box.conf[0].item()

                if class_id < len(self.yolo_fire_model.names):
                    label = self.yolo_fire_model.names[class_id]
                else:
                    label = "Unknown Class"

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if label.lower() in lower_fire_classes:
                    fire_detected_this_frame = True
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label_text = f"{label} {conf:.2f}"
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if fire_detected_this_frame:
                    fire_images.append(frame[y1:y2, x1:x2].copy())

        if self.app_state.get("fire_detection_active", False) and fire_detected_this_frame and getattr(self.config,
                                                                                                       'FIRE_INSTANT_ALERT',
                                                                                                       False):
            if not self.alert_manager.alert_sent_flags.get("fire", False):
                self.alert_manager.trigger_alarm(
                    "CRITICAL: Fire/Smoke Detected! Immediate Action Required (INSTANT ALERT).",
                    "fire_alert", fire_images if fire_images else [frame],
                    is_dashboard_alert=True, alert_type="fire_alert"
                )
                self.alert_manager.alert_sent_flags["fire"] = True
                threading.Timer(self.config.ALERT_CONFIRMATION_SEC, self.alert_manager.alert_sent_flags.pop,
                                args=("fire", None)).start()
            return

        self._update_generic_alert_status(
            frame, "fire", fire_detected_this_frame, self.config.FIRE_CONFIRMATION_SEC,
            "FIRE/SMOKE DETECTED!", "fire_alert", fire_images if fire_images else [frame], frame.shape[0] - 250
        )

        self.alert_manager.check_and_trigger_timed_alert(
            frame, "fire", fire_detected_this_frame, self.config.FIRE_CONFIRMATION_SEC,
            "CRITICAL: Fire/Smoke Detected! Immediate Action Required.", "fire_alert",
            fire_images if fire_images else [frame], 0
        )

    def process_fall_detection(self, frame: np.ndarray, rgb_frame: np.ndarray):
        """Detects if a person has fallen using a custom YOLO model."""
        if not self.app_state["fall_detection_active"] or self.yolo_fall_model is None:
            self.alert_manager.alert_timers.pop("fall_detection", None)
            self.alert_manager.alert_sent_flags.pop("fall_detection", None)
            self.generic_alert_status["fall_detection"] = None
            return

        fall_detected_this_frame = False
        fall_images = []

        results = self.yolo_fall_model.predict(
            rgb_frame,
            imgsz=640,
            conf=self.config.YOLO_CONFIDENCE_THRESHOLD,
            verbose=False
        )

        for result in results:
            if len(result.boxes) > 0:
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    FALL_CLASS_ID = 0

                    if class_id == FALL_CLASS_ID:
                        fall_detected_this_frame = True

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        label = f"Fall {conf:.2f}"

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                        fall_images.append(frame[y1:y2, x1:x2].copy())

        self._update_generic_alert_status(
            frame, "fall_detection", fall_detected_this_frame, self.config.FALL_CONFIRMATION_SEC,
            "Fall Detected", "fall_alert", fall_images if fall_images else [frame], frame.shape[0] - 160
        )

        self.alert_manager.check_and_trigger_timed_alert(
            frame, "fall_detection", fall_detected_this_frame, self.config.FALL_CONFIRMATION_SEC,
            "CRITICAL: Fall Detected! Immediate Action Required.", "fall_alert",
            fall_images if fall_images else [frame], 0
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

        results = self.yolo_safety_model.predict(rgb_frame, conf=self.config.YOLO_CONFIDENCE_THRESHOLD, verbose=False)
        detected_gear = {cls: False for cls in self.config.YOLO_SAFETY_CLASSES}

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                label = self.yolo_safety_model.names[class_id]
                conf = box.conf[0].item()

                if label in self.config.YOLO_SAFETY_CLASSES:
                    detected_gear[label] = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 0, 255), 2)

        missing_gear = [gear for gear, detected in detected_gear.items() if not detected]
        is_violation = bool(missing_gear)

        if is_violation:
            if self.patient_bbox or self.other_people_bboxes:
                message = f"SAFETY VIOLATION: Missing {', '.join(missing_gear)}"
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