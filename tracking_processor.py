import cv2
import numpy as np
import face_recognition
import time
from typing import List, Tuple, Dict, Any, Optional


class TrackingMixin:
    """Handles Person Detection, CSRT Tracking, Face Rec, and Bed Exit limits."""

    def process_person_detection(self, frame: np.ndarray, rgb_frame: np.ndarray) -> Tuple[
        int, Optional[Tuple[int, int, int, int]]]:
        """
        Detects, tracks, and identifies all persons in the frame.
        Optimization: Detects patient ONCE via FaceRec/YOLO, then switches to CSRT tracking
        for massive FPS improvement.
        """
        # --- PHASE 1: TRACKING (Fast Path) ---
        if self.patient_tracker is not None:
            # CHECK TIMEOUT: If tracking for more than 15 seconds, force a re-detection
            if time.time() - self.tracking_start_time > 15.0:
                self.patient_tracker = None
                # We do NOT return here. We let it fall through to Phase 2 to re-detect immediately.
            else:
                success, box = self.patient_tracker.update(rgb_frame)
                if success:
                    x, y, w, h = map(int, box)
                    self.patient_bbox = (x, y, x + w, y + h)

                    # Visual Feedback for Tracking Mode
                    remaining_time = 15.0 - (time.time() - self.tracking_start_time)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{self.alert_manager.patient_name} (Tracked: {remaining_time:.1f}s)",
                                (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Since we are skipping YOLO, we assume only the patient is of interest.
                    # Clear intruder alerts so they don't persist falsely.
                    self.generic_alert_status["unknown"] = None
                    self.generic_alert_status["unidentified"] = None
                    self.generic_alert_status["crowd"] = None

                    # Return count=1 (Patient) and the bbox
                    return 1, self.patient_bbox
                else:
                    # Tracking lost, fall back to heavy detection
                    self.patient_tracker = None
                    self.patient_bbox = None

        # --- PHASE 2: DETECTION (Slow Path - Run only if tracker is None or Timer Expired) ---
        if self.yolo_model is None:
            return 0, None

        # Run YOLO detection
        results = self.yolo_model(rgb_frame, verbose=False, conf=self.config.YOLO_CONFIDENCE_THRESHOLD)

        person_count = 0
        all_bboxes = []
        unknown_face_detected = False
        unidentified_person_present = False
        intruder_rois = []

        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == 0:  # Class 0 is 'person'
                    person_count += 1
                    bbox = tuple(map(int, box.xyxy[0]))
                    all_bboxes.append(bbox)

        self.other_people_bboxes.clear()
        temp_patient_bbox = None
        patient_found_by_face = False

        # Attempt to identify persons via Face Recognition
        for bbox in all_bboxes:
            x1, y1, x2, y2 = bbox
            # Ensure safe cropping
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

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

        # Logic to initialize tracker
        if patient_found_by_face:
            self.patient_bbox = temp_patient_bbox
            # Initialize CSRT tracker
            self.patient_tracker = cv2.TrackerCSRT_create()
            self.patient_tracker.init(rgb_frame, (self.patient_bbox[0], self.patient_bbox[1],
                                                  self.patient_bbox[2] - self.patient_bbox[0],
                                                  self.patient_bbox[3] - self.patient_bbox[1]))
            # Start the 15-second tracking timer
            self.tracking_start_time = time.time()
        else:
            # STRICT MODE: If patient is not identified by face, do NOT track.
            self.patient_bbox = None
            self.patient_tracker = None

        if self.patient_bbox:
            x1, y1, x2, y2 = self.patient_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, self.alert_manager.patient_name if patient_found_by_face else "Subject (Monitoring)",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Update generic alert status (Only runs during Detection Phase)
        self._update_generic_alert_status(frame, "unknown", unknown_face_detected, self.config.ALERT_CONFIRMATION_SEC,
                                          "Other Person Detected", "unknown_alert", intruder_rois, frame.shape[0] - 10)
        self._update_generic_alert_status(frame, "unidentified", unidentified_person_present,
                                          self.config.UNIDENTIFIED_CONFIRMATION_SEC,
                                          "Unidentified Person", "unidentified_alert", intruder_rois,
                                          frame.shape[0] - 40)

        is_crowd = person_count > self.config.CROWD_THRESHOLD
        self._update_generic_alert_status(frame, "crowd", is_crowd, self.config.ALERT_CONFIRMATION_SEC,
                                          f"Crowd ({person_count})", "crowd_alert", [frame], frame.shape[0] - 70)

        # Trigger logic for intruder alerts
        if self.app_state["unknown_person_active"]:
            self.alert_manager.check_and_trigger_timed_alert(frame, "unknown", unknown_face_detected,
                                                             self.config.ALERT_CONFIRMATION_SEC,
                                                             "Other Person Detected", "unknown_alert", intruder_rois, 0)
        if self.app_state["unidentified_person_active"]:
            self.alert_manager.check_and_trigger_timed_alert(frame, "unidentified", unidentified_person_present,
                                                             self.config.UNIDENTIFIED_CONFIRMATION_SEC,
                                                             "Unidentified Person", "unidentified_alert", intruder_rois,
                                                             0)
        if self.app_state["crowd_alert_active"]:
            self.alert_manager.check_and_trigger_timed_alert(frame, "crowd", is_crowd,
                                                             self.config.ALERT_CONFIRMATION_SEC,
                                                             f"Crowd ({person_count})", "crowd_alert", [frame], 0)

        return person_count, self.patient_bbox

    def process_bed_exit(self, frame: np.ndarray, patient_bbox: Optional[Tuple[int, int, int, int]]):
        """Detects if the SAFETY has left a predefined bed area."""
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
        self.alert_manager.check_and_trigger_timed_alert(
            frame, "bed_exit", patient_out_of_bed, self.config.BED_EXIT_CONFIRMATION_SEC,
            "Patient Left Bed", "bed_exit_alert", [frame] if patient_out_of_bed else [], 0
        )