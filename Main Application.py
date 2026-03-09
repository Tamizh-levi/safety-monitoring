import cv2
import os
import numpy as np
import face_recognition
import threading
import warnings


import time
import logging
import sys

# Local imports
from config import Config
from database import DatabaseManager
from alerts import AlertManager
from ui import UIManager
from SAFETY import PatientMonitorCore
from web_server import start_web_server

# Module imports for separated logical layers
from voice_handler import VoiceHandler
from background_workers import BackgroundWorkers

# Suppress all warnings for a cleaner console output
warnings.filterwarnings("ignore")


class MainApplication:
    def __init__(self, config: Config):

        self.config = config
        self.setup_logging()



        # Central state dictionary to control application flow and features
        self.app_state = {
            "running": True, "bed_roi": None, "cough_detection_active": False,
            "unknown_person_active": False, "unidentified_person_active": False,
            "crowd_alert_active": False, "gestures_active": False, "voice_active": False,
            "bed_exit_active": False, "stroke_detection_active": False,
            "knife_detection_active": False, "gun_detection_active": False,
            "fall_detection_active": False,
            "drowsiness_detection_active": False,
            "pain_detection_active": False,
            "emotion_detection_active": False,
            "safety_detection_active": False,
            "fire_detection_active": False,
            "is_companion_chat_active": False,
            "is_patient_in_safe_sleep_window": False,
            "patient_was_in_bed": False,
            "known_face_encodings": [],
            "known_face_names": []
        }

        # Frame counter for performance optimization
        self.frame_counter = 0
        self.fps_start_time = time.time()
        self.fps_frame_count = 0

        # Initialize managers
        self.db_manager = DatabaseManager(config)
        self.alert_manager = AlertManager(config, self.db_manager)
        self.ui_manager = UIManager(self.app_state)

        # Load models and update app state accordingly.
        self.app_state['yolo_model_loaded'] = self._load_model_path(self.config.YOLO_MODEL_PATH)
        self.app_state['knife_detection_active'] = self._load_model_path(self.config.YOLO_KNIFE_MODEL_PATH)
        self.app_state['gun_detection_active'] = self._load_model_path(self.config.YOLO_GUN_MODEL_PATH)
        self.app_state['fall_detection_active'] = self._load_model_path(self.config.YOLO_FALL_MODEL_PATH)
        self.app_state['safety_detection_active'] = self._load_model_path(self.config.YOLO_SAFETY_MODEL_PATH)
        self.app_state['fire_detection_active'] = self._load_model_path(self.config.YOLO_FIRE_MODEL_PATH)

        # Initialize the core processing module
        self.core_processor = PatientMonitorCore(self.config, self.app_state, self.alert_manager)

        # Explicitly reset feature activations to False
        self.app_state['cough_detection_active'] = False
        self.app_state['unknown_person_active'] = False
        self.app_state['unidentified_person_active'] = False
        self.app_state['crowd_alert_active'] = False
        self.app_state['gestures_active'] = False
        self.app_state['voice_active'] = False
        self.app_state['bed_exit_active'] = False
        self.app_state['stroke_detection_active'] = False
        self.app_state['knife_detection_active'] = False
        self.app_state['gun_detection_active'] = False
        self.app_state['fall_detection_active'] = False
        self.app_state['drowsiness_detection_active'] = False
        self.app_state['pain_detection_active'] = False
        self.app_state['emotion_detection_active'] = False
        self.app_state['safety_detection_active'] = False
        self.app_state['fire_detection_active'] = False

        # Share app state and core processor reference with alert manager
        self.alert_manager.set_app_state_ref(self.app_state)
        self.alert_manager.core_processor_ref = self.core_processor

        # Initialize sub-modules
        self.voice_handler = VoiceHandler(
            self.config, self.app_state, self.db_manager,
            self.alert_manager, self.core_processor
        )
        self.bg_workers = BackgroundWorkers(
            self.config, self.app_state, self.db_manager,
            self.alert_manager, self.voice_handler
        )

    def setup_logging(self):
        """Configures the logging for the application."""
        logging.basicConfig(filename=self.config.LOG_FILE_PATH, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        print(f"Logging to {self.config.LOG_FILE_PATH}")

    def _load_model_path(self, path: str) -> bool:
        """Helper to safely check if a model path exists."""
        if not path:
            return False
        if not os.path.exists(path):
            print(f"Warning: Model file not found at {path}. Disabling related feature.")
            return False
        return True

    def initialize(self):
        """Loads data from the database and starts background services."""
        self.load_patient_data()
        self.load_known_faces_from_db()
        os.makedirs(self.config.REMINDER_AUDIO_DIR, exist_ok=True)
        os.makedirs(self.config.INTRUDER_LOGS_DIR, exist_ok=True)

        # Start background threads utilizing
        threading.Thread(target=self.bg_workers.schedule_checker, daemon=True).start()
        threading.Thread(target=self.voice_handler.listen_for_voice_commands, daemon=True).start()
        threading.Thread(target=self.bg_workers.sleep_window_checker, daemon=True).start()
        threading.Thread(target=start_web_server, args=(self.app_state, self.alert_manager), daemon=True).start()

        print("Background threads (Schedule, Voice, Sleep Window, WEB SERVER) started.")

    def load_patient_data(self):
        """Loads the monitored patient's details from the database."""
        patient_doc = self.db_manager.get_patient_details(self.config.MONITORED_PATIENT_ID)
        if patient_doc and patient_doc.get("name"):
            patient_name = patient_doc["name"]
            self.alert_manager.set_patient_info(patient_name)
            print(f"Monitoring patient: {patient_name} (ID: {self.config.MONITORED_PATIENT_ID})")
        else:
            print(f"Warning: Could not find details for patient ID: {self.config.MONITORED_PATIENT_ID}")

        self.alert_manager.set_worker_info(
            self.config.MONITORED_WORKER_ID,
            self.config.MONITORED_WORKER_NAME
        )

    def load_known_faces_from_db(self):
        """Loads and encodes faces from the database for recognition."""
        print("Loading known faces from database...")
        patients = self.db_manager.get_all_patients_with_photos()
        for patient in patients:
            patient_name, photo_data = patient.get('name'), patient.get('photo')
            if isinstance(photo_data, bytes):
                try:
                    nparr = np.frombuffer(photo_data, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if image is None: continue
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    encodings = face_recognition.face_encodings(rgb_image)
                    if encodings:
                        self.app_state["known_face_encodings"].append(encodings[0])
                        self.app_state["known_face_names"].append(patient_name)
                    else:
                        logging.warning(f"No face found in photo for patient {patient_name}")
                except Exception as e:
                    print(f"Could not process photo for {patient_name}: {e}")
                    logging.error(f"Could not process photo for {patient_name}: {e}")
        print(f"Successfully loaded {len(self.app_state['known_face_names'])} faces from the database.")
        self.core_processor.known_face_encodings = self.app_state['known_face_encodings']
        self.core_processor.known_face_names = self.app_state['known_face_names']

    def run(self):
        """The main application loop."""
        self.initialize()
        cap = cv2.VideoCapture(self.config.VIDEO_SOURCE)
        if not cap.isOpened():
            print(f"Error: Cannot open camera source: {self.config.VIDEO_SOURCE}")
            sys.exit(1)

        window_name = "Patient Room Monitoring"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.ui_manager.handle_click, param={"alert_manager": self.alert_manager})

        person_count = 0
        patient_bbox = None

        try:
            while self.app_state["running"] and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("End of video stream or camera error.")
                    break

                # --- FPS Calculation ---

                self.fps_frame_count += 1
                if self.fps_frame_count >= 30:
                    current_time = time.time()
                    elapsed_time = current_time - self.fps_start_time
                    self.current_fps = self.fps_frame_count / elapsed_time
                    self.fps_frame_count = 0

                processed_frame = cv2.resize(frame, (640, 480))

                if self.app_state["bed_exit_active"] and self.app_state["bed_roi"] is None:
                    roi = cv2.selectROI("Select Bed Area", processed_frame, fromCenter=False, showCrosshair=True)
                    if roi[2] > 0 and roi[3] > 0:
                        self.app_state["bed_roi"] = roi
                    else:
                        print("Bed area selection cancelled. Turning off Bed Exit mode.")
                        self.app_state["bed_exit_active"] = False
                    cv2.destroyWindow("Select Bed Area")
                    continue

                # Store the original frame state BEFORE enhancement
                original_frame_for_detection = processed_frame.copy()

                # Apply enhancement for display purposes
                processed_frame = self.core_processor.process_frame_lighting(processed_frame)
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

                # Fire Detection
                self.core_processor.process_fire_detection(original_frame_for_detection,
                                                           cv2.cvtColor(original_frame_for_detection,
                                                                        cv2.COLOR_BGR2RGB))

                # Heavy Processing (Frame Skipping)
                if self.frame_counter % self.config.DETECTION_INTERVAL == 0:
                    person_count, patient_bbox = self.core_processor.process_person_detection(processed_frame,
                                                                                              rgb_frame)
                    self.core_processor.process_knife_detection(processed_frame, rgb_frame)
                    self.core_processor.process_gun_detection(processed_frame, rgb_frame)

                    self.core_processor.process_cough_detection(processed_frame, rgb_frame, patient_bbox)
                    self.core_processor.process_gestures(processed_frame, rgb_frame)
                    self.core_processor.process_bed_exit(processed_frame, patient_bbox)
                    self.core_processor.process_fall_detection(processed_frame, rgb_frame)
                    self.core_processor.process_stroke_detection_mediapipe(processed_frame, rgb_frame, patient_bbox)
                    self.core_processor.process_drowsiness_detection(processed_frame, rgb_frame, patient_bbox)
                    self.core_processor.process_pain_detection(processed_frame, rgb_frame, patient_bbox)
                    self.core_processor.process_emotion_detection(processed_frame, rgb_frame, patient_bbox)
                    self.core_processor.process_safety_gear_detection(processed_frame, rgb_frame)

                self.frame_counter = (self.frame_counter + 1) % self.config.DETECTION_INTERVAL

                # Light Processing (Every Frame)
                if self.core_processor.drowsiness_prompt_spoken:
                    if self.core_processor.check_drowsiness_gesture(processed_frame, rgb_frame):
                        self.core_processor._reset_drowsiness_state()

                # Drawing Overlays
                self.core_processor.draw_generic_alert_overlays(processed_frame)
                self.core_processor.draw_drowsiness_overlay(processed_frame)
                self.core_processor.draw_cough_detection_overlay(processed_frame)
                self.core_processor.draw_pain_detection_overlay(processed_frame)
                self.core_processor.draw_emotion_detection_overlay(processed_frame)

                processed_frame = self.ui_manager.draw_buttons(processed_frame)

                # Display FPS and People Count
                cv2.putText(processed_frame, f"People: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)
                if 'current_fps' in self.__dict__ and self.current_fps is not None:
                    cv2.putText(processed_frame, f"FPS: {self.current_fps:.1f}", (processed_frame.shape[1] - 150, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow(window_name, processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.app_state["running"] = False
                    break
        finally:
            print("Shutting down...")
            self.app_state["running"] = False
            cap.release()
            cv2.destroyAllWindows()
            self.db_manager.close()


if __name__ == "__main__":
    config = Config()
    monitor = MainApplication(config)
    monitor.run()
