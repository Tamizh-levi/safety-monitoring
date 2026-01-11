import cv2
import os
import numpy as np
import face_recognition
from ultralytics import YOLO
import threading
import speech_recognition as sr
import pywhatkit
from playsound import playsound
import warnings
import datetime
import time
import logging
from pymongo import MongoClient
import bson
from gtts import gTTS
import json  # ADDED for JSON parsing LLM response

from typing import List, Tuple, Dict, Any, Optional
import sys

# Assumes other modules are in the same directory
from config import Config
from database import DatabaseManager
from alerts import AlertManager
from ui import UIManager
from SAFETY import PatientMonitorCore
from web_server import start_web_server  # <--- NEW IMPORT

# Suppress all warnings for a cleaner console output
warnings.filterwarnings("ignore")


class MainApplication:
    def __init__(self, config: Config):
        self.config = config
        self.setup_logging()

        # Central state dictionary to control application flow and features
        # NOTE: All detection features start as False here, but some are overwritten by model loading
        # to ensure the Core Processor knows which models to load. They will be reset to False below.
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
            "fire_detection_active": False,  # ADDED: Fire detection state
            "is_companion_chat_active": False,  # ADDED companion chat status for global tracking
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

        # Load models and update app state accordingly. If the model is found, the flag is set to True.
        # This informs the PatientMonitorCore about which models to load into memory.
        self.app_state['yolo_model_loaded'] = self._load_model_path(self.config.YOLO_MODEL_PATH)
        self.app_state['knife_detection_active'] = self._load_model_path(self.config.YOLO_KNIFE_MODEL_PATH)
        self.app_state['gun_detection_active'] = self._load_model_path(self.config.YOLO_GUN_MODEL_PATH)
        self.app_state['fall_detection_active'] = self._load_model_path(self.config.YOLO_FALL_MODEL_PATH)
        self.app_state['safety_detection_active'] = self._load_model_path(self.config.YOLO_SAFETY_MODEL_PATH)
        self.app_state['fire_detection_active'] = self._load_model_path(
            self.config.YOLO_FIRE_MODEL_PATH)  # ADDED: Load Fire Model

        # Initialize the core processing module
        self.core_processor = PatientMonitorCore(self.config, self.app_state, self.alert_manager)

        # --- FIX: Ensure all detection features are OFF at startup, as requested by the user ---
        # We explicitly reset these activation flags to False. The Core Processor will still be
        # able to load models if the capability check above temporarily set them to True.
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
        # --- END FIX ---

        # Share app state and core processor reference with alert manager for conversational replies
        # This is CRUCIAL for two-way communication and correct flag clearing
        self.alert_manager.set_app_state_ref(self.app_state)
        self.alert_manager.core_processor_ref = self.core_processor

        # Initialize Speech Recognition
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()

        # --- VOICE STATE (Simplified, mic acquisition handled in the thread) ---
        self.mic_is_setup = False
        # NEW: Flag set by Alerts when companion speech starts, to trigger recovery wait here.
        self.awaiting_reply = False
        # --- END VOICE STATE ---

        # State for reminders
        self.played_reminders_today = set()
        self.last_reminder_check_date = datetime.date.today()

    def setup_logging(self):
        """Configures the logging for the application."""
        logging.basicConfig(filename=self.config.LOG_FILE_PATH, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        print(f"Logging to {self.config.LOG_FILE_PATH}")

    def _load_model_path(self, path: str) -> bool:
        """Helper to safely check if a model path exists."""
        # Note: We skip the check if the path is empty, assuming an empty path means the feature is disabled
        # This prevents flooding the console with warnings if optional models are missing
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

        # Start background threads for non-blocking tasks
        threading.Thread(target=self.schedule_checker, daemon=True).start()
        threading.Thread(target=self.listen_for_voice_commands, daemon=True).start()  # Voice thread is still started
        threading.Thread(target=self.sleep_window_checker, daemon=True).start()  # ADDED Sleep Window Checker

        # --- NEW: Start Web Server Thread ---
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

        # NEW: Load mock worker information from config to alert manager
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

    def speak_reminder(self, text: str):
        """Uses Google Text-to-Speech to read a reminder aloud."""
        try:
            print(f"Speaking reminder: {text}")
            tts = gTTS(text=text, lang='en')
            filename = os.path.join(self.config.REMINDER_AUDIO_DIR, f"reminder_{int(time.time())}.mp3")
            tts.save(filename)
            playsound(filename)
            os.remove(filename)
        except Exception as e:
            print(f"Error in text-to-speech: {e}")
            logging.error(f"Error in text-to-speech: {e}")

    def schedule_checker(self):
        """Background thread to check for scheduled reminders every 30 seconds."""
        while self.app_state["running"]:
            today = datetime.date.today()
            if today != self.last_reminder_check_date:
                self.played_reminders_today.clear()
                self.last_reminder_check_date = today
                print("Resetting daily reminders.")

            reminder = self.db_manager.get_schedule_for_now(self.config.MONITORED_PATIENT_ID)
            if reminder:
                reminder_id = f"{reminder['patient_id']}-{reminder['time']}"
                if reminder_id not in self.played_reminders_today:
                    full_message = f"Hello {self.alert_manager.patient_name}, {reminder.get('message', 'you have a scheduled reminder.')}"
                    threading.Thread(target=self.speak_reminder, args=(full_message,), daemon=True).start()
                    self.played_reminders_today.add(reminder_id)
            time.sleep(30)

    def sleep_window_checker(self):
        """Background thread to check if the current time falls within the safe sleep window."""
        while self.app_state["running"]:
            current_time = datetime.datetime.now().time()
            start_str = self.config.SLEEP_START_TIME
            end_str = self.config.SLEEP_END_TIME

            # Convert time strings to datetime.time objects
            start_time = datetime.datetime.strptime(start_str, "%H:%M").time()
            end_time = datetime.datetime.strptime(end_str, "%H:%M").time()

            is_in_window = False

            if start_time < end_time:
                # Simple case: start and end are in the same day (e.g., 10:00 to 18:00)
                is_in_window = start_time <= current_time <= end_time
            else:
                # Overnight case (e.g., 22:00 to 06:00 the next day)
                # Check if time is between start (22:00) and midnight (23:59) OR between midnight (00:00) and end (06:00)
                is_in_window = current_time >= start_time or current_time <= end_time

            if is_in_window != self.app_state["is_patient_in_safe_sleep_window"]:
                self.app_state["is_patient_in_safe_sleep_window"] = is_in_window
                print(f"Safe Sleep Window changed to: {'ON' if is_in_window else 'OFF'}")

            time.sleep(10)  # Check every 10 seconds

    def _handle_set_reminder(self, command: str):
        """
        Uses Ollama to extract time and message from a voice command and schedules the reminder.
        This must be run in a separate thread as it makes a blocking API call.
        """
        print(f"Attempting to process reminder command: {command}")

        # 1. Define the desired JSON structure for the LLM response
        json_schema = {
            "type": "OBJECT",
            "properties": {
                "time_24hr": {"type": "STRING",
                              "description": "The extracted time in HH:MM 24-hour format. E.g., '14:30'"},
                "message": {"type": "STRING", "description": "The reminder message or task to be performed."}
            },
            "required": ["time_24hr", "message"]
        }

        system_prompt = (
            "You are a sophisticated AI assistant specializing in scheduling. "
            "Analyze the user's request to set a reminder. Extract the intended time (convert to 24-hour HH:MM format) "
            "and the reminder message. Ensure the time is precisely in HH:MM format. "
            "If the time is relative (e.g., 'in 1 hour'), calculate the current time plus that duration and return the final HH:MM. "
            f"The current time is {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."
        )

        try:
            # 2. Call the AlertManager's LLM generation function (assuming one exists or creating a helper)

            response_json_text = self.alert_manager._ollama_generate_json(
                command,
                self.config.OLLAMA_MODEL_NAME,
                system_prompt,
                json_schema
            )

            # 3. Parse the LLM response
            response_data = json.loads(response_json_text)
            time_str = response_data.get('time_24hr')
            message = response_data.get('message')

            # 4. Validate and save
            if time_str and message:
                if self.db_manager.add_scheduled_reminder(self.config.MONITORED_PATIENT_ID, time_str, message):
                    confirmation_msg = f"Confirmation: I have scheduled a reminder for {time_str}. The message is: {message}."
                else:
                    confirmation_msg = "I am sorry, I couldn't save the reminder to the database. Please check the connection."
            else:
                confirmation_msg = "I'm having trouble understanding the time and message. Please try saying the command again, ensuring you mention a time."

        except Exception as e:
            logging.error(f"Error processing reminder command with LLM: {e}")
            confirmation_msg = "An error occurred while trying to process your reminder request. My apologies."

        # 5. Provide audio feedback
        self.speak_reminder(confirmation_msg)

    def listen_for_voice_commands(self):
        """
        Background thread to listen for voice commands from the patient.
        This function acquires the microphone resource once when activated and keeps it open.
        """

        # Define standard alert triggers globally for easier checking
        standard_alerts = {
            # Existing alerts (CHANGED 'call nurse' to 'call manager')
            ("call manager", "call nurse"): ("Voice Command: Call Manager", "call_manager"),  # UPDATED KEY
            ("need water", "thirsty"): ("Voice Command: Need Water", "need_water"),
            ("cancel", "stop"): ("Voice Command: Cancel Request", "cancel_request"),
            # NEW: Help command - Uses worker info and is flagged for DASHBOARD
            ("help", "assistance", "support"): ("Voice Command: Worker Requesting Help", "help_request"),
            # NEW: Thank You command (non-urgent)
            ("thank you", "thanks", "good job"): ("Voice Command: Thank You", "thank_you_response")
        }

        # New reminder trigger phrases
        reminder_triggers = ("set reminder", "schedule reminder", "remind me to")

        while self.app_state["running"]:

            # 1. WAIT FOR VOICE FEATURE TO BE ENABLED
            while not self.app_state["voice_active"] and self.app_state["running"]:
                time.sleep(1.5)  # Idle sleep when voice is intentionally off

            if not self.app_state["running"]:
                break

            # 2. ACQUIRE MICROPHONE (Blocking, runs only once when voice is ON)
            try:
                with self.mic as source:
                    print("Adjusting for ambient noise...")
                    self.recognizer.adjust_for_ambient_noise(source, duration=self.config.VOICE_AMBIENT_ADJUST_DUR)
                    self.recognizer.pause_threshold = self.config.VOICE_PAUSE_THRESHOLD
                    print("Voice Activated: Now listening for commands/replies/reminders.")

                    # 3. LISTENING LOOP (Stays active as long as the resource is acquired)
                    while self.app_state["voice_active"] and self.app_state["running"]:

                        # --- SYNCHRONIZATION POINT FOR COMPANION SPEECH ---
                        if self.app_state.get("is_companion_chat_active", False) and self.awaiting_reply:
                            print("[COMPANION] Mic Active: Waiting for companion speech to finish...")

                            # **RELIABLE SYNCHRONIZATION POINT**
                            # Wait for a brief moment to ensure audio recovery after speech finishes.
                            time.sleep(1.0)

                            # Print the synchronization message now that the audio is recovered and mic is ready.
                            print("[COMPANION] Initial chat turn finished. Awaiting patient reply.")
                            self.awaiting_reply = False  # Reset flag after printing the message

                        # Status Message for Active Listening (only after potential wait)
                        if not self.app_state.get("is_companion_chat_active", False):
                            print("Listening for command...")

                        try:
                            # Listen (timeout is 5 seconds for replies/commands)
                            audio = self.recognizer.listen(source, timeout=5,
                                                           phrase_time_limit=self.config.VOICE_PHRASE_LIMIT)

                            command = self.recognizer.recognize_google(audio, language='en-US').lower()
                            print(f"Voice Detected (HEARD): {command}")

                            # --- COMPANION REPLY LOGIC (Highest Priority) ---
                            if self.app_state.get("is_companion_chat_active", False):
                                print(f"[COMPANION] Patient Reply detected: {command}. Continuing conversation.")

                                threading.Thread(target=self.alert_manager.continue_companion_chat,
                                                 args=(command, self.config.OLLAMA_MODEL_NAME),
                                                 daemon=True).start()

                                # Extended sleep after launching thread to ensure state update/recovery before next listen
                                time.sleep(1.0)
                                continue
                                # --- END COMPANION REPLY LOGIC ---

                            # --- REMINDER LOGIC ---
                            if any(trigger in command for trigger in reminder_triggers):
                                threading.Thread(target=self._handle_set_reminder, args=(command,), daemon=True).start()
                                time.sleep(1.0)  # Pause briefly to avoid immediate re-trigger
                                continue

                            # --- COMPANION STARTUP LOGIC ---
                            if self.app_state.get("emotion_detection_active", False) and not self.app_state.get(
                                    "is_companion_chat_active", False):
                                if any(word in command for word in self.config.COMPANION_TRIGGER_WORDS):
                                    self.core_processor.is_companion_active = True
                                    self.app_state["is_companion_chat_active"] = True
                                    self.awaiting_reply = True  # Signal to the mic thread that speech is coming
                                    print("[COMPANION] Voice trigger detected. Initiating user-requested chat.")

                                    prompt = f"The patient, {self.alert_manager.patient_name}, initiated a chat by saying '{command}'. Engage in a helpful and friendly conversation. Start the conversation based on the patient's command."
                                    threading.Thread(target=self.alert_manager.start_companion_chat,
                                                     args=(prompt, self.config.OLLAMA_MODEL_NAME,
                                                           self.config.COMPANION_TIMEOUT_SEC),
                                                     daemon=True).start()

                                    continue

                                    # --- STANDARD ALERT LOGIC (Lowest Priority) ---
                            for phrases, (message, sound_key) in standard_alerts.items():
                                if any(p in command for p in phrases):

                                    requester_id = self.config.MONITORED_PATIENT_ID
                                    requester_name = self.alert_manager.patient_name
                                    is_dashboard_alert = False

                                    # Check if this is the high-priority "help request" command
                                    if sound_key == "help_request":
                                        # Use Worker Info for the alert and flag it for the dashboard
                                        requester_id = self.config.MONITORED_WORKER_ID
                                        requester_name = self.config.MONITORED_WORKER_NAME
                                        message = f"Worker {requester_name} requesting urgent help."
                                        is_dashboard_alert = True

                                    # All alerts are now triggered through the full function call
                                    self.alert_manager.trigger_alarm(
                                        message,
                                        sound_key,
                                        requester_id=requester_id,
                                        requester_name=requester_name,
                                        is_dashboard_alert=is_dashboard_alert,  # Set flag for API POST
                                        alert_type=sound_key
                                    )
                                    break

                        except sr.WaitTimeoutError:
                            # Timeout: Expected when no speech is heard.
                            print("Listening Timeout (NOT HEARD): Waiting for speech.")
                            time.sleep(0.5)  # Small sleep before next listen attempt
                        except (sr.UnknownValueError, sr.RequestError) as e:
                            # Recognition error: something was heard, but it couldn't be processed.
                            print(f"Voice Detected (UNINTELLIGIBLE): Could not recognize speech. Error: {e}")
                            time.sleep(0.5)
                        except Exception as e:
                            logging.error(f"Voice recognition general error: {e}")
                            time.sleep(2)

                    # When the inner loop breaks (voice_active is False), release the mic resource.
                    print("Voice Deactivated: Microphone resource released.")

            except Exception as e:
                logging.error(f"Microphone ACQUISITION failed: {e}")
                self.app_state["voice_active"] = False  # Ensure UI reflects failure
                time.sleep(5)
                # Outer loop continues, tries to acquire mic again if feature is re-enabled/restarted

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
        last_frame_time = time.time()

        try:
            while self.app_state["running"] and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("End of video stream or camera error.")
                    break

                # --- FPS Calculation ---
                self.fps_frame_count += 1
                if self.fps_frame_count >= 30:  # Recalculate FPS every 30 frames
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

                # --- Store the original frame state BEFORE enhancement ---
                original_frame_for_detection = processed_frame.copy()

                # Apply enhancement for display purposes
                processed_frame = self.core_processor.process_frame_lighting(processed_frame)

                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

                # --- CRITICAL FIX: Run Fire Detection ON EVERY FRAME using the original frame state ---
                # Pass the original, un-enhanced frame and its RGB conversion (for YOLO processing)
                self.core_processor.process_fire_detection(original_frame_for_detection,
                                                           cv2.cvtColor(original_frame_for_detection,
                                                                        cv2.COLOR_BGR2RGB))
                # --------------------------------------------------------------------------------------

                # --- Heavy Processing (Frame Skipping) ---
                if self.frame_counter % self.config.DETECTION_INTERVAL == 0:
                    # Note: Other detections use the enhanced 'processed_frame' and 'rgb_frame' for consistency
                    # The exception is person detection, which re-runs face recognition on ROIs.
                    person_count, patient_bbox = self.core_processor.process_person_detection(processed_frame,
                                                                                              rgb_frame)
                    self.core_processor.process_knife_detection(processed_frame, rgb_frame)
                    self.core_processor.process_gun_detection(processed_frame, rgb_frame)
                    self.core_processor.process_cough_detection(processed_frame, rgb_frame, patient_bbox)
                    self.core_processor.process_gestures(processed_frame, rgb_frame)
                    self.core_processor.process_bed_exit(processed_frame, patient_bbox)
                    self.core_processor.process_fall_detection(processed_frame, rgb_frame)
                    self.core_processor.process_stroke_detection_mediapipe(processed_frame, rgb_frame, patient_bbox)
                    self.core_processor.process_drowsiness_detection(processed_frame, rgb_frame,
                                                                     patient_bbox)
                    self.core_processor.process_pain_detection(processed_frame, rgb_frame,
                                                               patient_bbox)
                    self.core_processor.process_emotion_detection(processed_frame, rgb_frame,
                                                                  patient_bbox)  # ADDED Emotion Detection
                    self.core_processor.process_safety_gear_detection(processed_frame,
                                                                      rgb_frame)  # Added Safety Gear Detection

                self.frame_counter = (self.frame_counter + 1) % self.config.DETECTION_INTERVAL

                # --- Light Processing (Every Frame) ---

                # Check for and handle the interactive drowsiness gesture (must run every frame)
                if self.core_processor.drowsiness_prompt_spoken:
                    if self.core_processor.check_drowsiness_gesture(processed_frame, rgb_frame):
                        self.core_processor._reset_drowsiness_state()

                # --- Drawing Overlays (Every Frame for Persistence) ---

                # Generic alert countdowns (NEW: ensures persistence across skipped frames)
                self.core_processor.draw_generic_alert_overlays(processed_frame)

                # Drowsiness detection persistent overlay
                self.core_processor.draw_drowsiness_overlay(processed_frame)

                # Cough detection persistent overlay
                self.core_processor.draw_cough_detection_overlay(processed_frame)

                # Pain detection persistent overlay
                self.core_processor.draw_pain_detection_overlay(processed_frame)

                # Emotion detection persistent overlay
                self.core_processor.draw_emotion_detection_overlay(processed_frame)  # ADDED Emotion overlay

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
