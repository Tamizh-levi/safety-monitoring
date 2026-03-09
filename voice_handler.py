import os
import time
import threading
import datetime
import logging
import json
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound


class VoiceHandler:
    def __init__(self, config, app_state, db_manager, alert_manager, core_processor):
        self.config = config
        self.app_state = app_state
        self.db_manager = db_manager
        self.alert_manager = alert_manager
        self.core_processor = core_processor

        # Initialize Speech Recognition
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()

        # Voice State
        self.mic_is_setup = False
        self.awaiting_reply = False

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

    def _handle_set_reminder(self, command: str):
        """
        Uses Ollama to extract time and message from a voice command and schedules the reminder.
        This must be run in a separate thread as it makes a blocking API call.
        """
        print(f"Attempting to process reminder command: {command}")

        # Define the desired JSON structure for the LLM response
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
            response_json_text = self.alert_manager._ollama_generate_json(
                command,
                self.config.OLLAMA_MODEL_NAME,
                system_prompt,
                json_schema
            )

            # Parse the LLM response
            response_data = json.loads(response_json_text)
            time_str = response_data.get('time_24hr')
            message = response_data.get('message')

            # Validate and save
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

        # Provide audio feedback
        self.speak_reminder(confirmation_msg)

    def listen_for_voice_commands(self):
        """
        Background thread to listen for voice commands from the patient.
        This function acquires the microphone resource once when activated and keeps it open.
        """
        # Define standard alert triggers globally for easier checking
        standard_alerts = {
            ("call manager", "call nurse"): ("Voice Command: Call Manager", "call_manager"),
            ("need water", "thirsty"): ("Voice Command: Need Water", "need_water"),
            ("cancel", "stop"): ("Voice Command: Cancel Request", "cancel_request"),
            ("help", "assistance", "support"): ("Voice Command: Worker Requesting Help", "help_request"),
            ("thank you", "thanks", "good job"): ("Voice Command: Thank You", "thank_you_response")
        }

        # New reminder trigger phrases
        reminder_triggers = ("set reminder", "schedule reminder", "remind me to")

        while self.app_state["running"]:
            # 1. WAIT FOR VOICE FEATURE TO BE ENABLED
            while not self.app_state["voice_active"] and self.app_state["running"]:
                time.sleep(1.5)

            if not self.app_state["running"]:
                break

            # 2. ACQUIRE MICROPHONE
            try:
                with self.mic as source:
                    print("Adjusting for ambient noise...")
                    self.recognizer.adjust_for_ambient_noise(source, duration=self.config.VOICE_AMBIENT_ADJUST_DUR)
                    self.recognizer.pause_threshold = self.config.VOICE_PAUSE_THRESHOLD
                    print("Voice Activated: Now listening for commands/replies/reminders.")

                    # 3. LISTENING LOOP
                    while self.app_state["voice_active"] and self.app_state["running"]:

                        # --- SYNCHRONIZATION POINT FOR COMPANION SPEECH ---
                        if self.app_state.get("is_companion_chat_active", False) and self.awaiting_reply:
                            print("[COMPANION] Mic Active: Waiting for companion speech to finish...")
                            time.sleep(1.0)
                            print("[COMPANION] Initial chat turn finished. Awaiting patient reply.")
                            self.awaiting_reply = False

                        if not self.app_state.get("is_companion_chat_active", False):
                            print("Listening for command...")

                        try:
                            audio = self.recognizer.listen(source, timeout=5,
                                                           phrase_time_limit=self.config.VOICE_PHRASE_LIMIT)
                            command = self.recognizer.recognize_google(audio, language='en-US').lower()
                            print(f"Voice Detected (HEARD): {command}")

                            # --- COMPANION REPLY LOGIC ---
                            if self.app_state.get("is_companion_chat_active", False):
                                print(f"[COMPANION] Patient Reply detected: {command}. Continuing conversation.")
                                threading.Thread(target=self.alert_manager.continue_companion_chat,
                                                 args=(command, self.config.OLLAMA_MODEL_NAME),
                                                 daemon=True).start()
                                time.sleep(1.0)
                                continue

                            # --- REMINDER LOGIC ---
                            if any(trigger in command for trigger in reminder_triggers):
                                threading.Thread(target=self._handle_set_reminder, args=(command,), daemon=True).start()
                                time.sleep(1.0)
                                continue

                            # --- COMPANION STARTUP LOGIC ---
                            if self.app_state.get("emotion_detection_active", False) and not self.app_state.get(
                                    "is_companion_chat_active", False):
                                if any(word in command for word in self.config.COMPANION_TRIGGER_WORDS):
                                    self.core_processor.is_companion_active = True
                                    self.app_state["is_companion_chat_active"] = True
                                    self.awaiting_reply = True
                                    print("[COMPANION] Voice trigger detected. Initiating user-requested chat.")

                                    prompt = f"The patient, {self.alert_manager.patient_name}, initiated a chat by saying '{command}'. Engage in a helpful and friendly conversation. Start the conversation based on the patient's command."
                                    threading.Thread(target=self.alert_manager.start_companion_chat,
                                                     args=(prompt, self.config.OLLAMA_MODEL_NAME,
                                                           self.config.COMPANION_TIMEOUT_SEC),
                                                     daemon=True).start()
                                    continue

                            # --- STANDARD ALERT LOGIC ---
                            for phrases, (message, sound_key) in standard_alerts.items():
                                if any(p in command for p in phrases):
                                    requester_id = self.config.MONITORED_PATIENT_ID
                                    requester_name = self.alert_manager.patient_name
                                    is_dashboard_alert = False

                                    if sound_key == "help_request":
                                        requester_id = self.config.MONITORED_WORKER_ID
                                        requester_name = self.config.MONITORED_WORKER_NAME
                                        message = f"Worker {requester_name} requesting urgent help."
                                        is_dashboard_alert = True

                                    self.alert_manager.trigger_alarm(
                                        message,
                                        sound_key,
                                        requester_id=requester_id,
                                        requester_name=requester_name,
                                        is_dashboard_alert=is_dashboard_alert,
                                        alert_type=sound_key
                                    )
                                    break

                        except sr.WaitTimeoutError:
                            print("Listening Timeout (NOT HEARD): Waiting for speech.")
                            time.sleep(0.5)
                        except (sr.UnknownValueError, sr.RequestError) as e:
                            print(f"Voice Detected (UNINTELLIGIBLE): Could not recognize speech. Error: {e}")
                            time.sleep(0.5)
                        except Exception as e:
                            logging.error(f"Voice recognition general error: {e}")
                            time.sleep(2)

                    print("Voice Deactivated: Microphone resource released.")

            except Exception as e:
                logging.error(f"Microphone ACQUISITION failed: {e}")
                self.app_state["voice_active"] = False
                time.sleep(5)