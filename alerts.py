import os
import datetime
import time
import logging
import threading
import pywhatkit
from playsound import playsound
import numpy as np
import cv2
import json
import requests  # REQUIRED for making HTTP requests to the dashboard API
import socket    # REQUIRED for sending notifications to Port 2000
from gtts import gTTS
from typing import List, Tuple, Dict, Any, Optional

# NEW: Import the dedicated Ollama client
try:
    import ollama
except ImportError:
    # Fallback/warning if ollama library is not installed
    print("WARNING: 'ollama' Python library not found. Ollama companion may fail.")
    ollama = None

# Assume other modules are in the same directory
from config import Config
from database import DatabaseManager
from sms_sender import TwilioSMSSender  # Import the SMS sender


class AlertManager:
    """Handles triggering alarms, sending notifications, and managing alert states."""

    def __init__(self, config: Config, db_manager: DatabaseManager):
        self.config = config
        self.db_manager = db_manager
        self.patient_id = config.MONITORED_PATIENT_ID
        self.patient_name = "N/A"
        self.app_state = None  # To be set by MainApplication
        self.core_processor_ref = None  # NEW: Reference to PatientMonitorCore (for flag clearing)

        # New Worker Info fields for Voice/Gesture-based help requests
        self.worker_id = "WORKER_UNKNOWN"
        self.worker_name = "Unknown Staff"

        # Initialize the Ollama client
        self.ollama_client = None
        if ollama:
            try:
                # Use the configured API URL (which is set to localhost:11434 by default)
                ollama_host = self.config.OLLAMA_API_URL.replace("/api/generate", "")
                self.ollama_client = ollama.Client(host=ollama_host)
                print("Ollama client initialized successfully.")
            except Exception as e:
                logging.error(f"Failed to initialize Ollama client: {e}")
                self.ollama_client = None

        # Initialize the SMS sender with credentials from the config
        self.sms_sender = TwilioSMSSender(
            self.config.TWILIO_ACCOUNT_SID,
            self.config.TWILIO_AUTH_TOKEN,
            self.config.TWILIO_PHONE_NUMBER
        )

        # Timers and flags to manage alerts that require confirmation over time
        # ADDED: "fire"
        self.alert_timers = {
            "unknown": None, "unidentified": None, "crowd": None,
            "bed_exit": None, "stroke": None, "knife": None, "gun": None, "cough": None,
            "fall_detection": None, "drowsiness": None, "pain": None,
            "happiness": None, "sadness": None,
            "safety_violation": None, "fire": None,
            "help_request": None, "thank_you_response": None
        }
        # ADDED: "fire"
        self.alert_sent_flags = {
            "unknown": False, "unidentified": False, "crowd": False,
            "bed_exit": False, "stroke": False, "knife": False, "gun": False, "cough": False,
            "fall_detection": False, "drowsiness": False, "pain": False,
            "happiness": False, "sadness": False,
            "safety_violation": False, "fire": False,
            "help_request": False, "thank_you_response": False
        }

        # Ollama related attributes (from selection)
        self.is_chatting = False
        self.chat_history = []
        self.chat_lock = threading.Lock()
        self.stop_chat_flag = threading.Event()

    def set_app_state_ref(self, app_state: Dict[str, Any]):
        """Stores the global application state reference for cross-thread updates."""
        self.app_state = app_state

    def set_patient_info(self, name: str):
        """Sets the name of the monitored patient."""
        self.patient_name = name

    def set_worker_info(self, worker_id: str, worker_name: str):
        """Sets the ID and Name of the worker currently being recognized."""
        self.worker_id = worker_id
        self.worker_name = worker_name

    def speak_reminder(self, text: str):
        """Uses Google Text-to-Speech to read a reminder aloud."""
        try:
            print(f"Speaking reminder: {text}")
            tts = gTTS(text=text, lang='en')
            # Use a generic filename based on time to avoid threading conflicts
            filename = os.path.join(self.config.REMINDER_AUDIO_DIR, f"reminder_{int(time.time())}.mp3")
            tts.save(filename)

            # NOTE: MainApplication.py handles threading this function.
            playsound(filename)
            os.remove(filename)
        except Exception as e:
            print(f"Error in text-to-speech: {e}")
            logging.error(f"Error in text-to-speech: {e}")

    def _send_dashboard_alert_request(self, message: str, requester_id: str, requester_name: str, alert_type: str):
        """
        Private method to send an HTTP POST request to the external dashboard API.
        This must be run in a separate thread.
        """
        if not self.config.DASHBOARD_URL:
            logging.warning("DASHBOARD_URL is not configured. Skipping API alert.")
            return

        payload = {
            "alert_type": alert_type,
            "message": message,
            "timestamp": datetime.datetime.now().isoformat(),
            "patient_id": self.patient_id,
            "patient_name": self.patient_name,
            "requester_id": requester_id,
            "requester_name": requester_name,
        }

        try:
            # Send the HTTP POST request to the external API
            response = requests.post(self.config.DASHBOARD_URL, json=payload, timeout=5)

            # --- CORRECTION: Print confirmation in the monitoring terminal ---
            if response.status_code == 200 or response.status_code == 201:
                logging.info(f"Dashboard alert sent successfully to {self.config.DASHBOARD_URL}.")
                print(f"✅ Alert sent to URL successfully! Status: {response.status_code}")
            else:
                logging.error(
                    f"Failed to send dashboard alert to {self.config.DASHBOARD_URL}. Status: {response.status_code}")
                print(f"❌ Failed to send alert to URL. Status: {response.status_code}")
            # --- END CORRECTION ---

        except requests.exceptions.RequestException as e:
            logging.error(f"Error connecting to dashboard API at {self.config.DASHBOARD_URL}: {e}")
            print(f"❌ Error connecting to dashboard URL: {e}")

    def send_whatsapp_alert(self, message: str, requester_id: str, requester_name: str):
        """
        Sends an alert message via WhatsApp using pywhatkit.
        UPDATED to include dynamic requester information.
        """
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            full_message = (
                f"*** Patient Room Monitoring Alert ***\n\n"
                f"Monitored Patient: {self.patient_name} (ID: {self.patient_id})\n"
                f"Requester/Source: {requester_name} (ID: {requester_id})\n\n"  # NEW: Use dynamic requester info
                f"Timestamp: {timestamp}\n"
                f"Alert Details: {message}\n\n"
                f"This is an automated alert."
            )
            print(f"Attempting to send WhatsApp message: {message}")
            # Note: pywhatkit.sendwhatmsg_instantly is assumed, though the method name might vary based on library version.
            pywhatkit.sendwhatmsg_instantly(
                self.config.RECEIVER_PHONE_WHATSAPP,
                full_message,
                wait_time=15,
                tab_close=True
            )
        except Exception as e:
            print(f"Failed to send WhatsApp message: {e}")

    # --- NEW: Port Notification Method ---
    def send_port_notification(self, message: str, alert_type: str, requester_id: str):
        """
        Sends a JSON-formatted notification to the configured TCP port (default 2000).
        This connects, sends data, and closes the connection.
        """
        if not getattr(self.config, 'ENABLE_PORT_NOTIFICATIONS', False):
            return

        host = getattr(self.config, 'NOTIFICATION_HOST', '127.0.0.1')
        port = getattr(self.config, 'NOTIFICATION_PORT', 2000)

        data = {
            "type": alert_type,
            "message": message,
            "patient_id": self.patient_id,
            "requester_id": requester_id,
            "timestamp": datetime.datetime.now().isoformat()
        }

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2) # 2 second timeout to prevent blocking
                s.connect((host, port))
                s.sendall(json.dumps(data).encode('utf-8'))
                print(f"✅ Notification sent to Port {port}: {alert_type}")
        except Exception as e:
            # Log error but don't crash if listener is not running
            logging.error(f"Failed to send notification to port {port}: {e}")
            # Optional: print(f"❌ Port {port} connection failed: {e}")
    # --- END NEW METHOD ---

    def trigger_alarm(self, message: str, sound_key: Optional[str] = None, images: Optional[List[np.ndarray]] = None,
                      requester_id: Optional[str] = None, requester_name: Optional[str] = None,
                      is_dashboard_alert: bool = True,
                      alert_type: str = "general_alert"):  # MODIFIED: Set is_dashboard_alert default to True
        """
        The main alarm function: logs, notifies, saves images, and plays sound.
        UPDATED to accept requester_id/name, is_dashboard_alert flag, and alert_type.
        """

        # Use provided requester info, or fall back to patient info
        final_requester_id = requester_id if requester_id is not None else self.patient_id
        final_requester_name = requester_name if requester_name is not None else self.patient_name

        print(f"ALARM TRIGGERED: {message} (Source: {final_requester_name})")

        # Log event with dynamic requester ID/Name and dashboard flag
        # NOTE: We use the default value for is_dashboard_alert here unless explicitly set to False by the caller.
        self.db_manager.log_event(self.patient_id, self.patient_name, message, images,
                                  requester_id=final_requester_id, requester_name=final_requester_name,
                                  is_dashboard_alert=is_dashboard_alert)

        # Alerts triggered by a voice command are typically momentary and don't require confirmation,
        # so we immediately mark them as "sent" to prevent re-triggering within the alert loop,
        # although voice commands handle their own re-trigger logic via the listening loop's speed.
        if sound_key in self.alert_sent_flags:
            self.alert_sent_flags[sound_key] = True

        if images:
            self._save_images(message, images)

        # Run notifications in separate threads to avoid blocking the main loop
        threading.Thread(target=self.send_whatsapp_alert,
                         args=(message, final_requester_id, final_requester_name),
                         daemon=True).start()

        # START SMS INTEGRATION: Send SMS alert via the TwilioSMSSender
        # NEW: SMS message uses dynamic requester info
        sms_message = (
            f"Alert: {message} (Monitored Patient: {self.patient_name} / "
            f"Source: {final_requester_name} - ID: {final_requester_id})"
        )
        threading.Thread(
            target=self.sms_sender.send_sms,
            args=(self.config.RECEIVER_PHONE_SMS, sms_message),
            daemon=True
        ).start()
        # END SMS INTEGRATION

        # DASHBOARD API CALL (NOW TRIGGERED BY DEFAULT IF is_dashboard_alert IS TRUE)
        if is_dashboard_alert:
            threading.Thread(
                target=self._send_dashboard_alert_request,
                args=(message, final_requester_id, final_requester_name, alert_type),
                daemon=True
            ).start()
        # END DASHBOARD API CALL

        # --- NEW: SEND TO PORT 2000 ---
        # Launches in a separate thread to avoid blocking detection logic
        threading.Thread(
            target=self.send_port_notification,
            args=(message, alert_type, final_requester_id),
            daemon=True
        ).start()
        # --- END NEW ---

        if sound_key and sound_key in self.config.ALARM_SOUNDS:
            sound_file = self.config.ALARM_SOUNDS[sound_key]
            if os.path.exists(sound_file):
                # Run playsound in its own thread to ensure main application responsiveness
                threading.Thread(target=playsound, args=(sound_file,), daemon=True).start()
            else:
                logging.warning(f"Sound file not found: {sound_file}")

    def _save_images(self, message: str, images: List[np.ndarray]):
        """Saves snapshots associated with an alert to the intruder logs directory."""
        for i, img in enumerate(images):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(
                self.config.INTRUDER_LOGS_DIR,
                f"{message.replace(' ', '_')}_{timestamp}_{i}.jpg"
            )
            try:
                cv2.imwrite(filename, img)
                print(f"Saved snapshot: {filename}")
            except Exception as e:
                print(f"Error saving image snapshot: {e}")

    def check_and_trigger_timed_alert(self, frame: np.ndarray, alert_type: str, condition: bool, conf_sec: int,
                                      message: str, sound_key: str, images: Optional[List[np.ndarray]], y_pos: int):
        """
        Generic handler for alerts that trigger only after a condition is met for a
        specified confirmation period (conf_sec).
        """
        timer = self.alert_timers.get(alert_type)
        sent_flag = self.alert_sent_flags.get(alert_type)

        if condition:
            # If the condition is met, start or continue the timer
            if timer is None:
                self.alert_timers[alert_type] = time.time()

            elapsed = time.time() - self.alert_timers[alert_type]

            # If the timer exceeds the confirmation time and no alert has been sent, trigger it
            if not sent_flag and elapsed > conf_sec:
                # Timed alerts are always related to the monitored patient
                # is_dashboard_alert is now TRUE by default in trigger_alarm()

                # NOTE: For historical context, certain high-priority alerts were often flagged true here,
                # but now all will be true unless specifically excluded by passing is_dashboard_alert=False.
                # is_dash_alert = alert_type in ["fall_detection", "stroke", "knife_alert", "gun_alert", "safety_violation", "fire_alert"]

                self.trigger_alarm(message, sound_key, images,
                                   requester_id=self.patient_id, requester_name=self.patient_name,
                                   is_dashboard_alert=True, alert_type=alert_type)
                self.alert_sent_flags[alert_type] = True
            # REMOVED: elif not sent_flag: countdown drawing logic removed here

        else:
            # If the condition is no longer met, reset the timer and flag
            self.alert_timers[alert_type] = None
            self.alert_sent_flags[alert_type] = False

    # --- NEW OLLAMA JSON HELPER METHOD ---
    def _ollama_generate_json(self, prompt: str, model_name: str, system_prompt: str, schema: dict) -> str:
        """
        Generates a structured JSON response from Ollama based on a schema.
        This method is designed for tasks like reminder extraction.
        """
        if self.ollama_client is None:
            logging.error("Ollama client is not initialized for JSON generation.")
            return json.dumps({"time_24hr": None, "message": "Ollama service unavailable."})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.ollama_client.chat(
                model=model_name,
                messages=messages,
                format="json",
                options={'num_predict': 128}
            )
            # The response content will be a JSON string
            return response.get("message", {}).get("content", '{"time_24hr": null, "message": "Empty response"}')

        except Exception as e:
            logging.error(f"Ollama API Error during JSON generation: {e}")
            return json.dumps({"time_24hr": None, "message": f"API Error: {e}"})

    # --- END OLLAMA JSON HELPER METHOD ---

    # --- EXISTING OLLAMA COMPANION METHODS (Non-JSON) ---
    def check_confirmation_status(self, alert_type: str, condition: bool, conf_sec: int) -> bool:
        """Helper to check if a condition has been confirmed for the alert time. (From selection)"""
        current_time = time.time()

        if condition:
            if alert_type not in self.alert_timers or self.alert_timers[alert_type] is None:
                self.alert_timers[alert_type] = current_time

            elapsed = current_time - self.alert_timers[alert_type]
            if elapsed >= conf_sec:
                return True
        else:
            self.alert_timers[alert_type] = None

        return False

    def speak_ollama_response(self, text: str):
        """Uses gTTS to convert the LLM response to speech NON-BLOCKINGLY. (From selection)"""
        if self.stop_chat_flag.is_set():
            return

        try:
            print(f"[OLLAMA RESPONSE]: {text}")
            tts = gTTS(text=text, lang='en')
            # Use a unique temporary filename
            filename = os.path.join(self.config.REMINDER_AUDIO_DIR, f"ollama_response_{int(time.time())}.mp3")
            tts.save(filename)

            # VITAL FIX: Start a nested thread to handle the blocking playsound call
            def play_and_cleanup(file_path):
                try:
                    # playsound is blocking, but this is contained within this dedicated thread
                    playsound(file_path)
                except Exception as e:
                    logging.error(f"Error playing sound in nested thread: {e}")
                finally:
                    # Clean up the audio file immediately after playback is done
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        logging.error(f"Error deleting audio file: {e}")

            # Start the playback and cleanup in a separate, non-blocking thread
            threading.Thread(target=play_and_cleanup, args=(filename,), daemon=True).start()

        except Exception as e:
            logging.error(f"Error in Ollama TTS or playback: {e}")
            print(f"Error in Ollama TTS or playback: {e}")

    def cancel_companion_chat(self, reason: str):
        """Sets a flag to stop the ongoing chat thread. (From selection)"""
        with self.chat_lock:
            if self.is_chatting:
                self.stop_chat_flag.set()
                self.is_chatting = False
                # Ensure the core processor flag is also cleared
                if self.core_processor_ref:
                    self.core_processor_ref.is_companion_active = False
                # Ensure the app state flag is also cleared
                if self.app_state and 'is_companion_chat_active' in self.app_state:
                    self.app_state["is_companion_chat_active"] = False
                # Clear the synchronization flag in MainApplication
                if self.app_state and self.app_state.get('awaiting_reply') is not None:
                    self.app_state['awaiting_reply'] = False

                print(f"[COMPANION] Chat cancelled due to: {reason}")

    def continue_companion_chat(self, patient_reply: str, model_name: str):
        """
        Handles the patient's reply to an ongoing conversation.
        """
        with self.chat_lock:
            if not self.is_chatting or self.stop_chat_flag.is_set():
                print("[COMPANION] Cannot continue chat: Not currently active or stopped.")
                return

            self.chat_history.append({"role": "user", "content": patient_reply})

        try:
            # This is the blocking call (LLM generation)
            response_text = self._ollama_generate(model_name, self.chat_history)

            # This queues the audio non-blockingly and returns instantly
            self.speak_ollama_response(response_text)

            self.chat_history.append({"role": "assistant", "content": response_text})

            # Check if the model is ending the conversation (e.g., if response is short or definitive)
            # End after 3 turns from patient (User, Assistant, User, Assistant, User, Assistant)
            if len(self.chat_history) >= 6:
                print("[COMPANION] Conversation ended automatically after 3 turns.")
                # We raise an exception here to immediately trigger the error handling/cancellation below
                raise Exception("Max turns reached")

            # If successful, signal the MainApplication voice thread to prepare for the next listen
            if self.app_state and self.app_state.get('awaiting_reply') is not None:
                self.app_state['awaiting_reply'] = True


        except Exception as e:
            logging.error(f"Ollama continue chat error: {e}")
            print(f"[OLLAMA CHAT] Conversation closed: {e}")
            # If any error occurs, treat it as the end of the conversation
            self.cancel_companion_chat("Conversation error or max turns reached")

    def start_companion_chat(self, initial_prompt: str, model_name: str, timeout: int):
        """
        Initiates a non-blocking conversation with the Ollama model.
        This function handles the prompt, API call, response processing, and TTS. (From selection)
        """
        # Ensure only one thread enters the setup block
        with self.chat_lock:
            if self.is_chatting:
                return  # Already chatting

            self.is_chatting = True
            self.stop_chat_flag.clear()  # Clear stop flag for new chat

            # Immediately allow the main thread to run before the blocking Ollama call starts.
            time.sleep(0.01)
            self.chat_history.clear()
            self.chat_history.append({"role": "system",
                                      "content": "You are MediMind, a compassionate and supportive emotional companion for a patient. Your purpose is to check on their well-being, listen actively, and provide encouragement or suggestions. Keep responses concise and focus on the patient's emotional state. End your response with a question or a gentle invitation to talk more."})

            # Add initial patient-generated prompt/trigger
            self.chat_history.append({"role": "user", "content": initial_prompt})

        try:
            # First turn of conversation (Proactive check-in)
            if self.stop_chat_flag.is_set(): return

            # This is the blocking call (LLM generation)
            response_text = self._ollama_generate(model_name, self.chat_history)

            # This queues the audio non-blockingly and returns instantly
            self.speak_ollama_response(response_text)

            self.chat_history.append({"role": "assistant", "content": response_text})

            # The chat remains active and waits for the patient's voice reply.
            # CRITICAL: Signal the MainApplication voice thread that speech has been initiated
            if self.app_state and self.app_state.get('awaiting_reply') is not None:
                self.app_state['awaiting_reply'] = True


        except Exception as e:
            logging.error(f"Ollama chat loop error: {e}")
            print(f"[OLLAMA CHAT] An error occurred: {e}")
            # If an error occurs during the first turn, we cancel the chat immediately
            self.cancel_companion_chat("Initial chat turn error.")

        finally:
            # If the chat started successfully, the conversation continues.
            # If an error/cancellation occurred, self.cancel_companion_chat handles cleanup.
            pass

    def _ollama_generate(self, model_name: str, messages: list) -> str:
        """Makes the API call to Ollama using the dedicated Python client. (Non-JSON)"""
        # Ensure the client is initialized
        if self.ollama_client is None:
            logging.error("Ollama client is not initialized.")
            return "I apologize, but my assistance service is not configured correctly."

        try:
            # Use the ollama.chat method
            response = self.ollama_client.chat(
                model=model_name,
                messages=messages,
                options={'num_predict': 128}  # Use options for faster response/shorter answers
            )

            # Extract the assistant's reply content
            return response.get("message", {}).get("content", "I am having trouble connecting right now.")

        except Exception as e:
            logging.error(f"Ollama API Error: {e}")
            # The client should automatically detect connection issues and raise exceptions
            return "I apologize, but I am unable to connect to my assistance service at the moment. Please check the Ollama server."