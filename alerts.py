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
import socket  # REQUIRED for sending notifications to Port 2000
from gtts import gTTS
from typing import List, Tuple, Dict, Any, Optional

# NEW: Import the dedicated Ollama client
try:
    import ollama
except ImportError:
    # Fallback/warning if ollama library is not installed
    print("WARNING: 'ollama' Python library not found. Ollama companion may fail.")
    ollama = None

# Local modules
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
        self.core_processor_ref = None  # Reference to PatientMonitorCore (for flag clearing)

        # Worker Info fields for Voice/Gesture-based help requests
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
        self.alert_types = [
            "unknown", "unidentified", "crowd", "bed_exit", "stroke", "knife",
            "gun", "cough", "fall_detection", "drowsiness", "pain",
            "happiness", "sadness", "safety_violation", "fire",
            "help_request", "thank_you_response"
        ]

        self.alert_timers = {k: None for k in self.alert_types}
        self.alert_sent_flags = {k: False for k in self.alert_types}

        # Ollama related attributes
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
        print(f"Worker info updated: {worker_name} ({worker_id})")

    def speak_reminder(self, text: str):
        """Uses Google Text-to-Speech to read a reminder aloud."""
        try:
            print(f"Speaking reminder: {text}")
            tts = gTTS(text=text, lang='en')
            filename = os.path.join(self.config.REMINDER_AUDIO_DIR, f"reminder_{int(time.time())}.mp3")
            tts.save(filename)
            playsound(filename)
            if os.path.exists(filename):
                os.remove(filename)
        except Exception as e:
            logging.error(f"Error in text-to-speech: {e}")

    def _send_dashboard_alert_request(self, message: str, requester_id: str, requester_name: str, alert_type: str):
        """Private method to send an HTTP POST request to the external dashboard API."""
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
            response = requests.post(self.config.DASHBOARD_URL, json=payload, timeout=5)
            if response.status_code == 200 or response.status_code == 201:
                logging.info(f"Dashboard alert sent successfully to {self.config.DASHBOARD_URL}.")
                print(f"✅ Alert sent to URL successfully! Status: {response.status_code}")
            else:
                logging.error(f"Failed to send dashboard alert. Status: {response.status_code}")
                print(f"❌ Failed to send alert to URL. Status: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Error connecting to dashboard API: {e}")
            print(f"❌ Error connecting to dashboard URL: {e}")

    def send_whatsapp_alert(self, message: str, requester_id: str, requester_name: str):
        """Sends an alert message via WhatsApp using pywhatkit."""
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            full_message = (
                f"*** Patient Room Monitoring Alert ***\n\n"
                f"Monitored Patient: {self.patient_name} (ID: {self.patient_id})\n"
                f"Requester/Source: {requester_name} (ID: {requester_id})\n\n"
                f"Timestamp: {timestamp}\n"
                f"Alert Details: {message}\n\n"
                f"This is an automated alert."
            )
            print(f"Attempting to send WhatsApp message: {message}")
            pywhatkit.sendwhatmsg_instantly(
                self.config.RECEIVER_PHONE_WHATSAPP,
                full_message,
                wait_time=15,
                tab_close=True
            )
        except Exception as e:
            logging.error(f"Failed to send WhatsApp message: {e}")

    def send_port_notification(self, message: str, alert_type: str, requester_id: str):
        """Sends a JSON-formatted notification to the configured TCP port."""
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
                s.settimeout(2)
                s.connect((host, port))
                s.sendall(json.dumps(data).encode('utf-8'))
                print(f"✅ Notification sent to Port {port}: {alert_type}")
        except Exception as e:
            logging.error(f"Failed to send notification to port {port}: {e}")

    def trigger_alarm(self, message: str, sound_key: Optional[str] = None, images: Optional[List[np.ndarray]] = None,
                      requester_id: Optional[str] = None, requester_name: Optional[str] = None,
                      is_dashboard_alert: bool = True,
                      alert_type: str = "general_alert"):
        """The main alarm function: logs, notifies, saves images, and plays sound."""

        final_requester_id = requester_id if requester_id is not None else self.patient_id
        final_requester_name = requester_name if requester_name is not None else self.patient_name

        print(f"ALARM TRIGGERED: {message} (Source: {final_requester_name})")

        # Log event to Database
        self.db_manager.log_event(self.patient_id, self.patient_name, message, images,
                                  requester_id=final_requester_id, requester_name=final_requester_name,
                                  is_dashboard_alert=is_dashboard_alert)

        if sound_key in self.alert_sent_flags:
            self.alert_sent_flags[sound_key] = True

        if images:
            self._save_images(message, images)

        # Threaded notifications
        threading.Thread(target=self.send_whatsapp_alert,
                         args=(message, final_requester_id, final_requester_name),
                         daemon=True).start()

        sms_message = f"Alert: {message} (Monitored Patient: {self.patient_name} / Source: {final_requester_name})"
        threading.Thread(target=self.sms_sender.send_sms,
                         args=(self.config.RECEIVER_PHONE_SMS, sms_message),
                         daemon=True).start()

        if is_dashboard_alert:
            threading.Thread(target=self._send_dashboard_alert_request,
                             args=(message, final_requester_id, final_requester_name, alert_type),
                             daemon=True).start()

        threading.Thread(target=self.send_port_notification,
                         args=(message, alert_type, final_requester_id),
                         daemon=True).start()

        if sound_key and sound_key in self.config.ALARM_SOUNDS:
            sound_file = self.config.ALARM_SOUNDS[sound_key]
            if os.path.exists(sound_file):
                threading.Thread(target=playsound, args=(sound_file,), daemon=True).start()

    def _save_images(self, message: str, images: List[np.ndarray]):
        """Saves snapshots associated with an alert."""
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
                logging.error(f"Error saving image snapshot: {e}")

    def check_and_trigger_timed_alert(self, frame: np.ndarray, alert_type: str, condition: bool, conf_sec: int,
                                      message: str, sound_key: str, images: Optional[List[np.ndarray]], y_pos: int):
        """Generic handler for timed confirmation alerts."""
        if condition:
            if self.alert_timers.get(alert_type) is None:
                self.alert_timers[alert_type] = time.time()

            elapsed = time.time() - self.alert_timers[alert_type]
            if not self.alert_sent_flags.get(alert_type) and elapsed > conf_sec:
                self.trigger_alarm(message, sound_key, images,
                                   requester_id=self.patient_id, requester_name=self.patient_name,
                                   is_dashboard_alert=True, alert_type=alert_type)
                self.alert_sent_flags[alert_type] = True
        else:
            self.alert_timers[alert_type] = None
            self.alert_sent_flags[alert_type] = False

    def _ollama_generate_json(self, prompt: str, model_name: str, system_prompt: str, schema: dict) -> str:
        """Generates a structured JSON response from Ollama based on a schema."""
        if self.ollama_client is None:
            return json.dumps({"time_24hr": None, "message": "Ollama service unavailable."})

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        try:
            response = self.ollama_client.chat(model=model_name, messages=messages, format="json",
                                               options={'num_predict': 128})
            return response.get("message", {}).get("content", '{}')
        except Exception as e:
            logging.error(f"Ollama JSON Error: {e}")
            return json.dumps({"error": str(e)})

    def check_confirmation_status(self, alert_type: str, condition: bool, conf_sec: int) -> bool:
        """Helper to check if a condition has been confirmed for the alert time."""
        current_time = time.time()
        if condition:
            if self.alert_timers.get(alert_type) is None:
                self.alert_timers[alert_type] = current_time
            if current_time - self.alert_timers[alert_type] >= conf_sec:
                return True
        else:
            self.alert_timers[alert_type] = None
        return False

    def speak_ollama_response(self, text: str):
        """Uses gTTS to convert the LLM response to speech NON-BLOCKINGLY."""
        if self.stop_chat_flag.is_set(): return
        try:
            print(f"[OLLAMA RESPONSE]: {text}")
            tts = gTTS(text=text, lang='en')
            filename = os.path.join(self.config.REMINDER_AUDIO_DIR, f"ollama_response_{int(time.time())}.mp3")
            tts.save(filename)

            def play_and_cleanup(file_path):
                try:
                    playsound(file_path)
                finally:
                    if os.path.exists(file_path): os.remove(file_path)

            threading.Thread(target=play_and_cleanup, args=(filename,), daemon=True).start()
        except Exception as e:
            logging.error(f"Error in Ollama TTS: {e}")

    def cancel_companion_chat(self, reason: str):
        """Sets a flag to stop the ongoing chat thread."""
        with self.chat_lock:
            if self.is_chatting:
                self.stop_chat_flag.set()
                self.is_chatting = False
                if self.core_processor_ref: self.core_processor_ref.is_companion_active = False
                if self.app_state: self.app_state["is_companion_chat_active"] = False
                print(f"[COMPANION] Chat cancelled: {reason}")

    def continue_companion_chat(self, patient_reply: str, model_name: str):
        """Handles the patient's reply to an ongoing conversation."""
        with self.chat_lock:
            if not self.is_chatting or self.stop_chat_flag.is_set(): return
            self.chat_history.append({"role": "user", "content": patient_reply})

        try:
            response_text = self._ollama_generate(model_name, self.chat_history)
            self.speak_ollama_response(response_text)
            self.chat_history.append({"role": "assistant", "content": response_text})
            if len(self.chat_history) >= 6:
                self.cancel_companion_chat("Max turns reached")
            elif self.app_state:
                self.app_state['awaiting_reply'] = True
        except Exception as e:
            self.cancel_companion_chat(f"Error: {e}")

    def start_companion_chat(self, initial_prompt: str, model_name: str, timeout: int):
        """Initiates a non-blocking conversation with the Ollama model."""
        with self.chat_lock:
            if self.is_chatting: return
            self.is_chatting = True
            self.stop_chat_flag.clear()
            self.chat_history = [
                {"role": "system", "content": "You are MediMind, a compassionate companion. Keep responses concise."},
                {"role": "user", "content": initial_prompt}
            ]

        try:
            response_text = self._ollama_generate(model_name, self.chat_history)
            self.speak_ollama_response(response_text)
            self.chat_history.append({"role": "assistant", "content": response_text})
            if self.app_state: self.app_state['awaiting_reply'] = True
        except Exception:
            self.cancel_companion_chat("Initial turn error")

    def _ollama_generate(self, model_name: str, messages: list) -> str:
        """Makes the API call to Ollama using the dedicated Python client."""
        if self.ollama_client is None: return "Service unavailable."
        try:
            response = self.ollama_client.chat(model=model_name, messages=messages, options={'num_predict': 128})
            return response.get("message", {}).get("content", "I am having trouble connecting.")
        except Exception as e:
            logging.error(f"Ollama API Error: {e}")
            return "Connection error."