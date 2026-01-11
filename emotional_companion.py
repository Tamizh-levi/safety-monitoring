import os
import time
import threading
import logging
import speech_recognition as sr
import ollama  # REPLACED requests and json with ollama for client interaction
import json
from typing import Dict, Any, Optional

# Assuming necessary speech libraries are installed (gTTS, playsound)
from gtts import gTTS
from playsound import playsound
# Local modules
from config import Config
from alerts import AlertManager


class EmotionalCompanion:
    """
    Manages the conversational flow, Ollama interaction, and speech I/O for
    the Emotional Support Companion. This class makes direct API calls to a
    local Ollama server (e.g., running Llama3) using the official client library.
    """

    def __init__(self, config: Config, app_state: Dict[str, Any], alert_manager: AlertManager):
        self.config = config
        self.app_state = app_state
        self.alert_manager = alert_manager

        # Ollama configuration
        # Assuming Llama3 is pulled and running via `ollama run llama3`
        # We retain these config attributes for reference but rely on ollama module defaults
        self.ollama_url = getattr(config, 'OLLAMA_API_URL', 'http://localhost:11434/api/generate')
        self.ollama_model = getattr(config, 'OLLAMA_MODEL_NAME', 'llama3')

        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()

        # State for conversation
        self.is_conversing = False
        self.last_proactive_prompt_time = 0.0
        self.patient_name = alert_manager.patient_name

    def start_listening_thread(self):
        """Starts the main listener loop in a background thread."""
        threading.Thread(target=self._run_listener_loop, daemon=True).start()
        print("Emotional Companion listening thread started.")

    def _get_ollama_response(self, user_text: str, context: str) -> str:
        """
        Sends the user text and context to the local Ollama LLM using ollama.chat().
        """

        # --- System Instruction & Prompt for Empathy ---
        system_instruction = f"""
        You are 'MediMind', a warm, empathetic, supportive AI companion for a patient named {self.patient_name}. 
        Your goal is to provide comfort, reassurance, and distraction. Always respond in a brief (1-3 sentences), friendly, 
        and non-judgmental tone. Never give medical advice.

        Current context: {context}.
        """

        # Critical Distress Check (Rule-Engine Override for immediate nurse call)
        user_text_lower = user_text.lower()
        if any(keyword in user_text_lower for keyword in ["hurt", "pain", "scared", "anxiety", "dying", "emergency"]):
            self.alert_manager.trigger_alarm(f"Companion Protocol: Patient expressed severe distress ('{user_text}').",
                                             "call_nurse")
            return "I hear the immediate concern in your voice. I'm calling the nurse right now, and I will stay on the line to talk with you until they arrive. Just focus on breathing deeply, slowly."

        # Prepare messages list for ollama.chat
        messages = [
            {
                'role': 'system',
                'content': system_instruction
            },
            {
                'role': 'user',
                'content': user_text
            }
        ]

        try:
            # Using ollama.chat as requested, which simplifies the interaction
            response = ollama.chat(
                model=self.ollama_model,
                messages=messages,
                options={
                    "temperature": 0.7,
                    "num_ctx": 4096,
                }
            )

            # The response is a dictionary, and the content is nested
            full_response_text = response['message']['content']

            logging.info(f"Ollama Response: {full_response_text.strip()}")
            return full_response_text.strip()

        except Exception as e:
            # Catching generic Exception covers connection errors and potential API errors from the Ollama client
            if "ConnectionError" in str(e):  # Attempt to identify connection error specifically
                error_msg = f"Error: Could not connect to Ollama at {self.ollama_url}. Is the server running? Falling back to rule-based response."
                logging.error(error_msg)
                return "I'm having trouble connecting to my central systems right now, but I'm here to listen. Tell me more about what you're thinking."

            logging.error(f"Error during Ollama API call: {e}")
            return "I apologize, my systems encountered an error. Please try asking again, or if this is urgent, please call the nurse."

    def speak(self, text: str):
        """Uses Google Text-to-Speech to read the companion's response aloud."""
        try:
            print(f"Companion Speaking: {text}")
            tts = gTTS(text=text, lang='en')
            filename = os.path.join(self.config.REMINDER_AUDIO_DIR, f"companion_{int(time.time())}.mp3")
            tts.save(filename)
            playsound(filename)
            os.remove(filename)
        except Exception as e:
            logging.error(f"Error in companion text-to-speech: {e}")

    def _handle_conversation(self, initial_prompt: Optional[str] = None):
        """Manages the full conversation flow (speaking -> listening -> LLM response)."""
        if self.is_conversing:
            return

        self.is_conversing = True
        self.speak(
            initial_prompt or f"Hello {self.patient_name}, I noticed you might be feeling a bit down. Would you like to talk?")

        start_time = time.time()
        # Conversation loop runs until timeout or the SAFETY explicitly ends it
        while self.is_conversing and (time.time() - start_time < self.config.COMPANION_TIMEOUT_SEC):
            try:
                with self.mic as source:
                    # Listen for a short period during conversation
                    audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=8)

                user_command = self.recognizer.recognize_google(audio, language='en-US').lower()
                print(f"Patient said: {user_command}")
                logging.info(f"Companion received: {user_command}")

                # Contextual response from LLM
                pain_score = self.app_state.get('last_pain_score', 0.0)
                context_info = f"Current time is {time.strftime('%H:%M')}. Normalized pain score is {pain_score:.2f}. "

                # Check for explicit end phrases before sending to LLM (optional rule-based end)
                if any(keyword in user_command for keyword in ["bye", "stop talking", "I want to rest now"]):
                    self.speak("You're very welcome! I'm glad we talked. I'll be here if you need me.")
                    self.is_conversing = False
                    break

                llm_response = self._get_ollama_response(user_command, context_info)

                self.speak(llm_response)

            except sr.WaitTimeoutError:
                if self.is_conversing:
                    self.speak(
                        "I'm still here. Is there anything else you wanted to share, or would you prefer I wait quietly now?")
                else:
                    break
            except sr.UnknownValueError:
                self.speak("I'm sorry, I didn't quite catch that. Could you please say that again?")
            except Exception as e:
                logging.error(f"Error during companion conversation: {e}")
                self.is_conversing = False
                break

        # Explicitly end conversation if loop times out
        if self.is_conversing:
            self.speak("It looks like it's time for me to let you rest. Remember I'm just a call away if you need me.")
            self.is_conversing = False

        print("Emotional Companion conversation ended.")

    def proactively_engage(self, trigger_message: str):
        """Triggers a new conversation if enough time has passed since the last one."""
        if not self.app_state.get("companion_active", False):
            return

        current_time = time.time()

        if self.app_state.get("voice_active", False) or self.is_conversing:
            return

        # Check for cooldown
        if (current_time - self.last_proactive_prompt_time > self.config.COMPANION_COOLDOWN_SEC):
            self.last_proactive_prompt_time = current_time
            # Start the conversation in a new thread so it doesn't block the video loop
            threading.Thread(target=self._handle_conversation,
                             args=(
                                 f"Hello {self.patient_name}, I noticed a slight change in your expression—you seemed thoughtful or mildly distressed ({trigger_message}). Is everything alright, or is there anything I can help with?"),
                             daemon=True).start()

    def _run_listener_loop(self):
        """Main background loop to listen for 'Companion' trigger word."""
        with self.mic as source:
            pass

        while self.app_state["running"]:
            if not self.is_conversing:
                try:
                    with self.mic as source:
                        # Listen for only the trigger phrase or a short command
                        audio = self.recognizer.listen(source, timeout=4, phrase_time_limit=3)

                    phrase = self.recognizer.recognize_google(audio, language='en-US').lower()

                    if any(trigger in phrase for trigger in self.config.COMPANION_TRIGGER_WORDS):
                        self.proactively_engage("I heard you call for me. How can I help you right now?")

                except sr.WaitTimeoutError:
                    continue
                except (sr.UnknownValueError, sr.RequestError):
                    time.sleep(1)
                except Exception as e:
                    logging.error(f"Error in companion passive listener: {e}")
                    time.sleep(2)
            else:
                time.sleep(1)
