import time
import threading
import datetime
import logging


class BackgroundWorkers:
    def __init__(self, config, app_state, db_manager, alert_manager, voice_handler):
        self.config = config
        self.app_state = app_state
        self.db_manager = db_manager
        self.alert_manager = alert_manager
        self.voice_handler = voice_handler

        # State for reminders
        self.played_reminders_today = set()
        self.last_reminder_check_date = datetime.date.today()

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
                    # Delegate the speech task to the VoiceHandler
                    threading.Thread(target=self.voice_handler.speak_reminder, args=(full_message,),
                                     daemon=True).start()
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
                is_in_window = current_time >= start_time or current_time <= end_time

            if is_in_window != self.app_state["is_patient_in_safe_sleep_window"]:
                self.app_state["is_patient_in_safe_sleep_window"] = is_in_window
                print(f"Safe Sleep Window changed to: {'ON' if is_in_window else 'OFF'}")

            time.sleep(10)  # Check every 10 seconds