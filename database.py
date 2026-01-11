import os
import datetime
import logging
import bson
from typing import List, Tuple, Dict, Any, Optional
from pymongo import MongoClient

# Assumes config.py is in the same directory
from config import Config


class DatabaseManager:
    """Handles all interactions with the MongoDB database."""

    def __init__(self, config: Config):
        self.config = config
        self.db_client = None
        self.alerts_collection = None
        self.patients_collection = None
        self.schedules_collection = None
        self.connect()

    def connect(self):
        """Establishes connection to the MongoDB server and collections."""
        try:
            self.db_client = MongoClient(self.config.MONGO_URI, serverSelectionTimeoutMS=5000)
            self.db_client.admin.command('ismaster')  # Check connection
            db = self.db_client[self.config.MONGO_DB_NAME]
            self.alerts_collection = db[self.config.ALERTS_COLLECTION_NAME]
            self.patients_collection = db[self.config.PATIENTS_COLLECTION_NAME]
            self.schedules_collection = db[self.config.SCHEDULES_COLLECTION_NAME]
            logging.info("Successfully connected to MongoDB.")
            print("Successfully connected to MongoDB.")
        except Exception as e:
            logging.error(f"Could not connect to MongoDB: {e}")
            print(f"Error: Could not connect to MongoDB. Please ensure it is running.")
            self.db_client = None

    def get_patient_details(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Fetches a single patient's document from the database."""
        if self.patients_collection is None: return None
        try:
            return self.patients_collection.find_one({"_id": patient_id})
        except Exception as e:
            print(f"Error fetching patient details for {patient_id}: {e}")
            logging.error(f"Error fetching patient details for {patient_id}: {e}")
            return None

    def get_all_patients_with_photos(self) -> List[Dict[str, Any]]:
        """Retrieves all patients who have a photo stored for face recognition."""
        if self.patients_collection is None: return []
        try:
            return list(self.patients_collection.find({"photo": {"$exists": True}}))
        except Exception as e:
            print(f"Error fetching patients from DB: {e}")
            logging.error(f"Error fetching patients from DB: {e}")
            return []

    def log_event(self, patient_id: str, patient_name: str, message: str, images: Optional[List[Any]] = None,
                  requester_id: str = None, requester_name: str = None, is_dashboard_alert: bool = False):
        """
        Logs an alert or event into the alerts collection.
        Includes optional requester info and a flag for dashboard visibility.
        """
        if self.alerts_collection is None: return
        try:
            event_doc = {
                "timestamp": datetime.datetime.utcnow(),
                "patient_id": patient_id,  # Monitored patient ID
                "patient_name": patient_name,
                "event_message": message,
                "requester_id": requester_id,  # Source of the request (patient or worker)
                "requester_name": requester_name,
                "is_dashboard_alert": is_dashboard_alert  # NEW: Flag for real-time dashboard display
            }
            if images:
                # Convert images to BSON binary format for storage
                event_doc["image_snapshots"] = [
                    bson.binary.Binary(image) for image in images
                ]
            self.alerts_collection.insert_one(event_doc)
            print(f"Event logged to DB: {message}")
        except Exception as e:
            logging.error(f"Failed to log event to MongoDB: {e}")

    def add_scheduled_reminder(self, patient_id: str, time_str: str, message: str) -> bool:
        """
        Adds a new scheduled reminder to the schedules collection.
        Time format must be HH:MM (e.g., '08:30').
        """
        if self.schedules_collection is None:
            logging.error("Schedules collection not connected. Cannot add reminder.")
            return False

        try:
            # Simple check for HH:MM format (will raise ValueError if not matching)
            datetime.datetime.strptime(time_str, "%H:%M")
        except ValueError:
            logging.error(f"Invalid time format: {time_str}. Must be HH:MM.")
            return False

        try:
            reminder_doc = {
                "patient_id": patient_id,
                "time": time_str,
                "message": message,
                "created_at": datetime.datetime.utcnow()
            }
            self.schedules_collection.insert_one(reminder_doc)
            print(f"Scheduled reminder added for {patient_id} at {time_str}.")
            logging.info(f"Scheduled reminder added for {patient_id} at {time_str}.")
            return True
        except Exception as e:
            logging.error(f"Failed to add scheduled reminder to MongoDB: {e}")
            print(f"Error adding reminder: {e}")
            return False

    def get_schedule_for_now(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Checks if there is a scheduled reminder for the current time."""
        if self.schedules_collection is None: return None
        try:
            current_time_str = datetime.datetime.now().strftime("%H:%M")
            return self.schedules_collection.find_one({
                "patient_id": patient_id,
                "time": current_time_str
            })
        except Exception as e:
            print(f"Error checking schedule in DB: {e}")
            logging.error(f"Error checking schedule in DB: {e}")
            return None

    def close(self):
        """Closes the database connection."""
        if self.db_client:
            self.db_client.close()
            print("MongoDB connection closed.")
