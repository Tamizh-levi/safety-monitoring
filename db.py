import os
from pymongo import MongoClient
import datetime

# --- Configuration ---
# Make sure this matches the settings in your main application
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB_NAME = "patient_monitoring"
MONGO_COLLECTION_NAME = "schedules"

# --- 1. EDIT SCHEDULE DETAILS HERE ---
# Change these values for each new reminder you want to add.
patient_id = "PAT001"
# Time in 24-hour HH:MM format (with a leading zero for hours less than 10)
reminder_time = "12:01"
# The message that will be spoken after "Hello [Patient Name],"
reminder_message = "exercise time."


def add_schedule_to_db():
    """Connects to MongoDB and inserts a new schedule document."""

    # Connect to the database
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ismaster')  # Check if the connection is successful
        db = client[MONGO_DB_NAME]
        collection = db[MONGO_COLLECTION_NAME]
        print(f"Successfully connected to MongoDB database: '{MONGO_DB_NAME}'")
    except Exception as e:
        print(f"Error: Could not connect to MongoDB. {e}")
        return

    # Create the schedule document
    schedule_document = {
        "patient_id": patient_id,
        "time": reminder_time,
        "message": reminder_message,
        "created_date": datetime.datetime.utcnow()
    }

    # Insert the document into the collection
    try:
        # Check for an identical existing schedule to avoid duplicates
        if collection.find_one({"patient_id": patient_id, "time": reminder_time, "message": reminder_message}):
            print(f"An identical schedule already exists for patient '{patient_id}' at {reminder_time}.")
        else:
            collection.insert_one(schedule_document)
            print(f"Successfully added reminder for patient '{patient_id}' at {reminder_time}.")

    except Exception as e:
        print(f"Error inserting document into MongoDB: {e}")
    finally:
        # Close the connection
        client.close()


if __name__ == "__main__":
    add_schedule_to_db()
