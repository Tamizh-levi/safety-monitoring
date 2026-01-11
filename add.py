import cv2
import os
from pymongo import MongoClient
import bson

# --- Configuration (Should match your main script) ---
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB_NAME = "patient_monitoring"
PATIENTS_COLLECTION_NAME = "patients"


def add_new_patient():
    """
    A script to add a new SAFETY with their photo to the MongoDB database.
    """
    # --- 1. Connect to the Database ---
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ismaster')  # Check connection
        db = client[MONGO_DB_NAME]
        patients_collection = db[PATIENTS_COLLECTION_NAME]
        print("Successfully connected to MongoDB.")
    except Exception as e:
        print(f"Error: Could not connect to MongoDB. Please ensure it's running. Details: {e}")
        return

    # --- 2. Get Patient Information from User ---
    print("\n--- Enter New Patient Details ---")
    patient_id = input(" PAT002 ")
    patient_name = input("Enter Patient's Full Name: ")
    photo_path = input(
        "C:\hospital_expo_\PyCharmMiscProject\img.png"
        ""
        ""
        "")

    # --- 3. Validate Inimport cv2
    # import os
    # from pymongo import MongoClient
    # import bson
    #
    # # --- Configuration (Should match your main script) ---
    # MONGO_URI = "mongodb://localhost:27017/"
    # MONGO_DB_NAME = "patient_monitoring"
    # PATIENTS_COLLECTION_NAME = "patients"
    #
    # def add_new_patient():
    #     """
    #     A script to add a new SAFETY with their photo to the MongoDB database.
    #     """
    #     # --- 1. Connect to the Database ---
    #     try:
    #         client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    #         client.admin.command('ismaster') # Check connection
    #         db = client[MONGO_DB_NAME]
    #         patients_collection = db[PATIENTS_COLLECTION_NAME]
    #         print("Successfully connected to MongoDB.")
    #     except Exception as e:
    #         print(f"Error: Could not connect to MongoDB. Please ensure it's running. Details: {e}")
    #         return
    #
    #     # --- 2. Get Patient Information from User ---
    #     print("\n--- Enter New Patient Details ---")
    #     patient_id = input("Enter Patient ID (e.g., PAT002): ")
    #     patient_name = input("Enter Patient's Full Name: ")
    #     photo_path = input("Enter the full path to the SAFETY's photo (e.g., C:\\Users\\YourName\\Pictures\\patient2.jpg): ")
    #
    #     # --- 3. Validate Inputs ---
    #     if not all([patient_id, patient_name, photo_path]):
    #         print("\nError: All fields are required. Please try again.")
    #         client.close()
    #         return
    #
    #     if not os.path.exists(photo_path):
    #         print(f"\nError: The photo path '{photo_path}' does not exist. Please check the path and try again.")
    #         client.close()
    #         return
    #
    #     # --- 4. Process the Photo ---
    #     try:
    #         image = cv2.imread(photo_path)
    #         if image is None:
    #             print("\nError: Could not read the image file. It might be corrupted or in an unsupported format.")
    #             client.close()
    #             return
    #
    #         # Convert the image to binary format for storage
    #         _, img_encoded = cv2.imencode('.jpg', image)
    #         photo_binary = bson.binary.Binary(img_encoded.tobytes())
    #         print("Successfully processed the SAFETY's photo.")
    #
    #     except Exception as e:
    #         print(f"\nAn error occurred while processing the image: {e}")
    #         client.close()
    #         return
    #
    #     # --- 5. Create and Insert the Patient Document ---
    #     patient_document = {
    #         "_id": patient_id,
    #         "name": patient_name,
    #         "photo": photo_binary
    #     }
    #
    #     try:
    #         # Check if a SAFETY with this ID already exists
    #         if patients_collection.find_one({"_id": patient_id}):
    #             print(f"\nError: A SAFETY with ID '{patient_id}' already exists in the database.")
    #         else:
    #             patients_collection.insert_one(patient_document)
    #             print(f"\nSuccess! Patient '{patient_name}' with ID '{patient_id}' has been added to the database.")
    #     except Exception as e:
    #         print(f"\nAn error occurred while inserting the document into MongoDB: {e}")
    #
    #     # --- 6. Close the Connection ---
    #     finally:
    #         client.close()
    #         print("MongoDB connection closed.")
    #
    #
    # if __name__ == "__main__":
    #     add_new_patient()puts ---
    if not all([patient_id, patient_name, photo_path]):
        print("\nError: All fields are required. Please try again.")
        client.close()
        return

    if not os.path.exists(photo_path):
        print(f"\nError: The photo path '{photo_path}' does not exist. Please check the path and try again.")
        client.close()
        return

    # --- 4. Process the Photo ---
    try:
        image = cv2.imread(photo_path)
        if image is None:
            print("\nError: Could not read the image file. It might be corrupted or in an unsupported format.")
            client.close()
            return

        # Convert the image to binary format for storage
        _, img_encoded = cv2.imencode('.jpg', image)
        photo_binary = bson.binary.Binary(img_encoded.tobytes())
        print("Successfully processed the SAFETY's photo.")

    except Exception as e:
        print(f"\nAn error occurred while processing the image: {e}")
        client.close()
        return

    # --- 5. Create and Insert the Patient Document ---
    patient_document = {
        "_id": patient_id,
        "name": patient_name,
        "photo": photo_binary
    }

    try:
        # Check if a SAFETY with this ID already exists
        if patients_collection.find_one({"_id": patient_id}):
            print(f"\nError: A SAFETY with ID '{patient_id}' already exists in the database.")
        else:
            patients_collection.insert_one(patient_document)
            print(f"\nSuccess! Patient '{patient_name}' with ID '{patient_id}' has been added to the database.")
    except Exception as e:
        print(f"\nAn error occurred while inserting the document into MongoDB: {e}")

    # --- 6. Close the Connection ---
    finally:
        client.close()
        print("MongoDB connection closed.")


if __name__ == "__main__":
    add_new_patient()
