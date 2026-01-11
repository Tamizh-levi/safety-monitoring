import os
from twilio.rest import Client
import logging

class TwilioSMSSender:
    """
    A class to handle sending SMS messages using the Twilio API.
    """
    def __init__(self, twilio_account_sid: str, twilio_auth_token: str, twilio_phone_number: str):
        """
        Initializes the Twilio client with account credentials.

        Args:
            twilio_account_sid: Your Twilio Account SID.
            twilio_auth_token: Your Twilio Auth Token.
            twilio_phone_number: Your Twilio phone number.
        """
        if not all([twilio_account_sid, twilio_auth_token, twilio_phone_number]):
            logging.error("Twilio credentials are not fully configured. SMS sending is disabled.")
            self.client = None
            return

        try:
            self.client = Client(twilio_account_sid, twilio_auth_token)
            self.twilio_phone_number = twilio_phone_number
            logging.info("Twilio client initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize Twilio client: {e}")
            self.client = None

    def send_sms(self, to_phone_number: str, message: str):
        """
        Sends an SMS message to a specified phone number.

        Args:
            to_phone_number: The recipient's phone number.
            message: The message body to send.
        """
        if self.client is None:
            logging.warning("Twilio client is not initialized. Cannot send SMS.")
            return

        try:
            self.client.messages.create(
                to=to_phone_number,
                from_=self.twilio_phone_number,
                body=message
            )
            logging.info(f"SMS sent successfully to {to_phone_number}.")
            print(f"SMS sent successfully to {to_phone_number}.")
        except Exception as e:
            logging.error(f"Failed to send SMS to {to_phone_number}: {e}")
            print(f"Failed to send SMS to {to_phone_number}: {e}")
