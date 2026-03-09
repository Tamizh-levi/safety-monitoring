import socket
import json

HOST = "127.0.0.1"
PORT = 2000

def start_listener():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind((HOST, PORT))
        server.listen(5)

        print(f"✅ Listening on {HOST}:{PORT}...")

        while True:
            conn, addr = server.accept()
            with conn:
                print(f"📡 Connected by {addr}")

                data = conn.recv(4096)
                if not data:
                    continue

                try:
                    message = json.loads(data.decode("utf-8"))

                    print("\n🚨 ALERT RECEIVED 🚨")
                    print("Type:", message.get("type"))
                    print("Message:", message.get("message"))
                    print("Patient ID:", message.get("patient_id"))
                    print("Patient Name:", message.get("patient_name"))
                    print("Requester ID:", message.get("requester_id"))
                    print("Timestamp:", message.get("timestamp"))
                    print("-" * 40)

                except Exception as e:
                    print("❌ Invalid JSON:", e)

if __name__ == "__main__":
    start_listener()