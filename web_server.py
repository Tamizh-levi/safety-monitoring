from flask import Flask, render_template_string, jsonify, request
import threading
import logging

# Disable Flask default logging for a cleaner console
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


class WebControlServer:
    """
    A Flask-based web server to control the Patient Monitoring system remotely.
    It maps HTTP requests to the shared app_state dictionary.
    """

    def __init__(self, app_state, alert_manager):
        self.app = Flask(__name__)
        self.app_state = app_state
        self.alert_manager = alert_manager
        self.setup_routes()

    def setup_routes(self):
        @self.app.route('/')
        def index():
            """Simple dashboard to see and toggle states."""
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Patient Monitor Control</title>
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    body { font-family: sans-serif; padding: 20px; background: #f0f2f5; }
                    .container { max-width: 600px; margin: auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                    .toggle-btn { display: block; width: 100%; padding: 15px; margin: 10px 0; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; font-weight: bold; transition: 0.3s; }
                    .on { background-color: #4CAF50; color: white; }
                    .off { background-color: #ff9800; color: white; }
                    h1 { text-align: center; color: #333; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>System Control</h1>
                    <div id="controls"></div>
                </div>
                <script>
                    const features = [
                        "unidentified_person", "unknown_person", "bed_exit", 
                        "fall_detection", "drowsiness_detection", "pain_detection", 
                        "stroke_detection", "emotion_detection", "safety_detection",
                        "fire_detection", "crowd_alert", "gestures", "voice", "cough_detection"
                    ];

                    async function toggle(feature) {
                        const res = await fetch(`/toggle/${feature}`);
                        const data = await res.json();
                        updateUI();
                    }

                    async function updateUI() {
                        const res = await fetch('/status');
                        const status = await res.json();
                        const container = document.getElementById('controls');
                        container.innerHTML = '';

                        features.forEach(f => {
                            const isActive = status[f + "_active"];
                            const btn = document.createElement('button');
                            btn.className = `toggle-btn ${isActive ? 'on' : 'off'}`;
                            btn.innerText = `${f.replace('_', ' ').toUpperCase()}: ${isActive ? 'ON' : 'OFF'}`;
                            btn.onclick = () => toggle(f);
                            container.appendChild(btn);
                        });
                    }

                    updateUI();
                    setInterval(updateUI, 2000); // Sync with physical clicks every 2 seconds
                </script>
            </body>
            </html>
            """
            return render_template_string(html)

        @self.app.route('/status')
        def status():
            """Returns the current state of all active features."""
            return jsonify({k: v for k, v in self.app_state.items() if "_active" in k})

        @self.app.route('/toggle/<feature>')
        def toggle_feature(feature):
            """Toggles a feature state and handles cleanup (mirroring ui.py logic)."""
            state_key = f"{feature}_active"
            if state_key in self.app_state:
                self.app_state[state_key] = not self.app_state[state_key]

                # Cleanup logic if turned OFF
                if not self.app_state[state_key]:
                    self._cleanup_alerts(feature)

                print(f"[Web Server] Toggled {feature}: {self.app_state[state_key]}")
                return jsonify({"status": "success", "new_state": self.app_state[state_key]})
            return jsonify({"status": "error", "message": "Feature not found"}), 404

    def _cleanup_alerts(self, feature):
        """Internal helper to clear alert timers when a feature is disabled via Web."""
        mapping = {
            "unknown_person": "unknown", "unidentified_person": "unidentified",
            "bed_exit": "bed_exit", "stroke_detection": "stroke",
            "fall_detection": "fall_detection", "cough_detection": "cough",
            "crowd_alert": "crowd", "drowsiness_detection": "drowsiness",
            "pain_detection": "pain", "safety_detection": "safety_violation",
            "fire_detection": "fire", "emotion_detection": "happiness"
        }
        alert_key = mapping.get(feature)
        if alert_key:
            self.alert_manager.alert_timers.pop(alert_key, None)
            self.alert_manager.alert_sent_flags.pop(alert_key, None)
            if feature == "emotion_detection":
                self.alert_manager.alert_timers.pop("sadness", None)
                self.alert_manager.alert_sent_flags.pop("sadness", None)

    def run(self, host='0.0.0.0', port=1000):
        """Starts the Flask server."""
        self.app.run(host=host, port=port, threaded=True, use_reloader=False)


def start_web_server(app_state, alert_manager):
    server = WebControlServer(app_state, alert_manager)
    server.run()