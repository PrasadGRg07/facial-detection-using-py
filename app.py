import cv2
import face_recognition
import os
import threading
import time
import base64
import numpy as np
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ── Flask App Setup ───────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

KNOWN_FACES_DIR = os.path.join(os.path.dirname(__file__), "known_faces")
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")
ADMIN_TOKEN = "secret-admin-token-123"

# ── Globals (protected by lock) ───────────────────────────────────────────────
lock = threading.Lock()
known_encodings = []
known_names = []
latest_frame = None          # latest annotated JPEG bytes
detection_log = []           # [{name, time}]
MAX_LOG = 50


def load_known_faces():
    """Load all images from known_faces/ and build encodings."""
    global known_encodings, known_names
    encodings = []
    names = []
    for filename in os.listdir(KNOWN_FACES_DIR):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(KNOWN_FACES_DIR, filename)
        try:
            image = face_recognition.load_image_file(path)
            face_encs = face_recognition.face_encodings(image)
            if face_encs:
                encodings.append(face_encs[0])
                names.append(os.path.splitext(filename)[0])
        except Exception as e:
            print(f"[WARN] Could not load {filename}: {e}")
    with lock:
        known_encodings = encodings
        known_names = names
    print(f"[INFO] Loaded {len(names)} known face(s): {names}")


# ── Stateless Detection Route ───────────────────────────────────────────────────
@app.route("/detect", methods=["POST"])
def detect_faces():
    global detection_log
    data = request.json
    if not data or "image" not in data:
        return jsonify({"error": "Missing image data"}), 400

    image_data = data["image"]
    if "," in image_data:
        image_data = image_data.split(",")[1]

    try:
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Invalid image"}), 400
    except Exception as e:
        return jsonify({"error": f"Image decoding failed: {e}"}), 400

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)
    
    locations = face_recognition.face_locations(small)
    encodings = face_recognition.face_encodings(small, locations)

    with lock:
        enc_copy = list(known_encodings)
        names_copy = list(known_names)

    face_names = []
    for face_enc in encodings:
        name = "Unknown"
        if enc_copy:
            matches = face_recognition.compare_faces(enc_copy, face_enc, tolerance=0.5)
            if True in matches:
                name = names_copy[matches.index(True)]
        face_names.append(name)

    timestamp = time.strftime("%H:%M:%S")
    with lock:
        for n in face_names:
            detection_log.append({"name": n, "time": timestamp})
        global MAX_LOG
        detection_log = detection_log[-MAX_LOG:]

    results = []
    for (top, right, bottom, left), name in zip(locations, face_names):
        top *= 2; right *= 2; bottom *= 2; left *= 2
        results.append({
            "name": name,
            "box": [top, right, bottom, left]
        })

    return jsonify({"faces": results})


@app.route("/known_faces", methods=["GET"])
def get_known_faces():
    with lock:
        names = list(known_names)
    return jsonify({"faces": names})


@app.route("/admin/login", methods=["POST"])
def admin_login():
    data = request.json
    if data and data.get("password") == ADMIN_PASSWORD:
        return jsonify({"token": ADMIN_TOKEN})
    return jsonify({"error": "Invalid password"}), 401


@app.route("/register_face", methods=["POST"])
def register_face():
    if "image" not in request.files or "name" not in request.form:
        return jsonify({"error": "Missing image or name"}), 400

    name = request.form["name"].strip()
    if not name:
        return jsonify({"error": "Name cannot be empty"}), 400

    file = request.files["image"]
    filename = secure_filename(f"{name}.jpg")
    save_path = os.path.join(KNOWN_FACES_DIR, filename)
    file.save(save_path)

    # Verify the image has a detectable face
    try:
        image = face_recognition.load_image_file(save_path)
        encs = face_recognition.face_encodings(image)
        if not encs:
            os.remove(save_path)
            return jsonify({"error": "No face detected in the uploaded image. Please try a clearer photo."}), 400
            
        # Deduplication Check
        new_enc = encs[0]
        with lock:
            enc_copy = list(known_encodings)
            names_copy = list(known_names)
            
        if enc_copy:
            matches = face_recognition.compare_faces(enc_copy, new_enc, tolerance=0.5)
            if True in matches:
                existing_name = names_copy[matches.index(True)]
                os.remove(save_path)
                return jsonify({"error": f"Face already registered as '{existing_name}'"}), 400

    except Exception as e:
        os.remove(save_path)
        return jsonify({"error": f"Could not process image: {str(e)}"}), 400

    load_known_faces()
    return jsonify({"success": True, "message": f"'{name}' registered successfully!"})


@app.route("/remove_face/<name>", methods=["DELETE"])
def remove_face(name):
    token = request.headers.get("Authorization")
    if token != f"Bearer {ADMIN_TOKEN}":
        return jsonify({"error": "Unauthorized"}), 401

    removed = False
    for ext in [".jpg", ".jpeg", ".png"]:
        path = os.path.join(KNOWN_FACES_DIR, secure_filename(f"{name}{ext}"))
        if os.path.exists(path):
            os.remove(path)
            removed = True
            break
    if not removed:
        return jsonify({"error": "Face not found"}), 404
    load_known_faces()
    return jsonify({"success": True, "message": f"'{name}' removed."})


@app.route("/detection_log", methods=["GET"])
def get_detection_log():
    with lock:
        log = list(detection_log)
    return jsonify({"log": list(reversed(log))})


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_known_faces()
    # Note: On Render, use a different port if needed via os.environ.get('PORT')
    port = int(os.environ.get("PORT", 5001))
    print(f"[INFO] Flask server starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
