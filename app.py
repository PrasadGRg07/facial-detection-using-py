import cv2
import face_recognition
import os
import threading
import time
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ── Flask App Setup ───────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "http://localhost:3000"])

KNOWN_FACES_DIR = os.path.join(os.path.dirname(__file__), "known_faces")
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

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


# ── Background Capture Thread ─────────────────────────────────────────────────
def capture_loop():
    global latest_frame, detection_log

    video = cv2.VideoCapture(0)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    process_this_frame = True
    locations = []
    face_names = []

    while True:
        success, frame = video.read()
        if not success or frame is None:
            time.sleep(0.05)
            continue

        if process_this_frame:
            small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb_small)

            with lock:
                enc_copy = list(known_encodings)
                names_copy = list(known_names)

            encodings = face_recognition.face_encodings(rgb_small, locations)
            face_names = []
            for face_enc in encodings:
                name = "Unknown"
                if enc_copy:
                    matches = face_recognition.compare_faces(enc_copy, face_enc, tolerance=0.5)
                    if True in matches:
                        name = names_copy[matches.index(True)]
                face_names.append(name)

            # Update detection log
            timestamp = time.strftime("%H:%M:%S")
            with lock:
                for n in face_names:
                    detection_log.append({"name": n, "time": timestamp})
                detection_log = detection_log[-MAX_LOG:]

        process_this_frame = not process_this_frame

        # Draw bounding boxes
        for (top, right, bottom, left), name in zip(locations, face_names):
            top *= 4; right *= 4; bottom *= 4; left *= 4
            color = (0, 255, 128) if name != "Unknown" else (0, 80, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 28), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 8),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)

        # Encode as JPEG
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with lock:
            latest_frame = jpeg.tobytes()

    video.release()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/status")
def status():
    with lock:
        count = len(known_names)
        names = list(known_names)
    return jsonify({"status": "ok", "known_faces": count, "names": names})


def generate_frames():
    while True:
        with lock:
            frame = latest_frame
        if frame is None:
            time.sleep(0.05)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )
        time.sleep(1 / 30)  # ~30 fps cap


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/known_faces", methods=["GET"])
def get_known_faces():
    with lock:
        names = list(known_names)
    return jsonify({"faces": names})


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
    except Exception as e:
        os.remove(save_path)
        return jsonify({"error": f"Could not process image: {str(e)}"}), 400

    load_known_faces()
    return jsonify({"success": True, "message": f"'{name}' registered successfully!"})


@app.route("/remove_face/<name>", methods=["DELETE"])
def remove_face(name):
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
    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()
    print("[INFO] Flask server starting on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
