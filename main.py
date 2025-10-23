from flask import Flask, render_template, Response, jsonify
import threading
import time
import head_pose
import audio
import detection
import config
import cv2

app = Flask(__name__, template_folder='templates', static_folder='static')
config.RUNNING.set()

cap = None
processed_frame = None
display_thread = None

def run_proctoring_core():
    global cap, processed_frame
    cap = cv2.VideoCapture(0)
    # Corrected the function name from is_Opened() to isOpened()
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        config.RUNNING.clear()
        return

    while config.RUNNING.is_set():
        success, frame = cap.read()
        if not success:
            break
        
        processed_frame = head_pose.process_frame(frame)
        time.sleep(0.01)

    cap.release()
    cv2.destroyAllWindows()
    print("Proctoring core stopped.")

def run_detection_thread():
    detection.run_detection()

def run_audio_thread():
    audio.sound()
    
def display_frames():
    global processed_frame
    while config.RUNNING.is_set():
        if processed_frame is not None:
            cv2.imshow('Head Pose Estimation', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                config.RUNNING.clear()
        time.sleep(0.01)
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/cheat_status')
def cheat_status():
    return jsonify({
        'cheating': detection.GLOBAL_CHEAT,
        'percentage': detection.PERCENTAGE_CHEAT
    })

def generate_frames():
    global processed_frame
    while config.RUNNING.is_set():
        if processed_frame is not None:
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.01)

if __name__ == '__main__':
    proctoring_core_thread = threading.Thread(target=run_proctoring_core)
    detection_thread = threading.Thread(target=run_detection_thread)
    audio_thread = threading.Thread(target=run_audio_thread)
    display_thread = threading.Thread(target=display_frames)

    proctoring_core_thread.daemon = True
    detection_thread.daemon = True
    audio_thread.daemon = True
    display_thread.daemon = True
    
    proctoring_core_thread.start()
    detection_thread.start()
    audio_thread.start()
    display_thread.start()

    app.run(debug=True, use_reloader=False)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        config.RUNNING.clear()
        proctoring_core_thread.join()
        detection_thread.join()
        audio_thread.join()
        display_thread.join()
