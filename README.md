An online examination platform with real-time proctoring capabilities, designed to ensure academic integrity. 
The system features a user-friendly interface, multiple-choice questions, a countdown timer, and automated proctoring using audio and head pose detection to monitor suspicious behavior.

**_Features_**
**Student Login:** Simple login interface (currently without authentication backend).

**Exam Interface:**
5 multiple-choice questions with navigation (Previous/Next).
30-minute countdown timer.

**Proctoring System:**
Audio monitoring to detect suspicious sounds.
Head pose estimation to track student movement.
Real-time cheating probability calculation.

**Results Display:** Shows total questions, correct answers, score percentage, and average cheating probability.

**Backend Monitoring:** Logs average cheating percentage to the console upon exam submission.

**Responsive Design**: Modern, colorful UI with gradients and hover effects, optimized for mobile devices.

**Technologies Used**
**Frontend:** HTML, CSS, JavaScript
**Backend:** Flask (Python)

**Proctoring:**
OpenCV and MediaPipe for head pose detection
SoundDevice for audio analysis
Matplotlib for real-time cheating probability visualization (backend)
**Dependencies:** Listed in requirements.txt

**Usage**
**Run the Application:**
python main.py
The Flask server will start on http://localhost:5000.

**Access the Exam:**
Open your browser and navigate to http://localhost:5000.
Enter any username and password (no authentication enforced yet).
Click "Login" to start the exam.

**Take the Exam:**
Answer the 5 multiple-choice questions using the radio buttons.
Navigate with "Previous" and "Next" buttons.
Submit the exam with "Submit Exam" before or when the timer (30 minutes) runs out.

**View Results:**
See your score and average cheating probability in the results section.
Check the server console for the backend log of the average cheating percentage.

**Proctoring:**
The system monitors audio and head movements in the background.
Cheating probability is calculated but not displayed during the exam (logged in backend).

**Project Structure**
online_exam_proctoring/
├── static/
│   ├── stylesheet.css      # Styling for the frontend
│   └── script.js           # Frontend logic (exam flow, results)
├── templates/
│   └── index.html          # HTML template for the UI
├── config.py               # Shared variables (e.g., RUNNING flag)
├── audio.py                # Audio-based proctoring logic
├── head_pose.py            # Head pose detection using OpenCV/MediaPipe
├── detection.py            # Cheating probability calculation and visualization
├── main.py                 # Flask application and server logic
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

**How It Works**
**Frontend:**
index.html defines the UI with login, exam, and results sections.
stylesheet.css provides a colorful, modern design with gradients and shadows.
script.js handles exam navigation, timer, and results display.

**Backend:**
main.py runs the Flask server, manages proctoring threads, and logs results.
audio.py monitors sound amplitude for suspicious noise.
head_pose.py tracks head movements to detect looking away.
detection.py calculates cheating probability and stores percentages for averaging.
config.py holds the RUNNING flag to control proctoring threads.

**Proctoring:**
Starts when the exam begins and stops on submission.
Average cheating probability is logged in the backend and shown in results.
