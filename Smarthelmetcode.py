import cv2
import mediapipe as mp
import numpy as np
import RPi.GPIO as GPIO
import time
import threading
import firebase_admin
from firebase_admin import credentials, db
import requests
import smbus2 as smbus
import math
from twilio.rest import Client


# Firebase Configuration
cred = credentials.Certificate(r"/home/pranav/techathoncircuitpaglu-firebase-adminsdk-fbsvc-9f26469ff2.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://techathoncircuitpaglu-default-rtdb.asia-southeast1.firebasedatabase.app'
})
# Twilio Configuration
TWILIO_ACCOUNT_SID = 'ACc0fda36782c5837cae479ea9700d65e2'
TWILIO_AUTH_TOKEN = 'eebdfcfc032130d673190bad1fc335d8'
TWILIO_PHONE_NUMBER = '+19045290595'
RECIPIENT_PHONE_NUMBER = '+919324589670'  # Replace with the recipient's phone number
GOOGLE_MAPS_API_KEY = 'AIzaSyBlNLSwItNDfnUBb4D0KotOrcmy0KiyIeE'

# GPIO Pin Configuration
BUZZER_PIN = 17
LED_PIN = 18
ULTRASONIC_TRIG = 23
ULTRASONIC_ECHO = 24
ALCOHOL_SENSOR_PIN = 27

# Thresholds
CLOSED_DURATION_THRESHOLD = 2.0
HELMET_DISTANCE_THRESHOLD = 10.0
FALL_ANGLE_THRESHOLD = 45.0
class MPU6050:
    PWR_MGMT_1 = 0x6B
    SMPLRT_DIV = 0x19
    CONFIG = 0x1A
    GYRO_CONFIG = 0x1B
    ACCEL_CONFIG = 0x1C
    ACCEL_XOUT_H = 0x3B
    
    def __init__(self, bus_number=1, device_address=0x68):
        self.bus = smbus.SMBus(bus_number)
        self.address = device_address
        
        # Wake up the MPU6050
        self.bus.write_byte_data(self.address, self.PWR_MGMT_1, 0)
        self.bus.write_byte_data(self.address, self.SMPLRT_DIV, 7)
        self.bus.write_byte_data(self.address, self.CONFIG, 0)
        self.bus.write_byte_data(self.address, self.ACCEL_CONFIG, 0)
    
    def read_raw_data(self, addr):
        high = self.bus.read_byte_data(self.address, addr)
        low = self.bus.read_byte_data(self.address, addr + 1)
        value = ((high << 8) | low)
        return value - 65536 if value > 32768 else value
    
    
    def calculate_tilt_angle(self):
        accel_x = self.read_raw_data(self.ACCEL_XOUT_H) / 16384.0
        accel_y = self.read_raw_data(self.ACCEL_XOUT_H + 2) / 16384.0
        accel_z = self.read_raw_data(self.ACCEL_XOUT_H + 4) / 16384.0
        
        pitch = math.degrees(math.atan2(accel_y, math.sqrt(accel_x**2 + accel_z**2)))
        roll = math.degrees(math.atan2(-accel_x, accel_z))
        
        return math.sqrt(pitch**2 + roll**2)

class DrowsinessDetector:
    def __init__(self):
        # MediaPipe setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.eyes_closed_start_time = None
        self.is_alert_active = False
        
    def calculate_eye_aspect_ratio(self, eye_landmarks):
        # Vertical eye landmarks
        v1 = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
        v2 = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
        
        # Horizontal eye landmark
        h = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
        
        # Eye Aspect Ratio
        ear = (v1 + v2) / (2.0 * h)
        return ear

    def detect_drowsiness(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.face_mesh.process(rgb_frame)
        
        both_eyes_closed = False
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Eye landmark indices (specific to MediaPipe Face Mesh)
                left_eye_indices = [33, 160, 158, 133, 153, 144]
                right_eye_indices = [362, 385, 387, 263, 373, 380]
                
                # Extract eye landmarks
                left_eye = [(int(face_landmarks.landmark[i].x * frame.shape[1]), 
                             int(face_landmarks.landmark[i].y * frame.shape[0])) 
                            for i in left_eye_indices]
                right_eye = [(int(face_landmarks.landmark[i].x * frame.shape[1]), 
                              int(face_landmarks.landmark[i].y * frame.shape[0])) 
                             for i in right_eye_indices]
                
                # Compute Eye Aspect Ratios
                left_ear = self.calculate_eye_aspect_ratio(left_eye)
                right_ear = self.calculate_eye_aspect_ratio(right_eye)
                
                # Classify eye states
                left_state = "Closed" if left_ear < 0.2 else "Open"
                right_state = "Closed" if right_ear < 0.2 else "Open"
                
                # Check if both eyes are closed
                if left_state == "Closed" and right_state == "Closed":
                    both_eyes_closed = True
                else:
                    both_eyes_closed = False
                    # Reset timer if eyes open
                    self.eyes_closed_start_time = None
                    
                    # Deactivate alert if it was previously active
                    if self.is_alert_active:
                        self.is_alert_active = False
                        # Turn off buzzer and LED when eyes open
                        GPIO.output(BUZZER_PIN, GPIO.LOW)
                        GPIO.output(LED_PIN, GPIO.LOW)
                
                # Drowsiness detection logic
                if both_eyes_closed:
                    if self.eyes_closed_start_time is None:
                        # Start timer when both eyes first close
                        self.eyes_closed_start_time = time.time()
                    else:
                        # Check duration of closed eyes
                        current_time = time.time()
                        if current_time - self.eyes_closed_start_time >= CLOSED_DURATION_THRESHOLD:
                            # Trigger alert
                            if not self.is_alert_active:
                                self.is_alert_active = True
                                # Activate buzzer and LED for drowsiness
                                GPIO.output(BUZZER_PIN, GPIO.HIGH)
                                GPIO.output(LED_PIN, GPIO.HIGH)
                
                # Draw bounding boxes and state
                def draw_eye_info(eye_points, state, side):
                    x_min = min(point[0] for point in eye_points)
                    y_min = min(point[1] for point in eye_points)
                    x_max = max(point[0] for point in eye_points)
                    y_max = max(point[1] for point in eye_points)
                    
                    # Draw bounding box
                    cv2.rectangle(
                        frame, 

                        (x_min, y_min), 
                        (x_max, y_max), 
                        (0, 255, 0) if state == "Open" else (0, 0, 255), 
                        2
                    )
                    
                    # Add eye state label
                    cv2.putText(
                        frame, 
                        f"{side} Eye: {state}", 
                        (x_min, y_min - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, 
                        (0, 255, 0) if state == "Open" else (0, 0, 255), 
                        2
                    )
                
                # Process and draw eye information
                draw_eye_info(left_eye, left_state, "Left")
                draw_eye_info(right_eye, right_state, "Right")
        
        return frame

class SafetySystem:
    def __init__(self):
        # GPIO Setup
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(BUZZER_PIN, GPIO.OUT)
        GPIO.setup(LED_PIN, GPIO.OUT)
        GPIO.setup(ULTRASONIC_TRIG, GPIO.OUT)
        GPIO.setup(ULTRASONIC_ECHO, GPIO.IN)
        GPIO.setup(ALCOHOL_SENSOR_PIN, GPIO.IN)

        # Initialize components
        self.drowsiness_detector = DrowsinessDetector()
        
        # Initialize MPU6050
        try:
            self.mpu = MPU6050()
            self.fall_detection_active = True
        except Exception as e:
            print(f"MPU6050 initialization error: {e}")
            self.fall_detection_active = False

        # Initialize Twilio client
        self.twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        # Track fall alert to prevent multiple messages
        self.fall_alert_sent = False

    def get_location_coordinates(self):
        """
        Get approximate coordinates using Google Geolocation API
        """
        try:
            # Try to get location using Google's Geolocation API
            url = "https://www.googleapis.com/geolocation/v1/geolocate?key=" + GOOGLE_MAPS_API_KEY
            response = requests.post(url)
            data = response.json()
            
            if 'location' in data:
                lat = data['location']['lat']
                lon = data['location']['lng']
                print(f"Location detected: {lat}, {lon}")
                return lat, lon
            else:
                raise Exception("No location data in response")
        except Exception as e:
            print(f"Geolocation error: {e}")
            # Return fallback coordinates if geolocation fails
            return 37.7749, -122.4194  # Fallback to mock location

    def measure_distance(self):
        """
        Measure distance using ultrasonic sensor to detect helmet
        """
        # Trigger ultrasonic pulse
        GPIO.output(ULTRASONIC_TRIG, True)
        time.sleep(0.00001)
        GPIO.output(ULTRASONIC_TRIG, False)

        # Wait for echo
        start_time = time.time()
        stop_time = time.time()

        while GPIO.input(ULTRASONIC_ECHO) == 0:
            start_time = time.time()

        while GPIO.input(ULTRASONIC_ECHO) == 1:
            stop_time = time.time()

        # Calculate distance
        duration = stop_time - start_time
        distance = (duration * 34300) / 2  # Speed of sound is 343 m/s
        return distance

    def check_alcohol(self):
        """
        Check alcohol sensor status
        """
        return GPIO.input(ALCOHOL_SENSOR_PIN) == GPIO.HIGH

    def detect_fall(self):
        """
        Detect fall using MPU6050
        """
        if not self.fall_detection_active:
            return False
        
        try:
            tilt_angle = self.mpu.calculate_tilt_angle()
            
            if tilt_angle > FALL_ANGLE_THRESHOLD:
                return True
            
            return False
        except Exception as e:
            print(f"Fall detection error: {e}")
            return False

    def send_alert(self, message):
        """
        Send alert message using Twilio SMS
        """
        try:
            # Send SMS via Twilio
            sms = self.twilio_client.messages.create(
                body=message,
                from_=TWILIO_PHONE_NUMBER,
                to=RECIPIENT_PHONE_NUMBER
            )
            print(f"Alert sent with SID: {sms.sid}")
            return True
        except Exception as e:
            print(f"Error sending Twilio alert: {e}")
            return False

    def send_location_alert(self, lat, lon):
        """
        Send location alert with Google Maps link
        """
        maps_url = f"https://maps.google.com/maps?q={lat},{lon}"
        message = f"EMERGENCY ALERT: Fall detected! Location: {maps_url}"
        return self.send_alert(message)

    def upload_safety_data(self, helmet_status, alcohol_status, drowsy_status):
        """
        Upload safety data to Firebase
        """
        try:
            ref = db.reference('safety_status')
            ref.set({
                'helmet_worn': helmet_status,
                'alcohol_detected': alcohol_status,
                'drowsy': drowsy_status,
                'timestamp': time.time()
            })
        except Exception as e:
            print(f"Firebase upload error: {e}")

def main():
    # Initialize camera
    camera = cv2.VideoCapture(0)
    
    # Create safety system
    safety_system = SafetySystem()
    
    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame for drowsiness
            processed_frame = safety_system.drowsiness_detector.detect_drowsiness(frame)
            
            # Check helmet
            helmet_distance = safety_system.measure_distance()
            helmet_worn = helmet_distance < HELMET_DISTANCE_THRESHOLD
            
            # Check alcohol
            alcohol_detected = safety_system.check_alcohol()
            
            # Check drowsiness
            is_drowsy = safety_system.drowsiness_detector.is_alert_active
            
            # Check for fall
            fall_detected = safety_system.detect_fall()
            if fall_detected and not safety_system.fall_alert_sent:
                # Get location coordinates and send alert
                lat, lon = safety_system.get_location_coordinates()
                safety_system.send_location_alert(lat, lon)
                safety_system.fall_alert_sent = True
                
                # Removed buzzer and LED from fall detection
            
            # Reset fall alert status after some time to allow new alerts
            if not fall_detected and safety_system.fall_alert_sent:
                safety_system.fall_alert_sent = False
            
            # Upload safety data to Firebase
            safety_system.upload_safety_data(
                helmet_status=helmet_worn, 
                alcohol_status=alcohol_detected, 
                drowsy_status=is_drowsy
            )
            
            # Add safety status text
            status_text = "SAFE" if (helmet_worn and not alcohol_detected and not is_drowsy) else "UNSAFE"
            status_color = (0, 255, 0) if status_text == "SAFE" else (0, 0, 255)
            cv2.putText(
                processed_frame, 
                f"Status: {status_text}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                status_color, 
                2
            )
            
            # Display frame
            cv2.imshow("Drowsiness Detection", processed_frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Cleanup
        camera.release()
        cv2.destroyAllWindows()
        GPIO.cleanup()

if __name__ == "__main__":
    main()