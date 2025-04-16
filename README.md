# Tech-A-Thon-2025-Smart-Helmet

#This code consists of UNIQUE API KEYS, PLEASE CHANGE API KEYS FOR TWILIO AND GPS BEFORE YOU RUN THIS CODE ON YOUR SYSTEM
#This code has theoritical threshold values for Fall detection, and may return fall positives or no detection for real life fall, used only for sample demonstration
#Description : 
#Developed a smart helmet to enhance rider safety using Raspberry Pi 4 and ESP32. The system verifies helmet usage via an HC-SR04 ultrasonic sensor, detects alcohol with an MQ-3 sensor, and monitors drowsiness through eye pattern recognition using a camera and MediaPipe (OpenCV pretrained ML datasets). Fall detection is enabled using an MPU6050 accelerometer. Safety data is sent to Firebase Realtime Database and used to control an ESP32-based servo lock. The bike unlocks only when the rider is wearing the helmet, is sober, and not drowsy. Additionally, the system uses Twilio API to send emergency SMS alerts with GPS location in the event of a fall.
