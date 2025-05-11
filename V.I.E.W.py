import time
import datetime
import os
import RPi.GPIO as GPIO
from camera import Camera
from sensor import InfraredSensor, Radar, GPS, ServoMotor
from storage import SSD

GPIO.setmode(GPIO.BCM)

SERVO_PIN = 18
INFRARED_PIN = 17
RADAR_PIN = 22
SSD_MOUNT_POINT = "/mnt/ssd"
VIDEO_DURATION = 600
DATA_LOGGING_INTERVAL = 1
LOGGING_PERIOD = 604800
VIDEO_STORAGE_PATH = os.path.join(SSD_MOUNT_POINT, "videos")
DATA_STORAGE_PATH = os.path.join(SSD_MOUNT_POINT, "sensor_data.txt")

servo = ServoMotor(pin=SERVO_PIN)
infrared_sensor = InfraredSensor(pin=INFRARED_PIN)
radar = Radar(pin=RADAR_PIN)
gps_module = GPS()
camera = Camera(resolution="11K")
ssd = SSD(mount_point=SSD_MOUNT_POINT)

os.makedirs(VIDEO_STORAGE_PATH, exist_ok=True)

def log_sensor_data(time_stamp, distance, radar_data, location):
    with open(DATA_STORAGE_PATH, "a") as log_file:
        log_file.write(f"{time_stamp}, {distance}, {radar_data}, {location}\n")

def check_and_delete_old_logs():
    now = time.time()
    if os.path.exists(DATA_STORAGE_PATH):
        with open(DATA_STORAGE_PATH, "r") as log_file:
            lines = log_file.readlines()
        with open(DATA_STORAGE_PATH, "w") as log_file:
            for line in lines:
                log_time = float(line.split(",")[0])
                if now - log_time <= LOGGING_PERIOD:
                    log_file.write(line)

def save_video(video_name):
    camera.start_recording(os.path.join(VIDEO_STORAGE_PATH, video_name))
    time.sleep(VIDEO_DURATION)
    camera.stop_recording()

def display_output(output):
    print(f"Display Output: {output}")

def classify_object(distance, radar_data):
    # Dummy classification logic
    # You can modify this logic as per your radar data interpretation
    # Assuming radar_data includes the type of object detected
    if radar_data in ["bird", "balloon"]:
        return True, None  # Safe
    elif radar_data in ["fighter jet", "drone"]:
        return False, f"unsafe_object_{int(time.time())}.h264"  # Unsafe
    return True, None  # Default to safe for other cases

def main():
    while True:
        servo.rotate_continuously()
        distance = infrared_sensor.get_distance()
        radar_data = radar.get_data()  # Assuming this returns the object type
        location = gps_module.get_location()
        time_stamp = time.time()
        log_sensor_data(time_stamp, distance, radar_data, location)
        check_and_delete_old_logs()

        is_safe, video_name = classify_object(distance, radar_data)
        if not is_safe:
            save_video(video_name)
            display_output(f"Unsafe object detected! Video saved: {video_name}")
        else:
            time.sleep(VIDEO_DURATION)  # Sleep duration to simulate safe protocol
            display_output("Safe object detected. No video recorded.")
        
        time.sleep(DATA_LOGGING_INTERVAL)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        GPIO.cleanup()
