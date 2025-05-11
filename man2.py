import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random  # For simulating MPU-6050 data

# Load YOLO model with CPU support (without CUDA)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=True)
model.to('cpu')  # Ensure model runs on CPU
print("Model loaded successfully!")

# Initialize 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Depth')
ax.set_title("3D Object Mapping")

# Object styles for unique objects
object_styles = {
    "airplane": {"color": 'r', "label": "Airplane", "size": 60},
    "drone": {"color": 'g', "label": "Drone", "size": 50},
    "fighter jet": {"color": 'b', "label": "Fighter Jet", "size": 70},
    "bird": {"color": 'y', "label": "Bird", "size": 30},
    "hot air balloon": {"color": 'm', "label": "Hot Air Balloon", "size": 40},
}

# Specify the video path
video_path = "C:\\Users\\RAJA\\Downloads\\Birds Flying  in the Sky at Sunset -🌅beauty of nature- bird music video-birds flying video.mp4"  # Change this to your video file path
cap = cv2.VideoCapture(video_path)

# Function to estimate depth (z) based on bounding box size
def estimate_depth(x1, y1, x2, y2):
    area = (x2 - x1) * (y2 - y1)
    return 1000 / max(area, 1)  # Inverse proportion for depth estimation

# Simulated MPU-6050 data function
def read_mpu6050():
    # Simulating accelerometer data
    accel_x = random.uniform(-2, 2)  # Simulated range for accelerometer x-axis
    accel_y = random.uniform(-2, 2)  # Simulated range for accelerometer y-axis
    accel_z = random.uniform(0, 2)    # Simulated range for accelerometer z-axis
    return accel_x, accel_y, accel_z

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame or end of video reached")
        break

    # Detect objects in the frame
    results = model(frame)
    detections = results.pred[0].cpu().numpy()  # Fetch detections on CPU

    # Print detections for debugging
    print("Detections:", detections)

    # Clear previous points for fresh update
    ax.cla()
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Depth')
    ax.set_title("3D Object Mapping")

    for det in detections:
        class_id = int(det[5])
        confidence = det[4]

        if confidence < 0.3:  # Lowering the threshold for testing
            continue

        # Get bounding box and estimate depth
        x1, y1, x2, y2 = map(int, det[:4])
        center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
        z = estimate_depth(x1, y1, x2, y2)  # Simulated depth estimation
        print("Estimated depth (z):", z)
        
        # Print object details
        class_name = model.names[class_id]
        print(f"Detected: {class_name} | X: {center_x} | Y: {center_y} | Confidence: {confidence}")

        # Determine object type and style
        if class_name in object_styles:
            style = object_styles[class_name]
            color = style["color"]

            # Simulate MPU-6050 data for x, y, z positioning
            accel_x, accel_y, accel_z = read_mpu6050()

            # Plot in 3D space with defined color and size
            ax.scatter(center_x + accel_x * 10, center_y + accel_y * 10, z + accel_z * 10, 
                       c=color, label=style["label"], s=style["size"])

            # Display bounding box and label on video feed
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            label = f"{style['label']} | Confidence: {int(confidence * 100)}%"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the video feed and 3D plot
    cv2.imshow("Object Tracking Feed", frame)
    plt.pause(0.01)  # Update the plot in real-time

    # Quit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.close()
