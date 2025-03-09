import io
from PIL import Image, ImageDraw
from flask import current_app as app
import os

from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from emotion_model import process_emotions

import cv2
import numpy as np
import mediapipe as mp

from memory_profiler import profile




# Define a permanent folder to store images
UPLOAD_FOLDER = "uploads"  # Change this to your preferred directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the folder exists

# Download and load the YOLOv8 face detection model
MODEL_REPO = "arnabdhar/YOLOv8-Face-Detection"
MODEL_FILENAME = "model.pt"

model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
model = YOLO(model_path)

# using a YOLO model to detect faces
def process_image_yolo(image_file, second):
    # Load image
    image = Image.open(image_file.stream).convert("RGB")

    # Run inference
    output = model(image)
    results = Detections.from_ultralytics(output[0])

    # Extract bounding boxes
    boxes = results.xyxy  # Bounding boxes in [x_min, y_min, x_max, y_max] format

    # Draw bounding boxes if faces are detected
    if len(boxes) > 0:
        draw = ImageDraw.Draw(image)
        for i, box in enumerate(boxes):
            x_min, y_min, x_max, y_max = map(int, box)

            face_crop = image.crop((x_min, y_min, x_max, y_max))

            # draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red", width=3)

            # Save the cropped face
            # face_filename = f"face_yolo_{i}_{second}.jpg"
            # face_path = os.path.join(UPLOAD_FOLDER, face_filename)
            # face_crop.save(face_path)
            # app.logger.info(f"Saved cropped face as {face_filename}")
            app.logger.info(f"Face recognized in second {second} in rect [{x_min}, {y_min}, {x_max}, {y_max}]")


    # Save processed image to a BytesIO buffer
    img_io = io.BytesIO()
    image.save(img_io, format="JPEG")

    # # Define the path where the image will be saved
    # filename = f"screenshot_yolo_original.jpg"
    # file_path = os.path.join(UPLOAD_FOLDER, filename)
    # # Save original image
    # image.save(file_path)


    return {
        "face": True,
        "age": 25,
        "gender": 2,  # 1 for Female, 2 for Male
        "percent_neutral": 40.0,
        "percent_happy": 35.0,
        "percent_angry": 5.0,
        "percent_sad": 10.0,
        "percent_fear": 2.0,
        "percent_surprise": 6.0,
        "percent_disgust": 1.0,
        "percent_contempt": 1.0
    }


# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# @profile
def process_image_mediapipe(image_file, second):
    try:
        # Read image
        image_bytes = image_file.read()
        image_np = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        # Convert to RGB (Mediapipe requires RGB format)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run face detection
        results = face_detection.process(img_rgb)
        # faces = []

        if results.detections:
            for i, detection in enumerate(results.detections):
                bbox = detection.location_data.relative_bounding_box

                # Convert relative bbox to absolute pixel values
                img_height, img_width, _ = img.shape
                x_min = int(bbox.xmin * img_width)
                y_min = int(bbox.ymin * img_height)
                width = int(bbox.width * img_width)
                height = int(bbox.height * img_height)


                # # Add padding
                # padding = 20
                # x1, y1 = max(x_min - padding, 0), max(y_min - padding, 0)
                # x2, y2 = min(x_min + width + padding, img_width), min(y_min + height + padding, img_height)

                # # Crop the face with padding
                # face_crop = img[y1:y2, x1:x2]

                # # Calculate padding needed if the crop hits image edges
                # top_pad = abs(y_min - padding) if y_min - padding < 0 else 0
                # bottom_pad = abs((y_min + height + padding) - img_height) if (y_min + height + padding) > img_height else 0
                # left_pad = abs(x_min - padding) if x_min - padding < 0 else 0
                # right_pad = abs((x_min + width + padding) - img_width) if (x_min + width + padding) > img_width else 0

                # # Add white padding if necessary
                # face_crop = cv2.copyMakeBorder(
                #     face_crop,
                #     top_pad, bottom_pad, left_pad, right_pad,
                #     cv2.BORDER_CONSTANT,
                #     value=[255, 255, 255]  # White color
                # )


                # Ensure coordinates are within bounds
                x_max = min(x_min + width, img_width)
                y_max = min(y_min + height, img_height)

                # Crop the face
                face_crop = img[y_min:y_max, x_min:x_max]

                # Process emotions
                result_emotions=process_emotions(face_crop)

                # Save the cropped face
                face_filename = f"face_mediapipe_{i}_{second}.jpg"
                face_path = os.path.join(UPLOAD_FOLDER, face_filename)
                cv2.imwrite(face_path, face_crop)  # Correct method for saving OpenCV images

                # faces.append({
                #     "xmin": x_min,
                #     "ymin": y_min,
                #     "width": width,
                #     "height": height,
                #     "score": detection.score[0]
                # })

                print(f"✅ Face recognized in second {second}")

        # return faces

    except Exception as e:
        print(f"❌ Error processing image: {str(e)}")
        return None

    # pass
    return {
        "face": True,
        "age": 25,
        "gender": 2,  # 1 for Female, 2 for Male
        "percent_neutral": result_emotions["neutral"],
        "percent_happy": result_emotions["happy"],
        "percent_angry": result_emotions["angry"],
        "percent_sad": result_emotions["sad"],
        "percent_fear": result_emotions["fear"],
        "percent_surprise": result_emotions["surprise"],
        "percent_disgust": result_emotions["disgust"],
        "percent_contempt": result_emotions["contempt"],
    }

def process_image_random():
    emotions = ["percent_neutral", "percent_happy", "percent_angry", "percent_sad",
                "percent_fear", "percent_surprise", "percent_disgust", "percent_contempt"]

    # Generate random values
    random_values = np.random.dirichlet(np.ones(len(emotions))) * 100

    # Create response dictionary
    resran = {
        "face": True,
        "age": int(25),
        "gender": int(np.random.choice([1, 2])),  # Randomize gender
    }

    # Assign randomized values
    for emotion, value in zip(emotions, random_values):
        resran[emotion] = float(round(value, 1))   # Round to 1 decimal place

    return resran