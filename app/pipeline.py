import io
from PIL import Image, ImageDraw
from flask import current_app as app
import os

from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections

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