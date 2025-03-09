from dotenv import load_dotenv
load_dotenv()
from app.utils import get_client_ip
from flask import Flask, jsonify, request
from flask_restful import Resource, Api
from flask_cors import CORS
from config import get_config
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from datetime import datetime
from app.pipeline import process_image_yolo, process_image_mediapipe, process_image_random
import requests
import logging
import numpy as np
import psutil
import os
import random
import mediapipe as mp

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB

# Define a permanent folder to store images
# UPLOAD_FOLDER = "uploads"  # Change this to your preferred directory
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the folder exists

# APP CONFIG
app = Flask(__name__)
app.config.from_object(get_config())
logging.basicConfig(level=logging.INFO)
db = SQLAlchemy(app)
migrate = Migrate(app, db)
api = Api(app)
CORS(app)
# CORS(app, origins=["https://tfm-front-d58c83913652.herokuapp.com"])

from app.models import Event
from app.models import EventRecognition

# ENDPOINTS
class Status(Resource):
     def get(self):
         try:
            return {'data': 'Api running'}
         except Exception as error:
            return {'data': str(error)}

class CommercialApi(Resource):
    def get(self):
        try:
            # Parse commercial_id from query parameters
            commercial_id = request.args.get('id', type=int)

            if commercial_id is None:
                return {'error': 'id is required'}, 400

            # Query events by commercial_id
            events = Event.query.filter_by(commercial_id=commercial_id, status='completed').all()

            # Build the response
            result = []
            for event in events:
                recognitions = EventRecognition.query.filter_by(event_id=event.id).all()
                result.append({
                    'id': event.id,
                    # 'country_name': event.country_name,
                    'cc': event.country_code, # country code
                    # 'created': event.created.isoformat(),
                    # 'updated': event.updated.isoformat(),
                    # 'status': event.status,
                    'rcs': [ # recognitions
                        {
                            'id': rec.id,
                            # 'created': rec.created.isoformat(),
                            # 'updated': rec.updated.isoformat(),
                            # 'event_id': rec.event_id,
                            's': rec.second,
                            'f': rec.face, # face
                            'a': rec.age, # age
                            'g': rec.gender, # gender
                            'neutral': rec.percent_neutral,
                            'happy': rec.percent_happy,
                            'angry': rec.percent_angry,
                            'sad': rec.percent_sad,
                            'fear': rec.percent_fear,
                            'surprise': rec.percent_surprise,
                            'disgust': rec.percent_disgust,
                            'contempt': rec.percent_contempt
                        }
                        for rec in recognitions
                    ]
                })

            return result, 200

        except Exception as e:
            return {'error': str(e)}, 500

class EventApi(Resource):
    def post(self):
        try:
            # Parse and validate input
            data = request.get_json()
            commercial_id = data.get('commercial_id')

            if commercial_id is None:
                return {'error': 'commercial_id is required'}, 400
            if not isinstance(commercial_id, int):
                return {'error': 'commercial_id must be an integer'}, 400




            # UNCOMMENT THIS

            # client_ip = get_client_ip()

            # # If running locally, avoid sending private/local IPs (e.g., 127.0.0.1)
            # if client_ip.startswith(('192.', '10.', '172.', '127.')):
            #     #replace with fake ip from costa rica
            #     client_ip = '201.207.176.114'
            #     # return {'error': f'Cannot determine geolocation for private IPs [{client_ip}]'}, 400

            # external_api_url = f'https://reallyfreegeoip.org/json/{client_ip}'

            # country_name = 'Unknown'
            # country_code = 'XX'

            # try:
            #     response = requests.get(external_api_url)
            #     response.raise_for_status()
            #     country_info = response.json()
            #     country_name = country_info['country_name']
            #     country_code = country_info['country_code']
            # except requests.exceptions.RequestException as e:
            #     country_name = 'Unknown'
            #     country_code = 'XX'


            country_name = 'Unknown'
            country_code = np.random.choice(['CR', 'MX', 'ES', 'PA', 'CA', 'IN', 'US', 'RU'])


            # Create a new Event instance
            new_event = Event(commercial_id=commercial_id, country_code=country_code, country_name=country_name)
            db.session.add(new_event)
            db.session.commit()

            # Return the created Event as JSON
            return {
                'id': new_event.id,
                'created': new_event.created.isoformat(),
                'updated': new_event.updated.isoformat(),
                'commercial_id': new_event.commercial_id,
                'status': new_event.status,
                'country_name': new_event.country_name,
                'country_code': new_event.country_code,
            }, 201
        except Exception as e:
            return {'error': str(e)}, 500
    def put(self):
        try:
            # Parse and validate input
            data = request.get_json()
            event_id = data.get('id')
            new_status = data.get('status')

            if event_id is None:
                return {'error': 'id is required'}, 400
            if not isinstance(event_id, int):
                return {'error': 'id must be an integer'}, 400
            if new_status not in ['initialized', 'completed']:
                return {'error': f'status must be one of {Event.STATUS_CHOICES}'}, 400

            # Retrieve the event by id
            event = Event.query.get(event_id)
            if not event:
                return {'error': 'Event not found'}, 404

            # Update the status and commit the changes
            event.status = new_status
            db.session.commit()

            # Return the updated event as JSON
            return {
                'id': event.id,
                'created': event.created.isoformat(),
                'updated': event.updated.isoformat(),
                'commercial_id': event.commercial_id,
                'status': event.status,
                'country_name': event.country_name,
                'country_code': event.country_code,
            }, 200
        except Exception as e:
            return {'error': str(e)}, 500
    def get(self):
        try:
            # Parse the event ID from the query parameters
            event_id = request.args.get('id', type=int)

            if event_id is None:
                return {'error': 'id is required'}, 400

            # Retrieve the event by id
            event = Event.query.get(event_id)
            if not event:
                return {'error': 'Event not found'}, 404

            # Return the event as JSON
            return {
                'id': event.id,
                'created': event.created.isoformat(),
                'updated': event.updated.isoformat(),
                'commercial_id': event.commercial_id,
                'status': event.status,
                'country_name': event.country_name,
                'country_code': event.country_code,
            }, 200
        except Exception as e:
            return {'error': str(e)}, 500

class RecognitionApi(Resource):
    def get(self):
        try:
            # Parse the recognition ID from the query parameters
            recognition_id = request.args.get('id', type=int)

            if recognition_id is None:
                return {'error': 'id is required'}, 400

            # Retrieve the event by id
            recognition = EventRecognition.query.get(recognition_id)
            if not recognition:
                return {'error': 'Recognition not found'}, 404

            # Return the recognition as JSON
            return {
                'id': recognition.id,
                'created': recognition.created.isoformat(),
                'updated': recognition.updated.isoformat(),
                'event_id': recognition.event_id,
                'second': recognition.second,
                'face': recognition.face,
                'age': recognition.age,
                'gender': recognition.gender,
                'percent_neutral': recognition.percent_neutral,
                'percent_happy': recognition.percent_happy,
                'percent_angry': recognition.percent_angry,
                'percent_sad': recognition.percent_sad,
                'percent_fear': recognition.percent_fear,
                'percent_surprise': recognition.percent_surprise,
                'percent_disgust': recognition.percent_disgust,
                'percent_contempt': recognition.percent_contempt
            }, 200
        except Exception as e:
            return {'error': str(e)}, 500
    def post(self):
        try:
            # Validate and parse input
            event_id = request.form.get("event_id")
            second = request.form.get("second")
            image = request.files.get('image')

            app.logger.info(f"Event value: {event_id} Second value: {second}")

            if event_id is None or second is None:
                return {"error": "event_id and second are required"}, 400

            try:
                event_id = int(event_id)  # Convert to int
                second = int(second)  # Convert to int
            except ValueError:
                return {"error": "event_id and second must be integers"}, 400

            if not image:
                return {'error': 'image is required'}, 400
            if not image.content_type.startswith('image/'):
                return {'error': 'Uploaded file must be an image'}, 400


            # # Define the path where the image will be saved
            # filename = f"screenshot_{event_id}_{second}.jpg"
            # file_path = os.path.join(UPLOAD_FOLDER, filename)

            # # Save the image
            # image.save(file_path)



            # Validate event_id exists
            event = Event.query.get(event_id)
            if not event:
                return {'error': 'Event not found'}, 404

            # Process the image: detect face, age, gender and facial expressions
            # image_data = process_image_yolo(image, second)
            # image_data = fake_pipeline(image, second)
            image_data = process_image_mediapipe(image, second)
            # image_data = process_image_random()

            # Create a new EventRecognition object
            recognition = EventRecognition(
                event_id=event_id,
                second=second,
                face=image_data['face'],
                age=image_data['age'],
                gender=image_data['gender'],
                percent_neutral=image_data['percent_neutral'],
                percent_happy=image_data['percent_happy'],
                percent_angry=image_data['percent_angry'],
                percent_sad=image_data['percent_sad'],
                percent_fear=image_data['percent_fear'],
                percent_surprise=image_data['percent_surprise'],
                percent_disgust=image_data['percent_disgust'],
                percent_contempt=image_data['percent_contempt']
            )

            # Save the recognition to the database
            db.session.add(recognition)
            db.session.commit()

            # Return the created object as JSON
            return {
                'id': recognition.id,
                'created': recognition.created.isoformat(),
                'updated': recognition.updated.isoformat(),
                'event_id': recognition.event_id,
                'second': recognition.second,
                'face': recognition.face,
                'age': recognition.age,
                'gender': recognition.gender,
                'percent_neutral': recognition.percent_neutral,
                'percent_happy': recognition.percent_happy,
                'percent_angry': recognition.percent_angry,
                'percent_sad': recognition.percent_sad,
                'percent_fear': recognition.percent_fear,
                'percent_surprise': recognition.percent_surprise,
                'percent_disgust': recognition.percent_disgust,
                'percent_contempt': recognition.percent_contempt
            }, 201

        except Exception as e:
            return {'error': str(e)}, 500

class RecognitionBatchApi(Resource):
    def post(self):
        try:
            # Validate and parse input
            event_id = request.form.get("event_id")
            num_images = request.form.get("num_images")
            images = request.files.getlist("images")

            app.logger.info(f"Event value: {event_id}, Number of images: {num_images}")

            if event_id is None or num_images is None:
                return {"error": "event_id and num_images are required"}, 400

            try:
                event_id = int(event_id)  # Convert to int
                num_images = int(num_images)  # Convert to int
            except ValueError:
                return {"error": "event_id and num_images must be integers"}, 400

            if not images or len(images) != num_images:
                return {'error': 'Number of uploaded images must match num_images'}, 400

            # Validate event_id exists
            event = Event.query.get(event_id)
            if not event:
                return {'error': 'Event not found'}, 404

            print(f"Memory usage before processing: {get_memory_usage()} MB")
            recognitions = []
            age = random.randint(14, 70)
            gender = random.randint(1, 2)

            for second, image in enumerate(images):
                if not image.content_type.startswith('image/'):
                    return {'error': f'File at position {second} is not an image'}, 400

                # Process the image
                # image_data = process_image_random()
                image_data = process_image_mediapipe(image, second)


                # Create new recognition entry
                recognition = EventRecognition(
                    event_id=event_id,
                    second=second,
                    face=image_data['face'],
                    age=age,
                    gender=gender,
                    percent_neutral=image_data['percent_neutral'],
                    percent_happy=image_data['percent_happy'],
                    percent_angry=image_data['percent_angry'],
                    percent_sad=image_data['percent_sad'],
                    percent_fear=image_data['percent_fear'],
                    percent_surprise=image_data['percent_surprise'],
                    percent_disgust=image_data['percent_disgust'],
                    percent_contempt=image_data['percent_contempt']
                )
                db.session.add(recognition)
                recognitions.append(recognition)

            db.session.commit()
            print(f"Memory usage after processing: {get_memory_usage()} MB")


            return {'message': 'Batch processing successful', 'processed': len(recognitions)}, 201

        except Exception as e:
            return {'error': str(e)}, 500

api.add_resource(Status,'/')
api.add_resource(CommercialApi,'/commercial')
api.add_resource(EventApi,'/event')
api.add_resource(RecognitionApi,'/recognition')
api.add_resource(RecognitionBatchApi, '/recognitionbatch')
