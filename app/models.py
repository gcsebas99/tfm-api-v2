from datetime import datetime
from app import db

class Event(db.Model):
    __tablename__ = 'event'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    created = db.Column(db.DateTime, default=datetime.utcnow)
    updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    commercial_id = db.Column(db.Integer, nullable=False)
    status = db.Column(db.String(20), default='initialized', nullable=False)
    country_name = db.Column(db.String(128), nullable=True)
    country_code = db.Column(db.String(10), nullable=True)

    STATUS_CHOICES = [
        ('initialized', 'Initialized'),
        ('completed', 'Completed')
    ]

    def __repr__(self):
        return '<Event {}>'.format(self.id)


class EventRecognition(db.Model):
    __tablename__ = 'event_recognition'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    created = db.Column(db.DateTime, default=datetime.utcnow)
    updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    event_id = db.Column(db.Integer, db.ForeignKey('event.id', ondelete='CASCADE'), nullable=False)
    second = db.Column(db.Integer, nullable=False)
    face = db.Column(db.Boolean, default=False, nullable=False)
    age = db.Column(db.Integer, nullable=True)
    gender = db.Column(db.Integer, default=0, nullable=False)
    percent_neutral = db.Column(db.Float, default=0.0, nullable=False)
    percent_happy = db.Column(db.Float, default=0.0, nullable=False)
    percent_angry = db.Column(db.Float, default=0.0, nullable=False)
    percent_sad = db.Column(db.Float, default=0.0, nullable=False)
    percent_fear = db.Column(db.Float, default=0.0, nullable=False)
    percent_surprise = db.Column(db.Float, default=0.0, nullable=False)
    percent_disgust = db.Column(db.Float, default=0.0, nullable=False)
    percent_contempt = db.Column(db.Float, default=0.0, nullable=False)

    GENDER_CHOICES = [
        (0, 'Unknown'),
        (1, 'Female'),
        (2, 'Male')
    ]

    event = db.relationship('Event', backref=db.backref('recognitions', cascade='all, delete-orphan'))

    def __repr__(self):
        return '<EventRecognition {}>'.format(self.id)
