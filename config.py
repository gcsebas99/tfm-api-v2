import os
basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
    #     'sqlite:///' + os.path.join(basedir, 'local.db')

class DevelopmentConfig(Config):
    # Use SQLite for local development
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'local.db') # os.getenv('DEV_DATABASE_URL', 'sqlite:///local.db')

class ProductionConfig(Config):
    # Use the Heroku Postgres database URL
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL')

# Choose configuration based on environment
def get_config():
    env = os.getenv('FLASK_ENV', 'development')
    if env == 'production':
        return ProductionConfig
    return DevelopmentConfig