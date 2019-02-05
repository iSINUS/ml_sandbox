from flask import Flask
from celery import Celery
from flask_restplus import Api

api = Api()
app = Flask(__name__)
api.init_app(app, version='1.0', title='ML Attrition Prediction Box')

app.config['CELERY_BROKER_URL'] = 'redis://redis:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://redis:6379/0'

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

from .core import hello

if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host='0.0.0.0', debug=True, port=80)
    argv = [
        'worker',
        '--loglevel=DEBUG',
    ]
    celery.worker_main(argv)
