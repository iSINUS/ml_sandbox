from ..main import app, api
from ..api import apis
from flask_restplus import Resource



@api.route("/hello")
class Hello(Resource):
    def get(self):
        # This could also be returning an index.html
        return 'Attrition Prediction using Flask in a uWSGI Nginx Docker container with Python 3.7'