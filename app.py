from flask import Flask, request, jsonify
from flask_restful import Api
from evaluator import Evaluator
from werkzeug.exceptions import HTTPException
from utils import logging_utils
from asgiref.wsgi import WsgiToAsgi
from error_handlers import setup_error_handlers
logging_utils.setup_logging("mainApp.log")


app = Flask(__name__)
asgi_app = WsgiToAsgi(app)
api = Api(app)

setup_error_handlers(app)

# Log header and request data before each request
# TODO: Later convert this to log to a file
@app.before_request
def log_request_info():
    app.logger.debug('Headers: %s', request.headers)
    app.logger.debug('Body: %s', request.get_data())

api.add_resource(Evaluator, '/api/evaluator')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='8080', threaded=True)

