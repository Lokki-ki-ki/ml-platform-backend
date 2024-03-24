from flask import Flask, request, jsonify
import logging
from flask_restful import Api
from app.routes.evaluator import Evaluator
from app.utils import setup_logging
from app.utils import setup_error_handlers
setup_logging("mainApp.log")

def create_app():
    app = Flask(__name__)
    api = Api(app)

    setup_error_handlers(app)

    api.add_resource(Evaluator, '/api/evaluator')

    return app


# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port='8080', threaded=True)

