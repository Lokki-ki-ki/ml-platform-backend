from flask import jsonify
from werkzeug.exceptions import HTTPException

def setup_error_handlers(app):
    @app.errorhandler(404)
    def page_not_found(e):
        return jsonify(error=str(e)), 404

    @app.errorhandler(500)
    def internal_server_error(e):
        return jsonify(error=str(e)), 500
    
    # @app.errorhandler(HTTPException)
    # def handle_http_exception(error):
    #     app.logger.error(error.description)
    #     return jsonify(error=str(error.description)), 501