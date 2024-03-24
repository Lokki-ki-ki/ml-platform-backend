from .app import create_app
from asgiref.wsgi import WsgiToAsgi

app = create_app()
asgi_app = WsgiToAsgi(app)


# if __name__ == "__main__":
#     app = create_app()
#     app.run(debug=True, host='0.0.0.0', port='8080', threaded=True)