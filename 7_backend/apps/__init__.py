from flask import Flask, send_from_directory

def create_app():
    app = Flask(__name__,static_folder='static')
    from .routes import routes_list
    routes_list(app)
    return app
