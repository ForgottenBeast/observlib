import threading
from flask import Flask
from prometheus_client import make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple



def start_server(parameters):
    [hostname, port] = parameters.split(":")

    flask_app = Flask(__name__)

    application = DispatcherMiddleware(flask_app, {
        "/metrics": make_wsgi_app()
    })

    def run():
       run_simple(hostname=host, port=int(port), application=application, use_reloader=False)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
