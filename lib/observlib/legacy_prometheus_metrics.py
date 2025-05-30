import threading
from prometheus_client import make_wsgi_app
from fastapi import FastAPI
import uvicorn


def start_server(parameters):
    [host, port] = parameters.split(":")

    app = FastAPI()
    app.mount("/metrics", make_asgi_app())


    def run():
       uvicorn.run(app, host=host, port=int(port), log_level="info", access_log=False)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
