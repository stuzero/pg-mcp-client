# client/app.py
import uvicorn
import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware import Middleware
from starlette.templating import Jinja2Templates

from client.config import logger
from client.routes.home import homepage
from client.routes.settings import settings_page, settings_update
from client.routes.query import query_page, execute_query

from client.middleware import ConfigSessionMiddleware

load_dotenv()
application_secret = os.getenv('APPLICATION_SECRET')
# Initialize templates at application level
templates = Jinja2Templates(directory='client/templates')

@asynccontextmanager
async def lifespan(app):
    # Make templates available in app state
    app.state.templates = templates
    
    logger.info("Application started")
    yield
    logger.info("Application shutdown")

routes = [
    Route('/', endpoint=homepage),
    Route('/settings', endpoint=settings_page, methods=['GET']),
    Route('/settings', endpoint=settings_update, methods=['POST']),
    Route('/query', endpoint=query_page, methods=["GET"]),
    Route('/query/execute', endpoint=execute_query, methods=["POST"]),
]

middleware = [
    Middleware(SessionMiddleware, secret_key= application_secret, max_age=3600*24*30),  # 30 days session
    Middleware(ConfigSessionMiddleware),
]

app = Starlette(
    debug=True,
    routes=routes,
    middleware=middleware,
    lifespan=lifespan
)

if __name__ == '__main__':
    uvicorn.run(
        app,
        host='0.0.0.0',
        port=8080
    )