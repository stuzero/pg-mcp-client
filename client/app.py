# client/app.py
import uvicorn
from contextlib import asynccontextmanager

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware import Middleware
from starlette.templating import Jinja2Templates

from client.config import logger
from client.routes.home import homepage
from client.routes.settings import settings_page, settings_update
from client.middleware import ConfigSessionMiddleware

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
]

middleware = [
    Middleware(SessionMiddleware, secret_key="your-secret-key-here", max_age=3600*24*30),  # 30 days session
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