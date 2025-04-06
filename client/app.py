# client/app.py
import uvicorn
from contextlib import asynccontextmanager

from starlette.applications import Starlette
from starlette.routing import Route

from client.config import logger
from client.routes.home import homepage
from client.routes.settings import settings_page, settings_update

@asynccontextmanager
async def lifespan(app):
    # Initialize application state
    app.state.GEMINI_API_KEY = ''
    app.state.DATABASE_URL = ''
    
    logger.info("Application started")
    yield
    logger.info("Application shutdown")

routes = [
    Route('/', endpoint=homepage),
    Route('/settings', endpoint=settings_page, methods=['GET']),
    Route('/settings', endpoint=settings_update, methods=['POST']),
]

app = Starlette(
    debug=True,
    routes=routes,
    lifespan=lifespan
)

if __name__ == '__main__':
    uvicorn.run(
        app,
        host='0.0.0.0',
        port=8080
    )