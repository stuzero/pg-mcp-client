# client/middleware.py
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

class ConfigSessionMiddleware(BaseHTTPMiddleware):
    """
    Middleware to store configuration values in the session.
    This middleware must be added after the SessionMiddleware.
    """
    
    async def dispatch(
        self, request: Request, call_next
    ) -> Response:
        # Initialize session keys if they don't exist
        if 'LLM_API_KEY' not in request.session:
            request.session['LLM_API_KEY'] = ''
        
        if 'LLM' not in request.session:
            request.session['LLM'] = 'Gemini'
        
        if 'PG_MCP_SERVER_URL' not in request.session:
            request.session['PG_MCP_SERVER_URL'] = ''
        
        if 'DATABASE_URL' not in request.session:
            request.session['DATABASE_URL'] = ''
            
        # Call the next middleware or route handler
        response = await call_next(request)
        
        return response