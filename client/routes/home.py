# client/routes/home.py
from starlette.requests import Request

async def homepage(request):
    """Render the home page."""
    # Check if all required settings are configured
    is_configured = all([
        request.session.get('LLM_API_KEY'),
        request.session.get('LLM'),
        request.session.get('PG_MCP_SERVER_URL'),
        request.session.get('DATABASE_URL')
    ])
    
    # Use templates from app.state
    return request.app.state.templates.TemplateResponse(
        'index.html.jinja2', 
        {
            'request': request,
            'is_configured': is_configured
        }
    )