# client/routes/settings.py
from starlette.templating import Jinja2Templates
from starlette.responses import HTMLResponse

from ..config import logger

# Initialize templates
templates = Jinja2Templates(directory='client/templates')

async def settings_page(request):
    """Display the settings page with current values if available."""
    context = {
        'request': request,
        'gemini_api_key': getattr(request.app.state, 'GEMINI_API_KEY', ''),
        'pg-mcp-server': getattr(request.app.state, 'PG_MCP_SERVER_URL', ''),
        'database_url': getattr(request.app.state, 'DATABASE_URL', ''),
    }
    return templates.TemplateResponse('settings.html.jinja2', context)

async def settings_update(request):
    """Update the settings based on form submission."""
    form_data = await request.form()
    
    # Update application state
    request.app.state.GEMINI_API_KEY = form_data.get('gemini_api_key', '')
    request.app.state.PG_MCP_SERVER_URL = form_data.get('pg-mcp-server', '')
    request.app.state.DATABASE_URL = form_data.get('database_url', '')
    
    logger.info("Settings updated successfully")
    
    # Render settings page with success message
    context = {
        'request': request,
        'gemini_api_key': request.app.state.GEMINI_API_KEY,
        'pg-mcp-server' : request.app.state.PG_MCP_SERVER_URL,
        'database_url': request.app.state.DATABASE_URL,
        'message': 'Settings updated successfully!',
        'success': True
    }
    return templates.TemplateResponse('settings.html.jinja2', context)