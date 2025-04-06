# client/routes/settings.py
from starlette.responses import HTMLResponse
from starlette.requests import Request

from ..config import logger

async def settings_page(request):
    """Display the settings page with current values if available."""
    context = {
        'request': request,
        'llm_api_key': request.session.get('LLM_API_KEY', ''),
        'llm': request.session.get('LLM', 'Gemini'),
        'pg_mcp_server': request.session.get('PG_MCP_SERVER_URL', ''),
        'database_url': request.session.get('DATABASE_URL', ''),
    }
    return request.app.state.templates.TemplateResponse('settings.html.jinja2', context)

async def settings_update(request):
    """Update the settings based on form submission."""
    form_data = await request.form()
    
    # Update session data
    request.session['LLM_API_KEY'] = form_data.get('llm_api_key', '')
    request.session['LLM'] = form_data.get('llm', 'Gemini')
    request.session['PG_MCP_SERVER_URL'] = form_data.get('pg-mcp-server', '')
    request.session['DATABASE_URL'] = form_data.get('database_url', '')
    
    logger.info("Settings updated successfully")
    
    # Render settings page with success message
    context = {
        'request': request,
        'llm_api_key': request.session['LLM_API_KEY'],
        'llm': request.session['LLM'],
        'pg_mcp_server': request.session['PG_MCP_SERVER_URL'],
        'database_url': request.session['DATABASE_URL'],
        'message': 'Settings updated successfully!',
        'success': True
    }
    return request.app.state.templates.TemplateResponse('settings.html.jinja2', context)