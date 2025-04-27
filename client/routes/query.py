# client/routes/query.py
import json
from starlette.requests import Request
from starlette.responses import JSONResponse, HTMLResponse

from client.config import logger
from client.services.agent import AgentService

async def query_page(request):
    """Render the query page."""
    # Check if user has configured the required settings
    is_configured = all([
        request.session.get('LLM_API_KEY'),
        request.session.get('LLM'),
        request.session.get('PG_MCP_SERVER_URL'),
        request.session.get('DATABASE_URL')
    ])
    
    context = {
        'request': request,
        'is_configured': is_configured,
        'error_message': None if is_configured else "Please configure settings before using the query interface."
    }
    
    return request.app.state.templates.TemplateResponse('query.html.jinja2', context)

async def view_query_results(request):
    """Handle HTMX requests to view different tabs (table, sql, visualization)."""
    # Get the requested view type from the query parameters
    view_type = request.query_params.get('view', 'table')
    include_tabs = request.query_params.get('include_tabs', 'false').lower() == 'true'
    
    # Get results from session
    sql = request.session.get('current_sql', '')
    results = request.session.get('current_results', [])
    grid_id = request.session.get('current_grid_id', f"results-grid")
    
    # Get visualization data if available
    visualization_spec = request.session.get('current_visualization_spec', None)
    chart_explanation = request.session.get('current_chart_explanation', None)
    
    # Create headers from the first result if available
    headers = list(results[0].keys()) if results and len(results) > 0 else []
    
    context = {
        'request': request,
        'active_tab': view_type,
        'sql': sql,
        'results': results,
        'headers': headers,
        'grid_id': grid_id,
        'visualization_spec': visualization_spec,
        'chart_explanation': chart_explanation
    }
    
    # If include_tabs is true, return the entire results area with tabs
    if include_tabs:
        template = 'query/query_results.html.jinja2'
    else:
        # Otherwise, return just the tab content
        if view_type == 'sql':
            template = 'query/generated_sql.html.jinja2'
        elif view_type == 'visualization':
            template = 'query/results_visualization.html.jinja2'
        else:  # Default to table view
            template = 'query/results_table.html.jinja2'
    
    return request.app.state.templates.TemplateResponse(template, context)

async def execute_query(request):
    """Handle the query submission and return results via HTMX."""
    # Check if all settings are configured
    if not all([
        request.session.get('LLM_API_KEY'),
        request.session.get('LLM'),
        request.session.get('PG_MCP_SERVER_URL'),
        request.session.get('DATABASE_URL')
    ]):
        return HTMLResponse(
            """<div class="p-4 bg-red-100 text-red-700 rounded-lg">
                Please configure all required settings before using the query interface.
                <a href="/settings" class="underline">Go to settings</a>
            </div>"""
        )
    
    try:
        # Parse form data
        form_data = await request.form()
        user_query = form_data.get('query', '').strip()
        
        if not user_query:
            return HTMLResponse(
                """<div class="p-4 bg-yellow-100 text-yellow-700 rounded-lg">
                    Please enter a query.
                </div>"""
            )
            
        # Initialize agent service
        agent = await AgentService.from_request(request)
        
        try:
            # Process the query
            logger.info(f"Processing query: {user_query}")
            result = await agent.process_query(user_query)
            
            if not result['success']:
                return HTMLResponse(
                    f"""<div class="p-4 bg-red-100 text-red-700 rounded-lg">
                        Error: {result.get('error', 'An unknown error occurred.')}
                    </div>"""
                )
            
            # Extract data from result
            sql = result.get('sql', '')
            results = result.get('results', [])
            grid_id = f"results-grid-{hash(user_query) % 10000}"
            
            # Store in session for tab views
            request.session['current_sql'] = sql
            request.session['current_results'] = results
            request.session['current_grid_id'] = grid_id
            
            # Create headers from the first result if available
            headers = list(results[0].keys()) if results and len(results) > 0 else []
            
            # Clear any previous visualization data
            # In a future version, this is where you would generate the visualization
            request.session['current_visualization_spec'] = None
            request.session['current_chart_explanation'] = None
            
            # Render the results template with tabs
            context = {
                'request': request,
                'active_tab': 'table',  # Default to table view initially
                'results': results,
                'headers': headers,
                'grid_id': grid_id,
                'sql': sql,
                'visualization_spec': None,
                'chart_explanation': None
            }
            
            # Return just the query results panel with tabs
            return request.app.state.templates.TemplateResponse(
                'query/query_results.html.jinja2', 
                context
            )
            
        finally:
            # Clean up resources
            await agent.close()
            
    except Exception as e:
        logger.error(f"Error handling query: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        return HTMLResponse(
            f"""<div class="p-4 bg-red-100 text-red-700 rounded-lg">
                An error occurred while processing your query: {str(e)}
            </div>"""
        )