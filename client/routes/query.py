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
            
            # Format SQL for display
            sql = result.get('sql', '')
            explanation = result.get('explanation', '')
            results = result.get('results', [])
            
            # Format results into an HTML table
            if results:
                # Get column headers from first result
                headers = list(results[0].keys())
                
                # Generate the HTML table
                table_html = """
                <div class="overflow-x-auto mt-4">
                    <table class="min-w-full divide-y divide-gray-200">
                        <thead class="bg-gray-100">
                            <tr>
                """
                
                # Add headers
                for header in headers:
                    table_html += f'<th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">{header}</th>'
                
                table_html += """
                            </tr>
                        </thead>
                        <tbody class="bg-white divide-y divide-gray-200">
                """
                
                # Add rows
                for row in results:
                    table_html += '<tr class="hover:bg-gray-50">'
                    for header in headers:
                        value = row.get(header, '')
                        # Format the value based on type
                        if value is None:
                            formatted_value = '<span class="text-gray-400">NULL</span>'
                        elif isinstance(value, (int, float)):
                            formatted_value = f'<span class="text-right">{value}</span>'
                        else:
                            formatted_value = str(value)
                        
                        table_html += f'<td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{formatted_value}</td>'
                    table_html += '</tr>'
                
                table_html += """
                        </tbody>
                    </table>
                </div>
                <div class="mt-2 text-sm text-gray-500">
                    Total rows: {row_count}
                </div>
                """.format(row_count=len(results))
                
                # Create results section
                results_section = f"""
                <div class="mt-6">
                    <h3 class="text-lg font-medium text-gray-900">Results</h3>
                    {table_html}
                </div>
                """
            else:
                results_section = """
                <div class="mt-6">
                    <h3 class="text-lg font-medium text-gray-900">Results</h3>
                    <div class="p-4 bg-gray-50 text-gray-700 rounded-lg">
                        Query executed successfully but returned no results.
                    </div>
                </div>
                """
            
            # Create the complete response
            response_html = f"""
            <div class="space-y-6">
                <div>
                    <h3 class="text-lg font-medium text-gray-900">Generated SQL</h3>
                    <div class="mt-2 p-4 bg-gray-800 text-white rounded-lg overflow-x-auto">
                        <pre><code>{sql}</code></pre>
                    </div>
                </div>
                
                {results_section}
            </div>
            """
            
            return HTMLResponse(response_html)
            
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