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
            
            # Format results using Grid.js
            if results:
                # Get column headers from first result
                headers = list(results[0].keys())
                
                # Create a unique ID for this Grid.js instance
                grid_id = f"results-grid-{hash(user_query) % 10000}"
                
                # Convert results to JSON for Grid.js
                import json
                results_json = json.dumps(results)
                
                # Create Grid.js initialization
                gridjs_init = f"""
                <div id="{grid_id}" class="mt-4 w-full"></div>
                <script>
                    new gridjs.Grid({{
                        columns: {json.dumps(headers)},
                        data: {results_json},
                        sort: true,
                        pagination: {{
                            limit: 10
                        }},
                        search: true,
                        className: {{
                            table: 'w-full border-collapse'
                        }}
                    }}).render(document.getElementById("{grid_id}"));
                </script>
                <div class="mt-2 text-sm text-gray-500">
                    Total rows: {len(results)}
                </div>
                """
                
                # Add download button for CSV export
                download_script = f"""
                <div class="mt-3 flex justify-end">
                    <button 
                        id="download-csv" 
                        class="bg-green-500 text-white px-3 py-1.5 rounded hover:bg-green-600 text-sm flex items-center"
                        onclick="downloadCSV()"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                        </svg>
                        Export CSV
                    </button>
                </div>
                <script>
                    function downloadCSV() {{
                        const results = {results_json};
                        if (!results.length) return;
                        
                        const headers = Object.keys(results[0]);
                        let csvContent = headers.join(',') + '\\n';
                        
                        results.forEach(row => {{
                            const values = headers.map(header => {{
                                const value = row[header];
                                // Handle null values and escape quotes
                                const formattedValue = value === null ? '' : String(value).replace(/"/g, '""');
                                // Wrap in quotes to handle commas and special characters
                                return `"${{formattedValue}}"`;
                            }});
                            csvContent += values.join(',') + '\\n';
                        }});
                        
                        const blob = new Blob([csvContent], {{ type: 'text/csv;charset=utf-8;' }});
                        const url = URL.createObjectURL(blob);
                        const link = document.createElement('a');
                        link.setAttribute('href', url);
                        link.setAttribute('download', 'query_results.csv');
                        link.style.visibility = 'hidden';
                        document.body.appendChild(link);
                        link.click();
                        document.body.removeChild(link);
                    }}
                </script>
                """

                # Create results section with Grid.js and download button
                results_section = f"""
                <div class="mt-6">
                    <h3 class="text-lg font-medium text-gray-900">Results</h3>
                    {gridjs_init}
                    {download_script}
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
            
            # Create the complete response with results first, then SQL
            # Note the 'language-sql' class for highlight.js
            response_html = f"""
            <div class="space-y-6">
                {results_section}
                
                <div class="mt-6">
                    <h3 class="text-lg font-medium text-gray-900">Generated SQL</h3>
                    <div class="mt-2 p-4 bg-gray-800 rounded-lg overflow-x-auto">
                        <pre><code class="language-sql">{sql}</code></pre>
                    </div>
                </div>
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