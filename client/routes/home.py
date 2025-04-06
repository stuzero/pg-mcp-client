# client/routes/home.py
from starlette.templating import Jinja2Templates

# Initialize templates
templates = Jinja2Templates(directory='client/templates')

async def homepage(request):
    """Render the home page."""
    return templates.TemplateResponse('index.html.jinja2', {'request': request})