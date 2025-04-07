# client/routes/home.py
from starlette.responses import RedirectResponse

async def homepage(request):
   return RedirectResponse(url='/query')