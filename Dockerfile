FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

ADD https://astral.sh/uv/install.sh /uv-installer.sh

RUN sh /uv-installer.sh && rm /uv-installer.sh

ENV PATH="/root/.local/bin/:$PATH"

ADD . /app

WORKDIR /app
# Install dependencies using uv
RUN uv sync

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application
CMD ["uv", "run", "-m", "client.app"]