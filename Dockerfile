FROM python:3.10-slim

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /workspace

# Install basic tools
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy only requirements first (better caching)
COPY ../requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir --default-timeout=100 -r /tmp/requirements.txt

# Default shell
CMD ["bash"]