# Use an official Python runtime as a base image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install system dependencies for Python and Ollama
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    sudo \
    gnupg && \
    rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Add Ollama to PATH
ENV PATH="/usr/local/bin:${PATH}"

RUN ollama serve & sleep 3 && \
    ollama pull llama3.2 && \
    ollama pull openhermes

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application into the container
COPY . .

# Expose the Streamlit default port
EXPOSE 8501

# Command to run the Streamlit app
CMD ollama serve & sleep 3 && streamlit run local_rag_agent.py