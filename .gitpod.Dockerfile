# Use the official Gitpod base image
FROM gitpod/workspace-full:latest

# Install system dependencies
USER root

# Install Tesseract OCR and other dependencies
RUN apt-get update && \
    apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    python3-pip \
    python3-dev \
    && apt-get clean

# Set up Python environment and install packages
RUN pip3 install --upgrade pip setuptools wheel \
    && pip3 install \
    opencv-python \
    pytesseract \
    torch \
    transformers \
    markdown \
    pyspellchecker

# Set the working directory
WORKDIR /workspace
