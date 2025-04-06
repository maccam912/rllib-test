# Use a base Ray image matching the version you want
FROM rayproject/ray:2.44.1
# FROM python:3.10-slim

# Set working directory (optional, but good practice)
WORKDIR /home/ray/marl_job

# Install PettingZoo and dependencies
# Since we're on CPUs, install cpu version of torch
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu

# Copy your script into the image
COPY marl_script.py .

# Optional: Set permissions if needed
# USER ray
# RUN chmod +x marl_script.py

# Default command can be empty, as RayJob specifies the entrypoint
CMD ["bash"]
