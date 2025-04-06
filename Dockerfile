# Use a base Ray image matching the version you want
#FROM rayproject/ray-ml:2.9.3
FROM python:3.10-slim

# Set working directory (optional, but good practice)
WORKDIR /home/ray/marl_job

# Install PettingZoo and dependencies
# RUN pip install --no-cache-dir "pettingzoo[mpe]==1.23.1" pygame # Specify versions if needed
RUN pip install --no-cache-dir "pettingzoo[mpe]" pygame "ray[all]"

# Copy your script into the image
COPY marl_script.py .

# Optional: Set permissions if needed
# USER ray
# RUN chmod +x marl_script.py

# Default command can be empty, as RayJob specifies the entrypoint
CMD ["bash"]
