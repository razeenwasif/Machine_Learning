# Dockerfile for the Record Linkage API backend service
FROM rapidsai/base:25.08-cuda12.9-py3.11

WORKDIR /workspace

# Copy the conda environment file and update the environment
COPY ml-rl-cuda12.yml ml-rl-cuda12.yml

# Update the environment with mamba
RUN mamba env update -n base -f ml-rl-cuda12.yml && \
    conda clean -afy && \
    rm -rf /opt/conda/pkgs/*

# Copy the entire project
COPY . .

# Expose port and run the API
EXPOSE 8000
CMD ["uvicorn", "recordLinkage.src.api:app", "--host", "0.0.0.0", "--port", "8000"]
