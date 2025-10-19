FROM rapidsai/base:25.08-cuda12.9-py3.11

WORKDIR /opt/prism

RUN apt-get update && \
    apt-get install -y --no-install-recommends gosu && \
    rm -rf /var/lib/apt/lists/*

COPY docker/conda-streamlit.yml .

RUN mamba env create -f conda-streamlit.yml \
    && conda clean -afy

ENV PATH="/opt/conda/envs/ml-rl-cuda12/bin:/opt/conda/bin:${PATH}"

WORKDIR /workspace

COPY docker/streamlit-entrypoint.sh /usr/local/bin/streamlit-entrypoint.sh

ENTRYPOINT ["streamlit-entrypoint.sh"]
CMD ["streamlit", "run", "src/gui/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
