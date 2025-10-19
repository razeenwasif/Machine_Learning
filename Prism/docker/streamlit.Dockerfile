FROM rapidsai/base:25.08-cuda12.9-py3.11

WORKDIR /opt/prism

COPY ml-rl-cuda12.yml .

RUN mamba env create -f ml-rl-cuda12.yml \
    && conda clean -afy

ENV PATH="/opt/conda/envs/ml-rl-cuda12/bin:${PATH}"

WORKDIR /workspace

COPY docker/streamlit-entrypoint.sh /usr/local/bin/streamlit-entrypoint.sh

ENTRYPOINT ["streamlit-entrypoint.sh"]
CMD ["streamlit", "run", "src/gui/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
