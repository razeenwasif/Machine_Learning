FROM condaforge/miniforge3:24.9.0-0

WORKDIR /opt/prism

COPY ml-rl-cuda12.yml .

RUN conda env create -f ml-rl-cuda12.yml \
    && conda clean -afy

ENV PATH="/opt/conda/envs/ml-rl-cuda12/bin:${PATH}"

WORKDIR /workspace

COPY docker/recordlinkage-entrypoint.sh /usr/local/bin/recordlinkage-entrypoint.sh

ENTRYPOINT ["recordlinkage-entrypoint.sh"]
CMD ["python", "-m", "src.main", "link"]
