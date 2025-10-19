FROM rapidsai/base:25.08-cuda12.9-py3.11

WORKDIR /opt/prism

RUN apt-get update && \
    apt-get install -y --no-install-recommends gosu && \
    rm -rf /var/lib/apt/lists/*

COPY docker/conda-recordlinkage.yml .

RUN mamba env create -f conda-recordlinkage.yml \
    && conda clean -afy

ENV PATH="/opt/conda/envs/ml-rl-cuda12/bin:/opt/conda/bin:${PATH}"

WORKDIR /workspace

COPY docker/recordlinkage-entrypoint.sh /usr/local/bin/recordlinkage-entrypoint.sh

ENTRYPOINT ["recordlinkage-entrypoint.sh"]
CMD ["python", "-m", "src.main", "link"]
