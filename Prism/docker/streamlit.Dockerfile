FROM rapidsai/base:25.08-cuda12.9-py3.11

SHELL ["/bin/bash", "-c"]

RUN mamba install --yes \
        -c pytorch \
        -c rapidsai \
        -c conda-forge \
        -c nvidia \
        faiss-gpu=1.12.0 \
    && python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir \
        streamlit==1.34.0 \
        plotly==5.20.0 \
        pydeck==0.8.1b0 \
        optuna==3.6.1 \
        rich==13.7.1 \
    && python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu124 \
        --extra-index-url https://pypi.org/simple \
        torch==2.5.1 \
    && conda clean -afy \
    && rm -rf ~/.cache/pip

RUN python - <<'PY'
from pathlib import Path
import shutil
import torch

lib_dir = Path(torch.__file__).resolve().parent / "lib"
remove_prefixes = (
    "libcublas",
    "libcublasLt",
    "libcusolver",
    "libcusolverMg",
)

if lib_dir.is_dir():
    for path in lib_dir.iterdir():
        if any(path.name.startswith(prefix) for prefix in remove_prefixes):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
PY

ENV PATH="/opt/conda/bin:${PATH}"

WORKDIR /workspace

COPY docker/streamlit-entrypoint.sh /usr/local/bin/streamlit-entrypoint.sh

ENTRYPOINT ["streamlit-entrypoint.sh"]
CMD ["run", "autoML/gui/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
