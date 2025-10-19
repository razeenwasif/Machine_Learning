FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /opt/streamlit

COPY docker/requirements.streamlit.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.streamlit.txt

WORKDIR /workspace

CMD ["streamlit", "run", "src/gui/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
