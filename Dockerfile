ARG VLLM_IMAGE=vllm/vllm-openai:v0.10.2
FROM ${VLLM_IMAGE}
WORKDIR /suite
COPY bench.py /suite/bench.py
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["python3", "/suite/bench.py"]
