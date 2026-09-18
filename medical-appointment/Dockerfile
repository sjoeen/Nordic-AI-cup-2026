FROM python:3.11-slim

WORKDIR /app

# Serving + ML stack. CPU torch by default; set TORCH_INDEX for a CUDA host.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir --extra-index-url ${TORCH_INDEX} -r requirements-ml.txt

# The server, the stage modules and the serving assets. The sample data under
# data/ is not copied: only local_evaluator.py needs it.
COPY dtos.py utils.py example.py api.py core_types.py audio_io.py transcript.py \
     asr_backend.py question_rewrite.py factcheck.py qa_backend.py qa_llm_backend.py \
     evidence.py response_checks.py calibration.py pipeline.py ./
COPY configs/ ./configs/
COPY prompts/ ./prompts/

# Model weights are downloaded at build time so the container serves offline.
ENV HF_HOME=/app/hf-cache
COPY prefetch_models.py ./
RUN python prefetch_models.py configs/default.json
ENV HF_HUB_OFFLINE=1

EXPOSE 9054

CMD ["python", "api.py"]
