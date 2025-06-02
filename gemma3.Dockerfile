FROM nvcr.io/nvidia/pytorch:25.03-py3
WORKDIR /workspace
COPY . .
RUN python -m venv --system-site-packages polar \
    && polar/bin/pip install -U pip \
    && polar/bin/pip install --no-deps -r requirements-gemma3-extra.txt

ENV PATH="/workspace/polar/bin:${PATH}"
ENV VIRTUAL_ENV=/workspace/polar

ENTRYPOINT ["/bin/bash"]