FROM nvcr.io/nvidia/pytorch:23.09-py3

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install imageio-ffmpeg==0.4.3 pyspng==0.1.1

WORKDIR /workspace

# ===== Add adaptive user =====
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} appgroup && \
    useradd -m -u ${USER_ID} -g appgroup appuser

# ===== Create entry.sh AS ROOT =====
RUN printf '#!/bin/bash\nunset TORCH_CUDA_ARCH_LIST\nexec "$@"\n' > /workspace/entry.sh && chmod a+x /workspace/entry.sh

# ===== Switch to non-root user =====
USER appuser

# Set entrypoint
#ENTRYPOINT ["/workspace/entry.sh"]
CMD ["bash"]
