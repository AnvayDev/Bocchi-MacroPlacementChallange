# CPU-only DREAMPlace + macro-place-challenge
# Works on both amd64 and arm64 (Apple Silicon via Docker Desktop)
#
# Build:
#   docker build -t macro-place .
#
# Run a benchmark (mounts external/ so the ICCAD04 benchmarks are visible):
#   docker run --rm \
#     -v "$(pwd)/external:/workspace/external" \
#     -v "$(pwd)/submissions:/workspace/submissions" \
#     macro-place \
#     python submissions/dreamplace_placer.py
#
# Interactive shell:
#   docker run --rm -it \
#     -v "$(pwd)/external:/workspace/external" \
#     macro-place bash

FROM python:3.9-slim-bookworm

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        flex \
        bison \
        libboost-all-dev \
        zlib1g-dev \
        libgomp1 \
        git \
        wget \
    && rm -rf /var/lib/apt/lists/*

# ── PyTorch CPU-only (matches DREAMPlace's supported range) ───────────────────
RUN pip install --no-cache-dir \
        torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu

# ── Python deps for macro_place package ───────────────────────────────────────
RUN pip install --no-cache-dir \
        numpy>=1.20.0 \
        matplotlib>=3.5.0 \
        tqdm>=4.65.0 \
        absl-py>=1.0.0

WORKDIR /workspace

# ── Copy DREAMPlace source and apply compatibility patches ────────────────────
COPY DREAMPlace/ /workspace/DREAMPlace/

# Fix 1: lemon uses deprecated CMake policy CMP0048 OLD (not allowed in CMake 3.24+)
RUN sed -i 's/CMAKE_POLICY(SET CMP0048 OLD)/CMAKE_POLICY(SET CMP0048 NEW)/' \
        /workspace/DREAMPlace/thirdparty/Limbo/limbo/thirdparty/lemon/CMakeLists.txt

# Fix 2: thread_hold must accept int argument for POSIX signal handler
RUN sed -i \
        -e 's/static void  thread_hold();/static void  thread_hold(int sig_id);/' \
        -e 's/static void thread_hold () {/static void thread_hold (int sig_id) {/' \
        /workspace/DREAMPlace/thirdparty/Limbo/limbo/thirdparty/CThreadPool/thpool.c

# ── Build DREAMPlace (CPU-only, no CUDA) ──────────────────────────────────────
RUN mkdir -p /workspace/DREAMPlace/build && \
    cd /workspace/DREAMPlace/build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/workspace/DREAMPlace/install \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    && make -j$(nproc) \
    && make install

ENV DREAMPLACE_HOME=/workspace/DREAMPlace/install

# ── Install the macro_place package ───────────────────────────────────────────
COPY macro_place/       /workspace/macro_place/
COPY pyproject.toml     /workspace/pyproject.toml
RUN pip install --no-cache-dir -e /workspace

# ── Copy the rest of the project (submissions, benchmarks, scripts, etc.) ─────
# external/ and benchmarks/ are intentionally excluded — mount them at runtime
# so you don't bake large benchmark files into the image.
COPY submissions/   /workspace/submissions/
COPY scripts/       /workspace/scripts/
COPY test/          /workspace/test/
COPY benchmarks/    /workspace/benchmarks/

# Default: print available commands
CMD ["python", "-c", "print('DREAMPlace ready. Run: docker run ... python submissions/dreamplace_placer.py')"]