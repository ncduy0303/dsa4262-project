FROM mambaorg/micromamba:2.9.0-debian12-slim@sha256:edce84d758b6c3029cebdf2777d9cdfdd6e44f9eee0a31c6a09a7df14be17fd2
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && micromamba clean --all --yes
ENV OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
WORKDIR /work
