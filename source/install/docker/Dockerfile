FROM python:3.11 AS compile-image
ARG VARIANT=""
ARG CUDA_VERSION="12"
RUN python -m venv /opt/deepmd-kit
# Make sure we use the virtualenv
ENV PATH="/opt/deepmd-kit/bin:$PATH"
# Install package
COPY dist /dist
RUN pip install "$(ls /dist/deepmd_kit${VARIANT}-*manylinux*_x86_64.whl)[gpu,cu${CUDA_VERSION},lmp,ipi]" \
    && dp -h \
    && lmp -h \
    && dp_ipi \
    && python -m deepmd -h

FROM python:3.11 AS build-image
COPY --from=compile-image /opt/deepmd-kit /opt/deepmd-kit
ENV PATH="/opt/deepmd-kit/bin:$PATH"
CMD ["/bin/bash"]
