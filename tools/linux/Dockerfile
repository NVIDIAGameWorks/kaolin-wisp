ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# used for cross-compilation in docker build
ARG FORCE_CUDA=1
ARG INSTALL_RENDERER

WORKDIR /wisp

COPY . .

RUN apt-get -y update \
    && apt-get -y upgrade \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libopenexr-dev \
        git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt
RUN pip install pyexr
RUN python setup.py develop

RUN if [ -z "${INSTALL_RENDERER}" ]; then \
        pip install -r requirements_app.txt; \
    fi
