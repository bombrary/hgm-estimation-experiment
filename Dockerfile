FROM bombrary/openxm-debian:latest

RUN apt-get update && apt-get install -y python3 python3-pip
COPY ox_asir_client/ /ox_asir_client
COPY hgm-system-estimation/ /hgm-system-estimation

RUN pip install /ox_asir_client && \
    pip install /hgm-system-estimation && \
    pip install ipython pytest

WORKDIR /home
