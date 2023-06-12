FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04
RUN apt-get update && apt-get install --no-install-recommends -y software-properties-common gcc && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install --no-install-recommends -y python3-venv python3-dev python3-distutils python3-pip build-essential git
RUN pip3 install --no-cache-dir cmake wheel
COPY NeSVoR /usr/local/NeSVoR
COPY install.sh /usr/local/NeSVoR/
WORKDIR /usr/local/NeSVoR

