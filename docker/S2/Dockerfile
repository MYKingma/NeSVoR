FROM nesvor-s1:latest as builder
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install --no-install-recommends -y python3.10 python3-distutils && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/NeSVoR /usr/local/NeSVoR
COPY --from=builder /usr/local/bin /usr/local/bin
WORKDIR /usr/local/NeSVoR
