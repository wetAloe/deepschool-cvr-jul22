FROM ubuntu:20.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    python3-pip \
    make \
    wget \
    ffmpeg \
    libsm6 \
    libxext6


WORKDIR /genres_service

COPY . /genres_service/
RUN make install

RUN make download_weights

EXPOSE 5000

CMD make run_app

