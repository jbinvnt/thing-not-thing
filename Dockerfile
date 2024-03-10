FROM nvidia/cuda:11.7.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update && apt-get -y install python3-pip git && apt-get clean

RUN pip3 install 'git+https://github.com/mlfoundations/open_clip.git'

WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY app.py .