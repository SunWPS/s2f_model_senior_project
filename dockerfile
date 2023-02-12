FROM python:3.9.16-slim-buster
RUN pip install --upgrade pip
RUN apt-get update && apt-get install -y --no-install-recommends \
      bzip2 \
      g++ \
      git \
      graphviz \
      libgl1-mesa-glx \
      libhdf5-dev \
      openmpi-bin \
      wget \
      python3-tk && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN mkdir /model
RUN mkdir /api
COPY requirements.txt /app
COPY model/ /app/model
COPY api/ /app/api
RUN pip install -r requirements.txt
RUN pip install --upgrade protobuf==3.20.0
CMD ["python", "api/server.py"]

