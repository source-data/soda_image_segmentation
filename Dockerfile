FROM nvcr.io/nvidia/pytorch:23.01-py3 AS compile-image

WORKDIR /app

# Set DEBIAN_FRONTEND to noninteractive to avoid tzdata prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for OpenCV and Python 3.9
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    openssh-server \
    poppler-utils \
    software-properties-common \
    ghostscript \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.9 \
    python3.9-distutils \
    python3.9-venv \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Set up SSH server
RUN mkdir /var/run/sshd
RUN echo 'root:root' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/UsePAM yes/UsePAM no/' /etc/ssh/sshd_config

# Update alternatives to set python3.9 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Install pip for Python 3.9
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9

# Copy and install Python dependencies
COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install opencv-python==4.8.0.74
RUN pip install -q git+https://github.com/THU-MIG/yolov10.git
RUN pip install opencv-fixer==0.2.5
RUN pip install jupyterlab
RUN python -c "from opencv_fixer import AutoFix; AutoFix()"

COPY . .

EXPOSE 9240
