# ---------- Base: CUDA 11.8 + cuDNN ----------
    FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

    ENV DEBIAN_FRONTEND=noninteractive
    
    # ---------- System deps + Python 3.8 ----------
    RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.8 python3.8-venv python3-pip \
        wget git ca-certificates curl \
        build-essential pkg-config \
        ffmpeg libsm6 libxext6 libxrender1 libgl1 libglib2.0-0 \
        libvips \
        && rm -rf /var/lib/apt/lists/*
    
    # ---------- Virtualenv ----------
    ENV VENV_PATH=/opt/venv
    RUN python3.8 -m venv $VENV_PATH
    ENV PATH="$VENV_PATH/bin:$PATH"
    
    # Upgrade pip & wheel
    RUN python -m pip install --upgrade pip wheel
    
    # ---------- PyTorch 2.0.0 (CUDA 11.8 wheels) ----------
    # Using official PyTorch wheel index for cu118
    RUN pip install \
        torch==2.0.0+cu118 torchvision==0.15.0+cu118 torchaudio==2.0.0+cu118 \
        --index-url https://download.pytorch.org/whl/cu118
    
    # ---------- Workdir & project ----------
    WORKDIR /workspace/edm
    
    # If you have a requirements.txt in the repo, copy it first for layer caching
    COPY requirements.txt /workspace/edm/requirements.txt
    RUN if [ -f requirements.txt ]; then \
          pip install -r requirements.txt ; \
        fi
    
    # Then copy the rest of your project
    COPY . /workspace/edm
    ARG USERNAME=mackop
    ARG USER_UID=1000
    ARG USER_GID=1000
    RUN groupadd --gid $USER_GID $USERNAME && \
        useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /bin/bash && \
        usermod -aG sudo $USERNAME && \
        echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

    USER $USERNAME


    RUN echo "alias ll='ls -alF --color=auto'" >> ~/.bashrc && \
        echo "alias la='ls -A --color=auto'" >> ~/.bashrc && \
        echo "alias l='ls -CF --color=auto'" >> ~/.bashrc && \
        echo "PS1='\[\e[1;32m\]\u@\h:\w \$\[\e[0m\] '" >> ~/.bashrc

    # (Optional) Jupyter convenience
    EXPOSE 8888
    
    CMD ["/bin/bash"]
    