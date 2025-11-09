ARG USERNAME=mackop
ARG USER_UID=1000
ARG USER_GID=1000

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda-11.8 \
    CONDA_ENV_NAME=edm \
    PYTHON_VERSION=3.10 \
    CONDA_DIR=/opt/conda

# --- System dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim wget git bzip2 build-essential libgl1 libglib2.0-0 libgl1-mesa-glx \
    libgomp1 curl imagemagick sudo libvips && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Install Miniconda ---
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# --- Custom requirements ---
COPY environment.yaml /tmp/environment.yaml

# --- Create conda env ---
RUN conda init bash && \
    conda config --set always_yes yes && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda env create -y -f /tmp/environment.yaml

ENV PATH=$CONDA_DIR/envs/${CONDA_ENV_NAME}/bin:$PATH
RUN echo "source activate ${CONDA_ENV_NAME}" >> ~/.bashrc


# --- User setup ---
ARG USERNAME
ARG USER_UID
ARG USER_GID

RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /bin/bash && \
    usermod -aG sudo $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER $USERNAME
WORKDIR /home/$USERNAME/workspace

# --- Auto activate env & colorful prompt ---
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate ${CONDA_ENV_NAME}" >> ~/.bashrc && \
    echo "alias ll='ls -alF --color=auto'" >> ~/.bashrc && \
    echo "alias la='ls -A --color=auto'" >> ~/.bashrc && \
    echo "alias l='ls -CF --color=auto'" >> ~/.bashrc && \
    echo "PS1='\[\e[1;32m\]\u@\h:\w \$\[\e[0m\] '" >> ~/.bashrc

# --- Verify environment ---
RUN conda run -n edm python -c "import sys; print('✅ Python in:', sys.executable)"
RUN conda info --envs

CMD ["conda", "run", "-n", "edm", "python", "homography_estimation.py"]
