FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel

# 地域設定
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# NVIDIAのGPGキーを追加
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

# システムの更新とビルドに必要なパッケージのインストール
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ninja-build \
    libopencv-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# CUDA環境変数の設定
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="3.7;5.0;6.0;7.0+PTX"

# 作業ディレクトリの作成
WORKDIR /workspace

# PyTorchとtorchvisionのインストール（特別なURLから）
RUN pip install torchvision==0.10.1+cu111 torchaudio==0.9.1 \
    --extra-index-url https://download.pytorch.org/whl/cu111

# requirements.txtをコピーし、PyTorch関連を除外した他のパッケージをインストール
COPY requirements.txt /workspace/
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu111 \
    --no-deps \
    numpy==1.19.5 \
    mmcv-full==1.4.0 \
    mmdet==2.14.0 \
    mmsegmentation==0.14.1 \
    scikit-image==0.19.3 \
    tensorboard==2.9.0 \
    timm==0.6.12 \
    matplotlib==3.5.3 \
    opencv-python==4.7.0.72 \
    scipy==1.7.3 \
    nuscenes-devkit==1.1.9 \
    pyquaternion==0.9.9 \
    shapely==1.8.5 \
    pyparsing==3.0.9 \
    networkx==2.2 \
    trimesh==2.35.39 \
    prettytable==3.7.0 \
    terminaltables==3.1.10 \
    yapf==0.43.0 \
    addict==2.4.0 \
    similaritymeasures==1.2.0

# mmdetection3dのインストール
RUN git clone https://github.com/open-mmlab/mmdetection3d.git \
    && cd mmdetection3d \
    && git checkout -f v0.17.1 \
    && pip install -v -e .

# VADのクローンと事前学習モデルのダウンロード
RUN git clone https://github.com/hustvl/VAD.git \
    && cd VAD \
    && mkdir -p ckpts \
    && cd ckpts \
    && wget https://download.pytorch.org/models/resnet50-19c8e357.pth

# 作業ディレクトリをVADに設定
WORKDIR /workspace/VAD