# 使用支持 CUDA 11.8 的 PyTorch 官方镜像作为基础镜像
FROM pytorch/pytorch:2.1.1-cuda11.8-cudnn8-runtime

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 复制 requirements.txt
COPY requirements.txt .

# 安装 Python 依赖
# 注意：DGL 和 PyG 的一些组件需要从特定的源安装以匹配 CUDA 版本
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 暴露端口（如果有需要，例如运行 notebook 或 API）
# EXPOSE 8888

# 默认启动命令（可以根据需要修改，例如运行 main_ali.py）
CMD ["python", "main_ali.py"]
