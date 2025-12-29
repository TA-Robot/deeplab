# Cursor Dev Container用 Dockerfile
FROM ubuntu:22.04

# 環境変数設定
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    SHELL=/bin/bash

# 基本パッケージのインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-transport-https \
    build-essential \
    ca-certificates \
    curl \
    git \
    gnupg \
    jq \
    less \
    locales \
    openssh-client \
    procps \
    python3 \
    python3-pip \
    python3-venv \
    ripgrep \
    sudo \
    unzip \
    vim \
    wget \
    xz-utils \
    zsh \
    && rm -rf /var/lib/apt/lists/*

# ロケール設定
RUN locale-gen en_US.UTF-8

# Node.js 22.x インストール
ENV NODE_VERSION=22.19.0
RUN curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz" -o /tmp/node.tar.xz \
    && tar -xJf /tmp/node.tar.xz -C /usr/local --strip-components=1 \
    && rm /tmp/node.tar.xz \
    && ln -sf /usr/local/bin/node /usr/bin/node \
    && ln -sf /usr/local/bin/npm /usr/bin/npm \
    && npm install -g npm@latest \
    && npm cache clean --force

# 開発用グローバルパッケージ（必要に応じて追加）
RUN npm install -g \
    typescript \
    ts-node \
    eslint \
    prettier \
    @openai/codex \
    @google/gemini-cli \
    && npm cache clean --force

# Codex セカンドエージェント用ラッパー（セッションID自動保持）
COPY scripts/codex-second-agent /usr/local/bin/codex-second-agent
COPY scripts/codex-second-agent-filter.py /usr/local/lib/codex-second-agent/filter.py
RUN chmod +x /usr/local/bin/codex-second-agent

# 開発ユーザー作成
ARG USERNAME=devuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /bin/bash \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Oh My Zsh のインストール（オプション - 快適なシェル環境）
USER $USERNAME
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended || true

# Codex/Gemini用エイリアス設定
RUN echo '# Codex CLI aliases' >> ~/.bashrc \
    && echo 'alias codex-auto="codex --dangerously-bypass-approvals-and-sandbox"' >> ~/.bashrc \
    && echo 'alias codex-full="codex --full-auto"' >> ~/.bashrc

# 作業ディレクトリ設定
WORKDIR /workspace

# デフォルトユーザー
USER $USERNAME

# デフォルトシェル
CMD ["bash"]

