#!/bin/bash

# Import container env vars (not inherited when run via nohup over SSH)
if [ -f /proc/1/environ ]; then
    export $(cat /proc/1/environ | tr '\0' '\n' | grep -E '^(HF_TOKEN|GH_TOKEN)=')
fi

# System tools (must run as root)
apt-get update && apt-get install -y tmux sudo
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null && apt update && apt install gh -y

# Create non-root user (--dangerously-skip-permissions doesn't work as root)
useradd -m -s /bin/bash coder
echo "coder ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/coder
chown -R coder:coder /workspace

# Everything below runs as coder
su - coder << SETUP

# Clone repo
if [ ! -d "/workspace/Preferences" ]; then
    git clone https://github.com/ogilg/Preferences.git /workspace/Preferences
fi

# Claude Code
curl -fsSL https://claude.ai/install.sh | bash
echo 'export PATH="\$HOME/.local/bin:\$PATH"' >> ~/.bashrc
export PATH="\$HOME/.local/bin:\$PATH"

# Python environment
pip install uv
uv venv --python 3.12 ~/.venvs/preferences
source ~/.venvs/preferences/bin/activate
echo 'source ~/.venvs/preferences/bin/activate' >> ~/.bashrc
cd /workspace/Preferences
uv pip install -e .
uv pip install -e ".[dev]"
uv pip install -e ".[viz]"

# Git config
git config --global user.name "Oscar Gilg"
git config --global user.email "oscar.gilg18@gmail.com"

# Auth
if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token $HF_TOKEN
    echo "Logged into Hugging Face."
fi

if [ -n "$GH_TOKEN" ]; then
    echo $GH_TOKEN | gh auth login --with-token
    echo "Logged into GitHub."
fi

SETUP

echo ""
echo "=== Setup complete ==="
echo "Run: su - coder"
echo "Then: cd /workspace/Preferences && claude --dangerously-skip-permissions"
