#!/bin/bash

# Import container env vars (not inherited when run via nohup over SSH)
if [ -f /proc/1/environ ]; then
    export $(cat /proc/1/environ | tr '\0' '\n' | grep -E '^(HF_TOKEN|GH_TOKEN)=')
fi

# Clone repo if not present
if [ ! -d "/workspace/Preferences" ]; then
    git clone https://github.com/ogilg/Preferences.git /workspace/Preferences
fi
cd /workspace/Preferences

# Claude Code
curl -fsSL https://claude.ai/install.sh | bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"

# Python environment (outside workspace for faster installs)
pip install uv
uv venv --python 3.12 /root/.venvs/preferences
source /root/.venvs/preferences/bin/activate
echo 'source /root/.venvs/preferences/bin/activate' >> ~/.bashrc
uv pip install -e .
uv pip install -e ".[dev]"
uv pip install -e ".[viz]"

# Git config
git config --global user.name "Oscar Gilg"
git config --global user.email "oscar.gilg18@gmail.com"

# System tools
apt-get update && apt-get install -y tmux
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null && apt update && apt install gh -y

# Auth (uses tokens passed as pod env vars)
if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token "$HF_TOKEN"
    echo "Logged into Hugging Face."
else
    echo "WARNING: HF_TOKEN not set. Run: huggingface-cli login"
fi

if [ -n "$GH_TOKEN" ]; then
    echo "$GH_TOKEN" | gh auth login --with-token
    echo "Logged into GitHub."
else
    echo "WARNING: GH_TOKEN not set. Run: gh auth login"
fi

echo ""
echo "=== Setup complete ==="
