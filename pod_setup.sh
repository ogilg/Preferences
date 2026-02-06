#!/bin/bash

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
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg && sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null && sudo apt update && sudo apt install gh -y

# Auth (interactive)
huggingface-cli login
gh auth login

# Copy over .env file manually
