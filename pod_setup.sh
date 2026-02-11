#!/bin/bash
# Usage: bash pod_setup.sh [branch]

BRANCH="${1:-main}"

# Import container env vars (not inherited when run via nohup over SSH)
if [ -f /proc/1/environ ]; then
    export $(cat /proc/1/environ | tr '\0' '\n' | grep -E '^(HF_TOKEN|GH_TOKEN)=')
fi

# System tools (must run as root)
apt-get update && apt-get install -y tmux sudo jq
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null && apt update && apt install gh -y

# Create non-root user (--dangerously-skip-permissions doesn't work as root)
useradd -m -s /bin/bash coder
echo "coder ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/coder
chown -R coder:coder /workspace

# Write coder setup script (avoids heredoc variable expansion issues)
cat > /tmp/coder_setup.sh << 'CODER_SCRIPT'
#!/bin/bash

# Clone repo and checkout branch
if [ ! -d "/workspace/Preferences" ]; then
    git clone https://github.com/ogilg/Preferences.git /workspace/Preferences
fi
cd /workspace/Preferences
if [ -f /tmp/pod_branch ]; then
    git fetch origin
    git checkout "$(cat /tmp/pod_branch)"
fi

# Claude Code
curl -fsSL https://claude.ai/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"

# Python environment
pip install uv
uv venv --python 3.12 ~/.venvs/preferences
source ~/.venvs/preferences/bin/activate
cd /workspace/Preferences
uv pip install -e .
uv pip install -e ".[dev]"
uv pip install -e ".[viz]"

# Git config
git config --global user.name "Oscar Gilg"
git config --global user.email "oscar.gilg18@gmail.com"

# Auth (tokens passed via environment)
if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token $HF_TOKEN
    echo "Logged into Hugging Face."
fi

if [ -n "$GH_TOKEN" ]; then
    echo $GH_TOKEN | gh auth login --with-token
    echo "Logged into GitHub."
fi

# Write .bash_profile so login shells (su - coder -c '...') get venv + tokens
cat > ~/.bash_profile << 'PROFILE'
export PATH="$HOME/.local/bin:$PATH"
source ~/.venvs/preferences/bin/activate
PROFILE

if [ -n "$HF_TOKEN" ]; then
    echo "export HF_TOKEN=$HF_TOKEN" >> ~/.bash_profile
fi
if [ -n "$GH_TOKEN" ]; then
    echo "export GH_TOKEN=$GH_TOKEN" >> ~/.bash_profile
fi
CODER_SCRIPT

chmod +x /tmp/coder_setup.sh

# Pass branch to coder script via file
echo "$BRANCH" > /tmp/pod_branch

# Run as coder, forwarding tokens
su - coder -c "HF_TOKEN=$HF_TOKEN GH_TOKEN=$GH_TOKEN bash /tmp/coder_setup.sh"

# Claude Code auth: copy credentials from root to coder if present
if [ -f /root/.claude/.credentials.json ]; then
    mkdir -p /home/coder/.claude
    cp /root/.claude/.credentials.json /home/coder/.claude/.credentials.json
    chown -R coder:coder /home/coder/.claude
    chmod 600 /home/coder/.claude/.credentials.json
    echo "Claude Code credentials copied to coder user."
fi

echo ""
echo "=== Setup complete ==="
