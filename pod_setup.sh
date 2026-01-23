curl -fsSL https://claude.ai/install.sh | bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc   
pip install uv
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
uv pip install -e ".[dev]"
uv pip install -e ".[viz]"

git config --global user.name "Oscar Gilg"
git config --global user.email "oscar.gilg18@gmail.com"


# Copy over .env file