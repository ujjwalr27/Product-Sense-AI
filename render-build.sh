pip install --upgrade pip setuptools wheel

# Install requirements
pip install --only-binary :all: wordcloud transformers
pip install -r requirements.txt
