FROM python:3.10-slim

WORKDIR /workspace

# ── SynEval: install dependencies first (cached unless requirements.txt changes)
COPY SynEval/requirements.txt ./SynEval/requirements.txt
RUN pip install --no-cache-dir -r SynEval/requirements.txt

# ── SynEval: copy source and install package (fast; no heavy downloads)
COPY SynEval/ ./SynEval/
RUN pip install --no-cache-dir -e SynEval/

# ── SynGen: install dependencies first
COPY SynGen/requirements.txt ./SynGen/requirements.txt
RUN pip install --no-cache-dir -r SynGen/requirements.txt

# ── SynGen: copy source and install package
COPY SynGen/ ./SynGen/
RUN pip install --no-cache-dir -e SynGen/

# ── Pre-download NLTK corpora needed by SynEval
RUN python -c "\
import nltk; \
[nltk.download(d, quiet=True) for d in \
 ('vader_lexicon', 'stopwords', 'punkt', 'punkt_tab', 'wordnet')]"

# ── Copy top-level reproduction script into the container
COPY reproduce.sh /workspace/reproduce.sh

CMD ["bash"]
