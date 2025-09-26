2. (Optional) Create and activate a virtual environment:

```bash
# Using conda
conda create -n syneval python=3.10
conda activate syneval

# Or using venv
python -m venv syneval_env
source syneval_env/bin/activate  # On Windows: syneval_env\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Setup NLTK data and create directories:

```bash
python setup_nltk.py
```

**That's it!** You can now use SynEval.

**Note**: If you encounter dependency conflicts, use a fresh virtual environment.
