# AI_Veritas_Dimensions_Application

Gemstone Dimensions Calculation App

## Development

### Venv

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install black isort  # Optional
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

## API

### Development API

```bash
python3 -m uvicorn deploy.api:app --host 0.0.0.0 --port 8000 --reload --reload-dir deploy
```

### UAT API

```bash
docker compose up --build
docker compose logs -f  # follow logs
docker builder prune --all --force  # clean up