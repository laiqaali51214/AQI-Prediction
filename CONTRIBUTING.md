# Contributing

## Development Setup

1. Clone the repository
2. Create virtual environment: `python -m venv venv`
3. Activate virtual environment
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `config/env.example` to `.env` and configure
6. Run tests: `pytest tests/`

## Code Style

- Follow PEP 8 style guide
- Use type hints where appropriate
- Add docstrings to all functions and classes
- Keep functions focused and modular

## Project Structure

```
10pearlsAQI/
├── api/              # FastAPI service
├── app/              # Streamlit dashboard
├── pipelines/        # Data and ML pipelines
├── config/           # Configuration files
├── scripts/          # Utility scripts
├── docs/             # Documentation
├── tests/            # Unit tests
└── notebooks/        # Jupyter notebooks
```

## Pull Request Process

1. Create a feature branch
2. Make your changes
3. Add tests if applicable
4. Update documentation
5. Submit pull request with clear description
