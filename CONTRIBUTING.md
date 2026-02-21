# Contributing to notfallmedizin

Thank you for your interest in contributing to notfallmedizin.

## Code and documentation standards

- **Style:** Follow [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) for code and docstrings.
- **Docstrings:** Use Google-style docstrings (Summary, Args, Returns, Raises, References) in all public modules, classes, and functions.
- **Type hints:** All public APIs must have type annotations. The project uses mypy in strict mode.
- **References:** Clinical and methodological references must be cited in module or class docstrings where applicable.
- **No emoji or em-dash:** Keep tone formal and professional; avoid emoji and em-dash in code and user-facing documentation.

## Repository

- **Author:** Gustav Olaf Yunus Laitinen-Fredriksson Lundström-Imanov  
- **Copyright:** 2026  
- **License:** Apache-2.0  
- **Repository:** [github.com/olaflaitinen/notfallmedizin](https://github.com/olaflaitinen/notfallmedizin)

## Development setup

```bash
git clone https://github.com/olaflaitinen/notfallmedizin.git
cd notfallmedizin
pip install -e ".[dev]"
pytest tests/
```

## Submitting changes

1. Fork the repository and create a branch.
2. Make your changes with tests and docstring updates.
3. Run `ruff check notfallmedizin` and `mypy notfallmedizin`.
4. Open a pull request with a clear description of the change.

## Disclaimer

This library is for research and educational use. It is not a certified medical device. Clinical use must follow local regulations and professional guidelines.
