# Building the documentation

Requirements: Sphinx and the Read the Docs theme (included in ``[dev]``).

From project root:

```bash
pip install -e ".[dev]"
sphinx-build -b html docs docs/_build/html
```

From the `docs/` directory:

```bash
sphinx-build -b html . _build/html
```

Open ``docs/_build/html/index.html`` in a browser.

Contents
--------

- **index**: Overview and table of contents.
- **installation**: PyPI and source install, optional extras.
- **quickstart**: Minimal code examples (scoring, triage, vitals, survival, benchmarks).
- **modules**: List of all 13 modules and 57 submodules.
- **api**: Full API reference (auto-generated from docstrings).
- **references**: Clinical and method references.
- **glossary**: Terms (ESI, SOFA, GCS, HEART, etc.).
- **license**: Apache License 2.0.
- **disclaimer**: Research use only; not a medical device.
