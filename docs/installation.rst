Installation
============

Requirements
------------

- Python >= 3.9
- NumPy >= 1.24
- SciPy >= 1.10
- pandas >= 2.0
- scikit-learn >= 1.3

From PyPI
---------

.. code-block:: bash

   pip install notfallmedizin

Optional extras:

.. code-block:: bash

   pip install notfallmedizin[imaging]   # PyTorch, torchvision for imaging
   pip install notfallmedizin[nlp]       # transformers for NLP
   pip install notfallmedizin[timeseries] # statsmodels for ARIMA
   pip install notfallmedizin[full]      # all optional dependencies
   pip install notfallmedizin[dev]       # pytest, mypy, ruff, sphinx

From source
-----------

.. code-block:: bash

   git clone https://github.com/olaflaitinen/notfallmedizin.git
   cd notfallmedizin
   pip install -e .

Verification
------------

.. code-block:: python

   import notfallmedizin
   print(notfallmedizin.__version__)
   from notfallmedizin.scoring.sepsis import qSOFAScore
   q = qSOFAScore()
   r = q.calculate(systolic_bp=95, respiratory_rate=24, altered_mentation=True)
   print(r.total_score)
