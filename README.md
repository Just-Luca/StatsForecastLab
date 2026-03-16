# StatsForecastLab — Statistical Forecast Laboratory

> A production-oriented experimentation framework for statistical time series forecasting, built on top of [Nixtla StatsForecast](https://github.com/Nixtla/statsforecast).

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Built on StatsForecast](https://img.shields.io/badge/built%20on-StatsForecast-orange)](https://github.com/Nixtla/statsforecast)

---

## Overview

**StatsForecastLab** is a structured pipeline for experimenting with statistical forecasting models, feature transformations, and forecast horizons. It provides a high-level orchestration layer on top of [Nixtla's StatsForecast](https://github.com/Nixtla/statsforecast) library, enabling systematic benchmarking, automated model selection, and reproducible forecasting experiments.

The framework is designed for data scientists and ML engineers who need to move quickly from hypothesis to production — running controlled experiments across multiple models and configurations while automatically identifying the best-performing setup.

**StatsForecastLab does not re-implement any forecasting models.** All core statistical models are provided by Nixtla StatsForecast. This project builds an experimentation and orchestration layer on top of that foundation.

---

## Key Features

| Feature | Description |
|---|---|
| **Grid Search over Models & Parameters** | Systematically evaluate combinations of models, transformations, and horizons |
| **Automated Best-Model Selection** | Ranks configurations by cross-validation metrics and selects the optimal setup |
| **Cross-Validation Pipeline** | Evaluate model accuracy across rolling windows with configurable horizons |
| **Forecast Visualization** | Built-in plotting utilities for forecast inspection and diagnostic charts |
| **Evaluation Summaries** | Aggregated metrics tables for model comparison across experiments |
| **Configurable Transformations** | Apply and benchmark different data transformations as part of the pipeline |
| **Production-Ready Design** | Clean interface suitable for embedding in batch forecasting pipelines |

---

## Architecture / Design Philosophy

StatsForecastLab follows three core principles:

**1. Separation of concerns.** Experimentation logic, model configuration, and evaluation are handled by distinct components. This makes it straightforward to swap models, modify grid search parameters, or extend the pipeline without touching unrelated code.

**2. StatsForecast as the forecasting engine.** StatsForecastLab does not compete with or duplicate StatsForecast. It provides the scaffolding for structured experimentation: configuration management, cross-validation orchestration, metric aggregation, and best-model selection.

**3. Reproducibility first.** All experiments are driven by explicit configuration files (`grid_search_parameters.py`, `constants.py`), ensuring that any experiment can be replicated exactly.

```
User Configuration (constants.py, grid_search_parameters.py)
         │
         ▼
  StatsForecastLab (hup_statsmacrocast.py)
         │
    ┌────┴────┐
    │         │
Cross-Val   Predict
    │         │
    └────┬────┘
         │
   Evaluation & Plotting (utils.py)
         │
         ▼
  Best Model Selection → Production Forecast
```

---

## Repository Structure

```
StatsForecastLab/
├── src/
│   ├── constants.py                # Global constants: horizons, frequencies, metric targets
│   ├── grid_search_parameters.py   # Model configurations and transformation grid definitions
│   ├── hup_statsmacrocast.py       # Core StatsForecastLab class and pipeline orchestration
│   └── utils.py                    # Evaluation utilities, metric aggregation, plotting helpers
├── test_statsmacrocast.ipynb       # End-to-end experiment walkthrough notebook
└── README.md
```

### Module Responsibilities

**`constants.py`** — Defines shared configuration constants used across the pipeline, including forecast horizons, time series frequencies, and evaluation metric targets.

**`grid_search_parameters.py`** — Specifies the search space for experimentation: which models to evaluate, which parameter combinations to test, and which transformations to apply.

**`hup_statsmacrocast.py`** — Houses the primary `StatsForecastLab` class. Exposes the core API: `predict`, `cross_validate`, `plot`, and `summary`.

**`utils.py`** — Helper functions for metric computation, result aggregation, and visualization. Designed to be imported independently when needed.

---

## Installation

### Requirements

- Python 3.8+
- [Nixtla StatsForecast](https://github.com/Nixtla/statsforecast) (`statsforecast>=1.0`)

### Install from source

```bash
git clone https://github.com/your-org/StatsForecastLab.git
cd StatsForecastLab
pip install -r requirements.txt
```

### Install dependencies manually

```bash
pip install statsforecast pandas numpy matplotlib
```

> **Note:** StatsForecast uses Numba for JIT-compiled model implementations. The first run may take additional time for Numba compilation. Subsequent runs will be significantly faster.

---

## Quick Start Example

```python
import pandas as pd
from src.hup_statsmacrocast import StatsForecastLab

# Load a long-format time series dataset
df = pd.read_csv("data/my_series.csv")
# Expected columns: unique_id, ds, y

# Initialize the lab
lab = StatsForecastLab(
    df=df,
    freq="M",          # Monthly frequency
    horizon=12,        # Forecast 12 periods ahead
    metric="MAE"       # Optimization target
)

# Run cross-validation grid search
lab.cross_validate()

# Inspect model comparison summary
lab.summary()

# Generate forecast with the best-selected model
forecast = lab.predict()

# Visualize results
lab.plot()
```

---

## Pipeline Workflow

StatsForecastLab executes experiments through the following stages:

```
1. DATA INGESTION
   └─ Load long-format time series (unique_id / ds / y)

2. CONFIGURATION
   └─ Load model grid and transformation parameters

3. CROSS-VALIDATION
   └─ Evaluate all model × transformation × horizon combinations
   └─ Compute evaluation metrics per configuration

4. BEST-MODEL SELECTION
   └─ Rank configurations by target metric
   └─ Select optimal model and parameters

5. PREDICTION
   └─ Train selected model on full history
   └─ Generate forecasts for defined horizon

6. REPORTING
   └─ Aggregated metric summaries
   └─ Forecast and diagnostic plots
```

---

## Core Components

### `StatsForecastLab` Class

The primary interface to the framework. Instantiate with a dataset and configuration, then call methods to run the pipeline.

#### Constructor

```python
StatsForecastLab(
    df: pd.DataFrame,       # Long-format time series (unique_id, ds, y)
    freq: str,              # Pandas-compatible frequency string (e.g., "D", "W", "M")
    horizon: int,           # Number of forecast steps
    metric: str = "MAE",    # Evaluation metric for model selection
    models: list = None,    # Override default model list (optional)
    transformations: list = None  # Override default transformations (optional)
)
```

#### Methods

| Method | Description |
|---|---|
| `predict()` | Trains the best model on full history and returns a forecast DataFrame |
| `cross_validate()` | Runs rolling-window cross-validation across all configurations |
| `summary()` | Returns a DataFrame of aggregated metrics per model/configuration |
| `plot()` | Renders forecast and residual diagnostic plots |

---

## Example Experiment Workflow

The following demonstrates a complete benchmarking experiment comparing classical models across a monthly retail demand series.

```python
from src.hup_statsmacrocast import StatsForecastLab
from src.grid_search_parameters import MODEL_GRID
import pandas as pd

# Load data
df = pd.read_csv("data/retail_demand.csv")

# Initialize with a custom model grid
lab = StatsForecastLab(
    df=df,
    freq="M",
    horizon=6,
    metric="MASE",
    models=MODEL_GRID
)

# Run full experiment
lab.cross_validate()

# Review results
results = lab.summary()
print(results.sort_values("MASE").head(10))

# Generate and visualize best-model forecast
forecast = lab.predict()
lab.plot()
```

Sample output from `summary()`:

```
         model  transformation    MASE    RMSE    MAE
0    AutoARIMA            none  0.821  142.3   88.4
1          ETS       log1p(y)  0.843  148.1   91.2
2        Theta            none  0.904  161.7   98.5
3  SeasonalNaive            none  1.204  201.2  124.8
```

---

## Visualization

StatsForecastLab provides built-in plotting for experiment diagnostics and forecast review.

```python
# Forecast plot: actuals vs. predicted with prediction intervals
lab.plot(kind="forecast")

# Model comparison: metric distributions across configurations
lab.plot(kind="comparison")

# Residual diagnostics for the best model
lab.plot(kind="residuals")
```

All plots are generated using `matplotlib` and can be customized by passing standard `matplotlib` keyword arguments.

---

## Integration with StatsForecast

StatsForecastLab is built on top of **[Nixtla StatsForecast](https://github.com/Nixtla/statsforecast)** and does not re-implement any forecasting algorithms. All statistical models used within the pipeline — including AutoARIMA, ETS, Theta, Naive, and Seasonal Naive — are sourced directly from the StatsForecast library.

### Supported StatsForecast Models

| Model | Class | Description |
|---|---|---|
| AutoARIMA | `AutoARIMA` | Automatic ARIMA with stepwise selection |
| ETS | `AutoETS` | Exponential Smoothing (Error, Trend, Seasonality) |
| Theta | `AutoTheta` | Theta method for seasonal decomposition |
| Naive | `Naive` | Random walk baseline |
| Seasonal Naive | `SeasonalNaive` | Seasonal repetition baseline |
| CES | `AutoCES` | Complex Exponential Smoothing |

Refer to the [StatsForecast documentation](https://nixtlaverse.nixtla.io/statsforecast/index.html) for the full model catalog and configuration options.

### Data Format Compatibility

StatsForecastLab uses the same long-format convention as StatsForecast:

| Column | Type | Description |
|---|---|---|
| `unique_id` | str / int | Series identifier |
| `ds` | datetime | Timestamp |
| `y` | float | Target variable |

---

## Performance and Scalability

StatsForecastLab inherits the performance characteristics of Nixtla StatsForecast:

- **Numba-accelerated model fitting** — Classical models (ETS, ARIMA, Theta) are JIT-compiled via Numba, enabling fast training across large collections of time series.
- **Vectorized multi-series forecasting** — StatsForecast is optimized for panel datasets with many concurrent series. StatsForecastLab is designed to operate in this same regime.
- **Parallelism** — StatsForecast supports parallel model fitting via the `n_jobs` parameter, which can be passed through the StatsForecastLab configuration.

For very large datasets (millions of series), consider integrating StatsForecast's [Spark or Ray backends](https://github.com/Nixtla/statsforecast) directly.

---

## Use Cases

StatsForecastLab is well-suited for:

- **Demand Forecasting** — Benchmarking statistical models across SKU-level or category-level retail and supply chain series.
- **Financial Time Series** — Evaluating model performance on macroeconomic indicators, asset prices, or revenue projections.
- **Operations Forecasting** — Forecasting service volumes, call center arrivals, energy consumption, or infrastructure load.
- **Model Benchmarking** — Systematic comparison of classical statistical models before committing to a production forecasting architecture.
- **Research & Experimentation** — Rapid iteration over model configurations and transformations in a controlled, reproducible environment.

---

## Roadmap / Future Improvements

- [ ] Support for exogenous regressors (future covariates)
- [ ] Integration with MLflow for experiment tracking
- [ ] Extended transformation library (Box-Cox, differencing, detrending)
- [ ] Probabilistic forecast calibration metrics (CRPS, coverage)
- [ ] REST API wrapper for serving forecasts
- [ ] CLI interface for pipeline execution
- [ ] Docker packaging for reproducible deployment
- [ ] Integration with additional Nixtla libraries (NeuralForecast, MLForecast)

---

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting a pull request.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a pull request

Please ensure all code is tested and documented before submission.

---

## License

This project is licensed under the **Apache 2.0 License**. See the [LICENSE](./LICENSE) file for details.

**StatsForecast** (the underlying library) is developed and maintained by [Nixtla](https://github.com/Nixtla) and is also licensed under the **Apache 2.0 License**.

---

## Acknowledgements

StatsForecastLab is built on top of [Nixtla StatsForecast](https://github.com/Nixtla/statsforecast), an open-source Python library for high-performance statistical forecasting. The authors of StatsForecastLab gratefully acknowledge the work of the Nixtla team in developing and maintaining StatsForecast.

- **StatsForecast GitHub:** https://github.com/Nixtla/statsforecast
- **StatsForecast Documentation:** https://nixtlaverse.nixtla.io/statsforecast/index.html

---

## Citation

If you use StatsForecastLab in your research or production work, please cite both this project and the underlying StatsForecast library.

**StatsForecast (Nixtla):**

```bibtex
@software{statsforecast,
  author  = {Nixtla},
  title   = {StatsForecast: Lightning fast forecasting with statistical and econometric models},
  year    = {2023},
  url     = {https://github.com/Nixtla/statsforecast}
}
```

**StatsForecastLab:**

```bibtex
@software{statsforecastlab,
  title   = {StatsForecastLab: Statistical Forecast Laboratory},
  year    = {2024},
  url     = {https://github.com/your-org/StatsForecastLab}
}
```

---

*StatsForecastLab is an independent project and is not affiliated with or endorsed by Nixtla.*
