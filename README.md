# Time Series Decomposition Streamlit App

An interactive Streamlit lesson that shows how a time series can be split into
level, trend, seasonality, event, and residual components.

The beginner tabs use a synthetic series with known components. The intermediate
tabs estimate level, trend, seasonality, and residuals from observed data,
support CSV upload, explain the fitting math step by step, and forecast future
values either by extending a smoothed trend-cycle or by fitting a linear
regression line. The regression example intentionally leaves seasonality out so
the level-plus-trend model is easier to inspect.

## Run

```powershell
pip install -r requirements.txt
streamlit run app.py
```

Use the sidebar controls to change the signal, then explore the tabs to compare
components, rebuild the observed series, and answer short practice questions.
