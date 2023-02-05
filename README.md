# odewei

:fire: Transform your PyTorch models to use on-demand weights loading with ease!

With this library, you can efficiently manage your model's memory usage by only loading the required weights when necessary. Say goodbye to loading all the weights at once and hello to a more efficient and lighter model!

## Setup

Setup your virtual environment using conda or venv.

### Install package from GitHub

```bash
    pip install -q git+https://github.com/Kinyugo/odewei.git
```

### Install package in edit mode

```bash
    git clone https://github.com/Kinyugo/odewei.git
    cd odewei
    pip install -q -r requirements.txt
    pip install -e .
```

## Usage

```python
from odewei import on_demand_weights_loading

model = init_on_demand_weights_model(model_fn, weights_loader_fn, enable_preloading=True)
```
