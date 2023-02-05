# odewei - On Demand Weights Loading

:rocket: Run any model inference on any hardware! :rocket:

odewei (/əʊdwei/) is a library that allows you to transform your regular model, where weights are loaded on initialization, into a model where weights are loaded only when they are needed for computation. This results in a lighter and more efficient model that can be run on virtually any hardware, regardless of its size.

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
from odewei.torch import init_on_demand_weights_model

model = init_on_demand_weights_model(model_fn, weights_loader_fn)
```

### `model_fn`

A function that returns a `torch.nn.Module` instance.

```python
from transformers import T5ForConditionalGeneration, T5Config

config = T5Config.from_pretrained("philschmid/flan-t5-xxl-sharded-fp16")
model_fn = lambda: T5ForConditionalGeneration(config)
```

### `weights_loader_fn`

A callable that takes in the name of the sub-model and a list of strings representing the missing weights, and returns a dictionary of weights names to tensor mapping. It is expected that the returned tensors will be already mapped to the correct device and have the correct data type.

```python
from odewei.torch import ShardedWeightsLoader

weights_loader_fn = ShardedWeightsLoader(
    index_file_path="path/to/index.json",
    weights_dir="path/to/weights",
    weights_mapping_key="weight_map",
    device=torch.device("cuda"),
    dtype=torch.float16,
)
```
