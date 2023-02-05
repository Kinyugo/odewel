# odewei - On Demand Weights Loading

:globe_with_meridians: :rocket: Run any model inference on any hardware! :globe_with_meridians: :rocket:

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

### `model_fn` - The Model's Blueprint

:blue_book: A function that returns a `torch.nn.Module` instance, your model's blueprint! :blue_book:

Here's an example of using `T5ForConditionalGeneration` from the transformers library to define your `model_fn`:

```python
from transformers import T5ForConditionalGeneration, T5Config

config = T5Config.from_pretrained("philschmid/flan-t5-xxl-sharded-fp16")
model_fn = lambda: T5ForConditionalGeneration(config)
```

:pencil: With odewei, you can use any model architecture that you want! :pencil:

### `weights_loader_fn` - The Heart of odewei

:heartpulse: The callable that takes your model's performance to the next level! :heartpulse:

This is where the magic happens. `weights_loader_fn` is a function that takes in the name of the sub-model and a list of strings representing the missing weights and returns a dictionary mapping weight names to tensors. This is where you get to decide your pre-loading strategy, device, and data type.

Here's an example of using `ShardedWeightsLoader` from the odewei library to define your `weights_loader_fn`:

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

:bulb: With odewei, you have the flexibility to choose how your weights are loaded and stored! :bulb:
