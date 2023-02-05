# odewei - On Demand Weights Loading

:zap: Limitless Model Inference :zap:

odewei (/əʊdwei/) is a handy library that enables you to take your models to the next level of efficiency. With odewei, you can transform your conventional models that load all weights during initialization, into models that load weights only when required for computation. This results in a leaner, more nimble model that can be run on any hardware, regardless of its size or capabilities. Say goodbye to the limitations of your hardware and hello to the freedom of on demand weights loading with odewei!

## Setup

### Prerequisites

Set up a virtual environment. You can use any environment manager you wish, e.g: `conda` or `venv` e.t.c.

### Installation

To install `odewei`, you have two options:

#### Option 1: Install package from GitHub

```bash
    pip install -q git+https://github.com/Kinyugo/odewei.git
```

#### Option 2: Install package in edit mode

```bash
    git clone https://github.com/Kinyugo/odewei.git
    cd odewei
    pip install -q -r requirements.txt
    pip install -e .
```

## Usage

```python
from odewei.torch import init_on_demand_weights_model

model_fn = # () -> model instance
weights_loader_fn = # (module name, list of weight names) -> mapping of weight name to weight
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

:magic_wand: This is where the magic happens :magic_wand:

`weights_loader_fn` is a function that takes in the name of the sub-model and a list of strings representing the missing weights and returns a dictionary mapping weight names to weights. This is where you get to decide your pre-loading strategy, device, and data type.

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

:bulb: With odewei, you have the flexibility to choose how and when your weights are loaded :bulb:
