# odewel - On Demand Weights Loading

:zap: Limitless Model Inference :zap:

odewel (/əʊdwel/) is a handy library that enables you to take your models to the next level of efficiency. With odewel, you can transform your conventional models that load all weights during initialization, into models that load weights only when required for computation. This results in a leaner, more nimble model that can be run on any hardware, regardless of its size or capabilities. Say goodbye to the limitations of your hardware and hello to the freedom of on demand weights loading with odewel!

## Setup

### Prerequisites

Set up a virtual environment. You can use any environment manager you wish, e.g: `conda` or `venv` e.t.c.

### Installation

To install `odewel`, you have two options:

#### Option 1: Install package from GitHub

```bash
    pip install -q git+https://github.com/Kinyugo/odewel.git
```

#### Option 2: Install package in edit mode

```bash
    git clone https://github.com/Kinyugo/odewel.git
    cd odewel
    pip install -q -r requirements.txt
    pip install -e .
```

## Usage

```python
from odewel.torch import init_on_demand_weights_model

model_fn = # () -> model instance
weights_loader_fn = # (module name, list of weight names) -> mapping of weight name to weight
model = init_on_demand_weights_model(model_fn, weights_loader_fn)
```

### `model_fn` - The Model's Blueprint

:blue_book: A function that returns a `torch.nn.Module` instance, your model's blueprint!

Here's an example of using `T5ForConditionalGeneration` from the transformers library to define your `model_fn`:

```python
from transformers import T5ForConditionalGeneration, T5Config

config = T5Config.from_pretrained("philschmid/flan-t5-xxl-sharded-fp16")
model_fn = lambda: T5ForConditionalGeneration(config)
```

:pencil: With odewel, you can use any model architecture that you want!

### `weights_loader_fn` - The Heart of odewel

:magic_wand: This is where the magic happens :magic_wand:

`weights_loader_fn` is a function that takes in the name of the sub-model and a list of strings representing the missing weights and returns a dictionary mapping weight names to weights. This is where you get to decide your pre-loading strategy, device, and data type.

Here's an example of using `ShardedWeightsLoader` from the odewel library to define your `weights_loader_fn`:

```python
from odewel.torch import ShardedWeightsLoader

weights_loader_fn = ShardedWeightsLoader(
    index_file_path="path/to/index.json",
    weights_dir="path/to/weights",
    weights_mapping_key="weight_map",
    device=torch.device("cuda"),
    dtype=torch.float16,
)
```

:bulb: With odewel, you have the flexibility to choose how and when your weights are loaded.

## Considerations

:spiral_notepad: A few things to note:

- The library only supports PyTorch models. Support for other libraries will be added in the future.

- When using odewel, all weights are initialized on the meta device without any data. Therefore, to ensure a correct forward pass, the required tensor must be set as a parameter or buffer and must be included in the weight mapping returned by the `weights_loader_fn`.

## :clipboard: To-Do

- Implement Smart and Efficient Weight Loaders: The current weight loaders load weights one after the other in a greedy manner. However, all weights in a network are not equal. Some weights may be reused and the larger ones may take more time to load. As such, implementing efficient pre-loading strategies is crucial for boosting performance.
