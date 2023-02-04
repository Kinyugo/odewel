from contextlib import contextmanager
from typing import Callable, Dict, Generator, Iterable, List, Optional, Tuple

import torch
from torch import Tensor, nn

from .utils import free_memory, prepend_string


def has_empty_tensor(tensors: Iterable[Tensor]) -> bool:
    """
    Returns True if any of the tensors are emptys, otherwise False.

    Parameters
    ----------
    tensors: Iterable[Tensor]
        An iterable of tensors.

    Returns
    -------
    bool
        True if any of the tensors are empty, otherwise False.
    """
    for tensor in tensors:
        if tensor.is_meta:
            return True
    return False


def has_empty_weights(module: nn.Module, recurse: bool = False) -> bool:
    """
    Check if a module has any empty weights.

    Parameters
    ----------
    module : nn.Module
        PyTorch module to check for empty weights.
    recurse : bool, default=False
        If True, recursively check sub-modules, by default False.

    Returns
    -------
    bool
        True if the module has any empty weight, False otherwise.
    """
    has_empty_param = has_empty_tensor(module.parameters(recurse=recurse))
    has_empty_buffer = has_empty_tensor(module.buffers(recurse=recurse))

    return has_empty_param or has_empty_buffer


def get_empty_tensors_names(
    names_and_tensors: Iterable[Tuple[str, Tensor]]
) -> List[str]:
    """
    Get names of empty tensors from a list of names and tensors.

    Parameters
    ----------
    names_and_tensors : Iterable[Tuple[str, Tensor]]
        List of (name, tensor) pairs.

    Returns
    -------
    List[str]
        List of names of empty tensors.

    """
    return [name for name, tensor in names_and_tensors if tensor.is_meta]


def get_empty_weights_names(
    module: nn.Module,
    recurse: bool = False,
    prefix: Optional[str] = None,
    separator: str = ".",
) -> List[str]:
    """
    Get the names of the empty tensors.

    Parameters
    ----------
    module : nn.Module
        The PyTorch module with empty tensors.
    recurse : bool, False
        Recurse into the module, by default False.
    prefix : str, optional
        Prefix to add to the names of the empty tensors, by default None.
    separator : str, default="."
        Separator to use when adding the prefix, by default ".".

    Returns
    -------
    List[str]
        List of names of empty tensors.
    """
    empty_params_names = get_empty_tensors_names(
        module.named_parameters(recurse=recurse)
    )
    empty_buffers_names = get_empty_tensors_names(module.named_buffers(recurse=recurse))
    empty_weights_names = empty_params_names + empty_buffers_names

    if prefix is not None:
        return prepend_string(prefix, empty_weights_names, separator)
    return empty_weights_names


def get_nested_module_by_weight_accessor(module: nn.Module, accessor: str) -> nn.Module:
    """Return the nested module specified by its weight accessor.

    Parameters
    ----------
    module : nn.Module
        A PyTorch module to access the nested module from.
    accessor : str
        A string representing the nested accessor.
        The string should be a dot-separated sequence of attribute accesses for the
        module's weights and not the module it self e.g `"conv.weight"`.

    Returns
    -------
    nn.Module
        The nested module specified by the weight `accessor`
    """

    attrs = accessor.split(".")
    for attr in attrs[:-1]:
        module = getattr(module, attr)

    return module


def get_nested_weight_accessor(accessor: str) -> str:
    """Extracts the name of a weight from its accessor string.

    Parameters
    ----------
    accessor : str
        The accessor string to extract the weight name from.

    Returns
    -------
    str
        The name of the weight.
    """
    return accessor.split(".")[-1]


def get_nested_weight(module: nn.Module, weight_accessor: str) -> Tensor:
    """
    Get a weight from a nested module.

    Parameters
    ----------
    module : torch.nn.Module
        A module containing the desired weight.
    weight_accessor : str
        A string representing the path to the desired weight. For example,
        "layer.weight" would access the "weight" attribute of the "layer"
        submodule.

    Returns
    -------
    weight : torch.Tensor
        The desired weight tensor.

    """
    module = get_nested_module_by_weight_accessor(module, weight_accessor)
    return getattr(module, get_nested_weight_accessor(weight_accessor))


def set_nested_parameter(
    module: nn.Module, parameter_accessor: str, parameter: Tensor
) -> None:
    """Sets the value of a nested parameter in a PyTorch model.

    Parameters
    ----------
    module : nn.Module
        The PyTorch module to set the parameter in.
    parameter_accessor : str
        A string indicating the path to the nested module and parameter to set.
    parameter : Tensor
        The tensor to set as the value of the parameter.

    Returns
    -------
    None
    """

    module = get_nested_module_by_weight_accessor(module, parameter_accessor)
    setattr(module, get_nested_weight_accessor(parameter_accessor), parameter)


def register_nested_buffer(
    module: nn.Module, buffer_accessor: str, buffer: Tensor
) -> None:
    """Registers a buffer with a nested module in a PyTorch model.

    Parameters
    ----------
    module : nn.Module
        The PyTorch module to register the buffer with.
    buffer_accessor : str
        A string indicating the path to the nested module to register the buffer with.
    buffer : Tensor
        The buffer tensor to register.

    Returns
    -------
    None
    """

    module = get_nested_module_by_weight_accessor(module, buffer_accessor)
    module.register_buffer(get_nested_weight_accessor(buffer_accessor), buffer)


def load_weights_dict(module: nn.Module, weights_dict: Dict[str, Tensor]) -> None:
    """Loads a mapping of weights into a PyTorch module.

    Parameters
    ----------
    module : nn.Module
        The PyTorch module to load the weights_dict into.
    weights_dict : Dict[str, Tensor]
        A dictionary mapping weight names to tensors.
    Returns
    -------
    None
    """
    for weight_name in weights_dict.keys():
        curr_weight = get_nested_weight(module, weight_name)

        # Avoid loading unkown weights as we cannot infer whether they are
        # parameters or buffers
        if curr_weight is None:
            return

        weight = weights_dict[weight_name]
        # Ensure that parameters and `torch.nn.Parameter` type instead of just tensors
        if isinstance(curr_weight, nn.Parameter):
            weight = nn.Parameter(weight, requires_grad=curr_weight.requires_grad)
            set_nested_parameter(module, weight_name, weight)
        # Register buffer to ensure they will be transferred to devices correctly
        else:
            register_nested_buffer(module, weight_name, weight)
        # Delete the current weight as it's no longer needed
        del curr_weight


def filter_weights_dict_by_weights_names(
    weights_dict: Dict[str, Tensor], weights_names: List[str]
) -> Dict[str, Tensor]:
    """
    Filter a dictionary of weights by the names of the weights.

    Parameters
    ----------
    weights_dict : Dict[str, Tensor]
        A dictionary of weights where the keys are the names of the weights
        and the values are the weight tensors.
    weights_names : List[str]
        A list of names of the weights to keep in the filtered dictionary.

    Returns
    -------
    filtered_weights_dict : Dict[str, Tensor]
        A filtered dictionary of weights where the keys are the names of the
        weights and the values are the weight tensors. The dictionary only
        contains the weights whose names are specified in the `weights_names` list.
    """
    return {
        name: weight for name, weight in weights_dict.items() if name in weights_names
    }


def make_weights_loading_hook(
    module_name: str,
    model: nn.Module,
    weights_loader_fn: Callable[[str, List[str]], Dict[str, Tensor]],
    enable_preloading: bool = True,
) -> Callable[..., None]:
    """
    Create a hook that loads the weights for a specific PyTorch module.

    Parameters
    ----------
    module_name : str
        The name of the PyTorch module.
    model : nn.Module
        The PyTorch model that contains the module.
    weights_loader_fn : callable
        A function that loads the weights for the module. The function should take the
        name of the module and a list of missing weights as inputs and return a dictionary
        mapping weight names to tensors.
    enable_preloading : bool, default=True
        Whether to pre-load the weights before the forward pass.

    Returns
    -------
    callable
        A hook that loads the weights for the specified PyTorch module.
    """

    @torch.no_grad()
    def weights_loading_hook(self, *args, **kwargs) -> None:
        # If there are no empty weights then we don't need to run
        if not has_empty_weights(self):
            return None

        # Free previously unreferenced tensors
        free_memory()

        # Get the empty weights names for the current module only
        empty_weights_names = get_empty_weights_names(
            self, recurse=False, prefix=module_name
        )

        # Fetch weights using the loader function
        weights_dict = weights_loader_fn(module_name, empty_weights_names)

        # Filter only the weights for the current model if pre-loading is not enabled
        if not enable_preloading:
            weights_dict = filter_weights_dict_by_weights_names(
                weights_dict, empty_weights_names
            )

        # Load the weights into the model
        load_weights_dict(model, weights_dict)

        # Check if we still have empty weights in the current module
        if has_empty_weights(self):
            # Get weights names that could not be loaded
            empty_weights_names = list(
                set(empty_weights_names) - set(weights_dict.keys())
            )
            raise RuntimeError(
                f"Some weights for the {module_name} module could not be loaded."
                f" Missing weights: {empty_weights_names}"
            )

        # Free memory by deleting the weights dict and freeing unreferenced tensors
        del weights_dict
        free_memory()

    return weights_loading_hook


def make_weights_unloading_hook(module_name: str) -> Callable:
    """
    Create a hook that unloads the weights of a PyTorch module by replacing them with empty tensors.

    Parameters
    ----------
    module_name : str
        The name of the PyTorch module.

    Returns
    -------
    callable
        A hook that unloads the weights of the specified PyTorch module.
    """

    @torch.no_grad()
    def weights_unloading_hook(self, *args, **kwargs) -> None:
        # Replace weights with empty weights
        self.to(torch.device("meta"))

    return weights_unloading_hook


def register_on_demand_weights_hooks(
    model: nn.Module,
    weights_loader_fn: Callable[[str, List[str]], Dict[str, torch.Tensor]],
    enable_preloading: bool = True,
) -> nn.Module:
    """
    Registers pre and post forward hooks to a PyTorch model for on demand weights loading.

    Parameters
    ----------
    model : nn.Module
        A PyTorch `nn.Module` instance.
    weights_loader_fn : callable
        A function that takes in the name of the sub-model and a list of missing weights,
        and returns a dictionary mapping weights names to PyTorch tensors. It is expected
        that the returned tensors will be mapped to the correct device and have the correct
        data type.
    enable_preloading : bool, default=True
        Whether to preload all the weights returned by the `weights_loader_fn`. If `False`,
        the weights will be loaded on demand for each sub-module during each forward pass
        (default: `True`).

    Returns
    -------
    nn.Module
        The input `nn.Module` instance with added pre and post forward hooks.
    """
    for module_name, module in model.named_modules():
        if has_empty_weights(module):
            # Register hook to load the weights before use
            module.register_forward_pre_hook(
                make_weights_loading_hook(
                    module_name,
                    model,
                    weights_loader_fn,
                    enable_preloading,
                )
            )
            # Register hook to unload the weights after use
            module.register_forward_hook(make_weights_unloading_hook(module_name))

    return model


@contextmanager
def init_empty_weights() -> Generator[None, None, None]:
    """
    Context manager that sets empty PyTorch parameters and buffers on modules registered
    after entering the context.

    Modules that have already been registered with parameters or buffers before entering
    the context will remain unchanged.
    """
    old_register_parameter = nn.Module.register_parameter
    old_register_buffer = nn.Module.register_buffer

    def register_empty_parameter(module: nn.Module, name: str, param: Tensor) -> None:
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            module._parameters[name] = param_cls(
                module._parameters[name].to(torch.device("meta")), **kwargs
            )

    def register_empty_buffer(
        module: nn.Module, name: str, buffer: Tensor, persistent: bool = False
    ) -> None:
        old_register_buffer(module, name, buffer, persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(torch.device("meta"))

    try:
        nn.Module.register_parameter = register_empty_parameter
        nn.Module.register_buffer = register_empty_buffer
        yield

    finally:
        nn.Module.register_parameter = old_register_parameter
        nn.Module.register_buffer = old_register_buffer


def init_empty_model(model_fn: Callable[[], nn.Module]) -> nn.Module:
    """Initializes a PyTorch model with empty weights.

    Parameters
    ----------
    model_fn : Callable
        A function that returns a `torch.nn.Module` instance.

    Returns
    -------
    nn.Module
        A `torch.nn.Module` instance with empty weights.

    Notes
    -----
    empty weights have no storage and thus no memory allocation.
    """
    with init_empty_weights():
        model = model_fn()

    return model


def init_on_demand_weights_model(
    model_fn: Callable[[], nn.Module],
    weights_loader_fn: Callable[[str, List[str]], Dict[str, Tensor]],
    enable_preloading: bool = True,
) -> nn.Module:
    """Transform a PyTorch model to use on demand weights loading.

    Parameters
    ----------
    model_fn : Callable
        A function that returns a `torch.nn.Module` instance.
    weights_loader_fn : Callable
        A function that takes in the name of the sub-model and a list of strings
        representing the missing weights, and returns a dictionary of weights names
        to tensor mapping. It is expected that the returned tensors will
        be already mapped to the correct device and have the correct data type.
    enable_preloading : bool, default=True
        Whether to load all the weights returned by the `weights_loader_fn`. If `False`,
        the weights will be loaded on demand for each sub-module during each forward pass.

    Returns
    -------
    nn.Module
        A `torch.nn.Module` instance that uses on demand weights loading.

    Notes
    -----
    The model is first initialized with empty weights, i.e parameters and buffers
    that have no data and thus no memory allocation. Then hooks are registered for
    each sub-module that has empty weights, to load the weights before the sub-module
    computation and to unload the weights after the computation.
    """

    # Initialize the model with empty parameters and buffers i.e: tensors with no data
    model = init_empty_model(model_fn)
    # Add weights loading and unloading hooks to the model
    model = register_on_demand_weights_hooks(
        model, weights_loader_fn, enable_preloading
    )

    return model


if __name__ == "__main__":
    model_fn = lambda: nn.Sequential(nn.Linear(1, 5), nn.GELU(), nn.Linear(5, 1))
    regular_model = model_fn()
    state_dict = regular_model.state_dict()
    # loader_fn = lambda _, missing_weights: {
    #     name: weight for name, weight in state_dict.items() if name in missing_weights
    # }
    loader_fn = lambda *_: {}
    odewei_model = init_on_demand_weights_model(model_fn, loader_fn)
    x = torch.rand(1, 1)
    odewei_model(x)
