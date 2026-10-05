import os
import getpass
from datetime import datetime
import torch
import torch.distributed as dist
import inspect
import importlib.util
import socket
import logging
from typing import Dict, Union, Type, List


def get_open_port():
    """Get an open port on localhost.
    
    Args:
        None
    Returns:
        int: An open port number.
    Example:
        >>> port = get_open_port()
        >>> print(port) # the port number which is available, e.g.,
        54321
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  # bind to all interfaces and use an OS provided port
        return s.getsockname()[1]  # return only the port number


def get_remote_file(remote_path, local_path=None):
    """Get a file from a remote host via scp, if the remote host is not the local host.
    
    Args:
        remote_path: The remote path in the format "hostname:path".
        local_path: The local path to save the file to. If None, uses the remote path.
    
    Returns:
        The local path to the file.
        
    Example:
        >>> get_remote_file("remote_host:/path/to/file.txt", "/local/path/to/file.txt")
        '/local/path/to/file.txt'
    """
    hostname, path = remote_path.split(':')
    local_hostname = socket.gethostname()
    if hostname == local_hostname or hostname == local_hostname[:local_hostname.find('.')]:
        return path

    if local_path is None:
        local_path = path
    # local_path = local_path.replace('/scr-ssd', '/scr')    
    if os.path.exists(local_path):
        return local_path
    local_dir = os.path.dirname(local_path)
    os.makedirs(local_dir, exist_ok=True)

    print(f'Copying {hostname}:{path} to {local_path}')
    os.system(f'scp {remote_path} {local_path}')
    return local_path


def rank0_print(*args, **kwargs):
    """Print, but only on rank 0.
    
    Example:
        >>> rank0_print("Hello from rank 0") # Only rank 0 will print this
        Hello from rank 0
    """
    if not dist.is_initialized() or dist.get_rank() == 0:
        logging.info(*args, **kwargs)


def get_local_dir(prefixes_to_resolve: List[str]) -> str:
    """Return the path to the cache directory for this user.
    
    Args:
        prefixes_to_resolve (List[str]): List of directory prefixes to check for existence.
        
    Returns:
        str: Path to the local cache directory for this user.
        
    Example:
        >>> get_local_dir(['/tmp/cache', '/var/cache'])
        '/tmp/cache/username'  # if /tmp/cache exists
        '/var/cache/username'  # if /tmp/cache does not exist but /var/cache does
    """
    for prefix in prefixes_to_resolve:
        if os.path.exists(prefix):
            return f"{prefix}/{getpass.getuser()}"
    os.makedirs(prefix)
    return f"{prefix}/{getpass.getuser()}"


def get_local_run_dir(exp_name: str, local_dirs: List[str]) -> str:
    """Create a local directory to store outputs for this run, and return its path.
    
    Args:
        exp_name (str): Name of the experiment.
        local_dirs (List[str]): List of directory prefixes to check for existence.
    Returns:
        str: Path to the created local run directory.
    Example:
        >>> run_dir = get_local_run_dir("my_experiment", ["/tmp/runs", "/var/runs"])
        >>> print(run_dir) # e.g., /tmp/runs/username/my_experiment_2023-10-05_12-30-45_123456
    """
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S_%f")
    run_dir = f"{get_local_dir(local_dirs)}/{exp_name}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def slice_and_move_batch_for_device(batch: Dict, rank: int, world_size: int, device: str) -> Dict:
    """Slice a batch into chunks, and move each chunk to the specified device.

    Raises:
        RuntimeError: If batch_size < world_size, which would cause some GPUs to receive empty data.
    """
    batch_size = len(list(batch.values())[0])

    # # Check if batch size is too small for the number of GPUs
    # if batch_size < world_size:
    #     raise RuntimeError(
    #         f"Batch size ({batch_size}) is smaller than world size ({world_size}). "
    #         f"This will cause some GPUs to receive empty data, leading to training errors. "
    #         f"Please ensure batch_size >= world_size (number of GPUs). "
    #         f"Solution: Increase batch_size in your training config to at least {world_size}."
    #     )

    chunk_size = batch_size // world_size
    start = chunk_size * rank
    end = chunk_size * (rank + 1)
    sliced = {k: v[start:end] for k, v in batch.items()}
    on_device = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in sliced.items()}
    return on_device


def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    ''' padding the tensor to the specified length on the given dimension, given the pad_value
    
    Args:
        tensor (torch.Tensor): input tensor to be padded
        length (int): desired length after padding
        pad_value (Union[int, float]): value to use for padding
        dim (int): dimension to pad (default: -1)
    Returns:
        torch.Tensor: padded tensor
        
    Example:
        >>> t = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> pad_to_length(t, 5, 0, dim=1)
        tensor([[1, 2, 3, 0, 0],
                [4, 5, 6, 0, 0]])   
    '''
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat([tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim)


def all_gather_if_needed(values: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """Gather and stack/cat values from all processes, if there are multiple processes."""
    if world_size == 1:
        return values

    all_values = [torch.empty_like(values).to(rank) for _ in range(world_size)]
    dist.all_gather(all_values, values)
    cat_function = torch.cat if values.dim() > 0 else torch.stack
    return cat_function(all_values, dim=0)


def formatted_dict(d: Dict) -> Dict:
    """Format a dictionary for printing.
    
    """
    return {k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in d.items()}


def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def print_gpu_memory(rank: int = None, message: str = ''):
    """Print the amount of GPU memory currently allocated for each GPU."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            device = torch.device(f'cuda:{i}')
            allocated_bytes = torch.cuda.memory_allocated(device)
            if allocated_bytes == 0:
                continue
            print('*' * 40)
            print(f'[{message} rank {rank} ] GPU {i}: {allocated_bytes / 1024 ** 2:.2f} MB')
        print('*' * 40)


def get_block_class_from_model(model: torch.nn.Module, block_class_name: str) -> torch.nn.Module:
    """Get the class of a block from a model, using the block's class name."""
    for module in model.modules():
        if module.__class__.__name__ == block_class_name:
            return module.__class__
    raise ValueError(f"Could not find block class {block_class_name} in model {model}")


def get_block_class_from_model_class_and_block_name(model_class: Type, block_class_name: str) -> Type:
    filepath = inspect.getfile(model_class)
    assert filepath.endswith('.py'), f"Expected a .py file, got {filepath}"
    assert os.path.exists(filepath), f"File {filepath} does not exist"
    assert "transformers" in filepath, f"Expected a transformers model, got {filepath}"

    module_name = filepath[filepath.find('transformers'):].replace('/', '.')[:-3]
    print(f"Searching in file {filepath}, module {module_name} for class {block_class_name}")

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the class dynamically
    class_ = getattr(module, block_class_name)
    print(f"Found class {class_} in module {module_name}")
    return class_


def init_distributed(rank: int, world_size: int, master_addr: str = 'localhost', port: int = 12355, backend: str = 'nccl'):
    print(rank, 'initializing distributed')
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


