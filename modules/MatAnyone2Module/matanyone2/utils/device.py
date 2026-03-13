import torch
import functools

def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def safe_autocast_decorator(enabled=True):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            device = get_default_device()
            if device.type in ["cuda", "cpu"]:
                with torch.amp.autocast(device_type=device.type, enabled=enabled):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator

import contextlib
@contextlib.contextmanager
def safe_autocast(enabled=True):
    device = get_default_device()
    if device.type in ["cuda", "cpu"]:
        with torch.amp.autocast(device_type=device.type, enabled=enabled):
            yield
    else:
        yield  # MPS or other unsupported backends skip autocast
