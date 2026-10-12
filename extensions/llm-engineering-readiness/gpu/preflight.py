"""Check local CUDA environment before running GPU-only optional demos."""
def require_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this optional demo; CPU CI cannot verify it")
    return torch.cuda.get_device_name(0)
if __name__=="__main__":
    print(require_gpu())
