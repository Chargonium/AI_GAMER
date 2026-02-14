import torch

def main():
    print("PyTorch version:", torch.__version__)
    print()

    # CPU (always available)
    print("CPU available: True")
    print("CPU device:", torch.device("cpu"))
    print()

    # CUDA (NVIDIA GPUs)
    cuda_available = torch.cuda.is_available()
    print("CUDA available:", cuda_available)

    if cuda_available:
        device_count = torch.cuda.device_count()
        print("CUDA device count:", device_count)

        for i in range(device_count):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Capability: {torch.cuda.get_device_capability(i)}")
            print(f"    Total memory (GB): "
                  f"{torch.cuda.get_device_properties(i).total_memory / 1e9:.2f}")
    print()

    # Apple Metal (MPS)
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print("MPS available:", mps_available)

    if mps_available:
        print("MPS device:", torch.device("mps"))

if __name__ == "__main__":
    main()
