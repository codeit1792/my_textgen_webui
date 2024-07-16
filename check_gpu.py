import torch

def test_cuda():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        
        # Create a tensor and move it to the GPU
        x = torch.rand(3, 3).to(device)
        print(f"Tensor on GPU: {x}")

        # Perform a simple GPU operation
        y = torch.matmul(x, x)
        print(f"Result of tensor multiplication on GPU: {y}")
    else:
        print("CUDA is not available. Cannot perform GPU operations.")

if __name__ == "__main__":
    test_cuda()
