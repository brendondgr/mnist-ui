# Imports
import torch

# Variables
no_gpu = False

print(f'#############################################')
print(f'## GPU INFORMATION ##')

# Print the Version of Torch/CUDA
print(f'Torch Version: {torch.__version__}')

# Check if CUDA is available
is_avail = torch.cuda.is_available()
print(f'Is CUDA Available: {is_avail}')

if is_avail:
    # Set Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Print the Device.
    try:
        print(f'Current Device: {torch.cuda.get_device_name()}')
    except:
        no_gpu = True
        print('No GPU Available')

    # Print the Number of GPUs
    if not no_gpu:
        try:
            print(f'Number of GPUs: {torch.cuda.device_count()}')
        except:
            print('No GPU Available')
    
        # Print the Total Amount of VRAM:
        try:
            total_vram = torch.cuda.get_device_properties(0).total_memory
            print(f'Total VRAM: {total_vram / (1024 ** 3):.2f} GB')
        except:
            print('No GPU Available')
        
        # Print the Current VRAM Usage
        try:
            print(f'Current VRAM Usage: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB')
        except:
            print('No GPU Available')
        
        # Unavailable VRAM
        try:
            total_vram = torch.cuda.get_device_properties(device).total_memory
            reserved_vram = torch.cuda.memory_reserved(device)
            unavailable_vram = total_vram - reserved_vram
            print(f"Unavailable VRAM: {unavailable_vram / (1024 ** 3):.2f} GB")
            
            # If unavailable_vram is within 5% of total_vram, then run empty_cache()
            if unavailable_vram / total_vram > 0.05:
                torch.cuda.empty_cache()
                print("Cleared VRAM")
            
        except:
            print('No GPU Available')

print(f'#############################################\n')