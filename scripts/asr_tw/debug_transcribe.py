import os
import debugpy
import torch
import matplotlib.pyplot as plt
import numpy as np


from examples.asr.transcribe_speech import main

def setup_debugging():
    # Ensure debugpy is only started on the main process (rank=0)
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0  # Default to main process in single GPU or non-distributed scenarios

    if rank == 0:
        # debugpy.configure(subProcess=True)
        # Set the port number for debugpy
        debugpy.listen(("0.0.0.0", 5678))
        print("Waiting for debugger to attach...")
        debugpy.wait_for_client()  # Wait for the debugger to connect
        print("Debugger attached!")

def plot_attn(avg_attn):
    # just a note for plotting
    plt.imshow(avg_attn, cmap='hot', aspect='auto')
    plt.colorbar()
    plt.title('attention Heatmap')
    plt.savefig("nemo_experiments/attnmap.png")
    plt.show()
    plt.close()

# 在訓練腳本中調用
if __name__ == "__main__":
    # Initialize distributed environment (if needed)
    if "LOCAL_RANK" in os.environ:
        torch.distributed.init_process_group(backend="nccl")

    # Set up debugpy
    setup_debugging()

    main()