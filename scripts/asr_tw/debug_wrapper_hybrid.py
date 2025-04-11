import os
import debugpy
import torch

from examples.asr.asr_hybrid_transducer_ctc.speech_to_text_hybrid_rnnt_ctc_bpe import main

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

# 在訓練腳本中調用
if __name__ == "__main__":
    # Initialize distributed environment (if needed)
    if "LOCAL_RANK" in os.environ:
        torch.distributed.init_process_group(backend="nccl")

    # Set up debugpy
    setup_debugging()

    main()