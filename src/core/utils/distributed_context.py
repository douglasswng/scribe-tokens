import os
import signal
import atexit

import torch
import torch.distributed as dist


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def clean_exit(sig, frame):
    print(f"Process {os.getpid()} gracefully exiting...")
    cleanup_distributed()
    exit(0)

  
signal.signal(signal.SIGINT, clean_exit)
signal.signal(signal.SIGTERM, clean_exit)
atexit.register(cleanup_distributed)


class DistributedContext:
    def __init__(self):
        if not self.is_distributed:
            return
        
        if dist.is_initialized():
            return
        
        if self.cuda_available:
            torch.cuda.set_device(self.local_rank)

        device_id = torch.device(f"cuda:{self.local_rank}") if self.cuda_available else None
        dist.init_process_group(backend=self.backend, device_id=device_id)

        if self.is_master:
            print(f"World Size: {self.world_size}")

    @property
    def is_distributed(self) -> bool:
        return "RANK" in os.environ and "LOCAL_RANK" in os.environ

    @property
    def rank(self) -> int:
        return dist.get_rank() if self.is_distributed else 0

    @property
    def world_size(self) -> int:
        return dist.get_world_size() if self.is_distributed else 1

    @property
    def local_rank(self) -> int:
        return int(os.environ["LOCAL_RANK"]) if self.is_distributed else 0
    
    @property
    def device(self) -> str:
        return f"cuda:{self.local_rank}" if self.cuda_available else "cpu"
    
    @property
    def device_ids(self) -> list[int] | None:
        return [self.local_rank] if self.device.startswith("cuda") else None
    
    @property
    def is_master(self) -> bool:
        return self.rank == 0
    
    @property
    def is_worker(self) -> bool:
        return not self.is_master
    
    @property
    def backend(self) -> str:
        return "nccl" if self.cuda_available else "gloo"
    
    @property
    def cuda_available(self) -> bool:
        return torch.cuda.is_available()
    
    def divide_by_world_size(self, value: int) -> int:
        return value // self.world_size
    
    def barrier(self) -> None:
        if self.is_distributed:
            dist.barrier()
    

distributed_context = DistributedContext()
    

if __name__ == "__main__":
    print(distributed_context.is_master)
    print(distributed_context.rank)
    print(distributed_context.world_size)
    print(distributed_context.local_rank)
    print(distributed_context.device)
    print(distributed_context.backend)