# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import time
import torch


""" This module contains utility function and classes for measuring latency / memory. """


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def colorize_time(elapsed):
    """Returns colors based on the significance of the time elapsed.
    """
    if elapsed > 1e-3:
        return bcolors.FAIL + "{:.3e}".format(elapsed) + bcolors.ENDC
    elif elapsed > 1e-4:
        return bcolors.WARNING + "{:.3e}".format(elapsed) + bcolors.ENDC
    elif elapsed > 1e-5:
        return bcolors.OKBLUE + "{:.3e}".format(elapsed) + bcolors.ENDC
    else:
        return "{:.3e}".format(elapsed)

def print_gpu_memory():
    """Prints GPU memory used.
    """
    torch.cuda.empty_cache()
    print(f"{torch.cuda.memory_allocated()//(1024*1024)} mb")


class PerfTimer():
    """Super simple performance timer.
    """
    def __init__(self, activate=False, show_memory=False, print_mode=True):
        self.prev_time = time.process_time()
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.prev_time_gpu = self.start.record()
        self.counter = 0
        self.activate = activate
        self.show_memory = show_memory
        self.print_mode = print_mode

    def reset(self):
        self.counter = 0
        self.prev_time = time.process_time()
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.prev_time_gpu = self.start.record()

    def check(self, name=None):
        if self.activate:
            cpu_time = time.process_time() - self.prev_time
          
            self.end.record()
            torch.cuda.synchronize()

            gpu_time = self.start.elapsed_time(self.end) / 1e3
            
            if self.print_mode:
                cpu_time_disp = colorize_time(cpu_time)
                gpu_time_disp = colorize_time(gpu_time)
                if name:
                    print("CPU Checkpoint {}: {} s".format(name, cpu_time_disp))
                    print("GPU Checkpoint {}: {} s".format(name, gpu_time_disp))
                else:
                    print("CPU Checkpoint {}: {} s".format(self.counter, cpu_time_disp))
                    print("GPU Checkpoint {}: {} s".format(self.counter, gpu_time_disp))
                if self.show_memory:
                    print(f"{torch.cuda.memory_allocated()//1048576}MB")
                

            self.prev_time = time.process_time()
            self.prev_time_gpu = self.start.record()
            self.counter += 1
            return cpu_time, gpu_time


