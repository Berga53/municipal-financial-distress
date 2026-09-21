"""Small shared helpers."""

import time


def print_one_line(msg):
    time.sleep(0.01)
    print(msg.ljust(120), end='\r', flush=True)
