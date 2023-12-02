import time
import unittest

from podecide.tools.gpu_monitor import GPUMonitor


class TestGPUMonitor(unittest.TestCase):

    def test_base(self):
        gpu_monitor = GPUMonitor(loglevel=10)
        for _ in range(5):
            time.sleep(1)
            print(gpu_monitor.get_report())
        gpu_monitor.stop()