import time
import unittest

from podecide.tools.devices_monitor import DEVMonitor


class TestDEVMonitor(unittest.TestCase):

    def test_base(self):
        gpu_monitor = DEVMonitor(loglevel=10)
        for _ in range(5):
            time.sleep(1)
            print(gpu_monitor.get_report())
        gpu_monitor.stop()