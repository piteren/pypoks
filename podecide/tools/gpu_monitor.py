import GPUtil
from pypaq.mpython.mptools import ExSubprocess, Que, QMessage
import time
from torchness.tbwr import TBwr
from typing import Dict, Optional

from envy import DMK_MODELS_FD


class GPUMonitor(ExSubprocess):
    """ Monitors GPU usage """

    def __init__(
            self,
            pause: float=           0.3,
            tb_name: Optional[str]= None,
            **kwargs):

        ExSubprocess.__init__(self, ique=Que(), oque=Que(), **kwargs)

        self.pause = pause
        self.tb_name = tb_name
        self._tb_counter = 0

        devs = list(GPUtil.getGPUs())
        self.dev_keys = [d.id                for d in devs]
        self.proc =     {d.id: []            for d in devs}
        self.mem =      {d.id: []            for d in devs}
        self.mem_size = {d.id: d.memoryTotal for d in devs}

        self.logger.info(f'*** GPUMonitor *** initialized for {len(devs)} GPUs')
        self.start()

    def subprocess_method(self):

        self.logger.debug(f'GPUMonitor stats process loop')

        tbwr = TBwr(logdir=f'{DMK_MODELS_FD}/{self.tb_name}') if self.tb_name else None

        while True:

            time.sleep(self.pause)

            msg = self.ique.get(block=False)

            if msg:

                if msg.type == 'stop':
                    break

                if msg.type == 'send_report':

                    report = self._prep_report()
                    msg = QMessage(type='report', data=report)
                    self.oque.put(msg)

                    if tbwr:
                        for k in report:
                            tbwr.add(value=report[k], tag=f'GPUMonitor/{k}', step=self._tb_counter)
                        self._tb_counter += 1

            devs = GPUtil.getGPUs()
            for d in devs:
                self.proc[d.id].append(d.load*100)
                self.mem[d.id].append(d.memoryUsed)

        self.logger.debug(f'UpdSync stopped process loop')

    def _prep_report(self) -> Dict[str,float]:
        n = len(self.proc[self.dev_keys[0]])
        if n:
            report = {}
            for d in self.proc:
                report[f'proc_avg_dev{d}'] = sum(self.proc[d]) / n
                report[f'mem_avg_dev{d}'] = sum(self.mem[d]) / n / self.mem_size[d]
                report[f'mem_max_dev{d}'] = max(self.mem[d]) / self.mem_size[d]
            for d in self.proc:
                self.proc[d] = []
                self.mem[d] = []
            return report
        return {}

    def get_report(self) -> Dict[str,float]:
        msg = QMessage(type='send_report', data=None)
        self.ique.put(msg)
        return self.oque.get().data

    def stop(self):
        msg = QMessage(type='stop', data=None)
        self.ique.put(msg)