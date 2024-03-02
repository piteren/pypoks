import GPUtil
from pypaq.mpython.mptools import ExSubprocess, Que, QMessage, sys_res_nfo
import time
from torchness.tbwr import TBwr
from typing import Dict, Optional

from envy import DMK_MODELS_FD


class DEVMonitor(ExSubprocess):
    """ Monitors Devices (CPU + GPU) usage """

    def __init__(
            self,
            pause: float=           0.3,
            tb_name: Optional[str]= None,
            **kwargs):

        ExSubprocess.__init__(self, ique=Que(), oque=Que(), **kwargs)

        self.pause = pause
        self.tb_name = tb_name
        self._tb_counter = 0

        self.cpu_proc = []
        self.cpu_mem = []

        gpu_devs = list(GPUtil.getGPUs())
        self.gpu_dev_keys = [d.id for d in gpu_devs]
        self.gpu_proc =     {d.id: [] for d in gpu_devs}
        self.gpu_mem =      {d.id: [] for d in gpu_devs}
        self.gpu_mem_size = {d.id: d.memoryTotal for d in gpu_devs}

        self.logger.info(f'*** {self.__class__.__name__} *** initialized for {len(gpu_devs)} GPUs')
        self.start()

    def subprocess_method(self):

        self.logger.debug(f'{self.__class__.__name__} stats process loop')

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
                            tbwr.add(value=report[k], tag=f'{self.__class__.__name__}/{k}', step=self._tb_counter)
                        self._tb_counter += 1

            devs = GPUtil.getGPUs()
            for d in devs:
                self.gpu_proc[d.id].append(d.load * 100)
                self.gpu_mem[d.id].append(d.memoryUsed * 100)

            srn = sys_res_nfo()
            self.cpu_proc.append(srn['cpu_used_%'])
            self.cpu_mem.append(srn['mem_used_%'])

        self.logger.debug(f'UpdSync stopped process loop')

    def _prep_report(self) -> Dict[str,float]:
        n = len(self.gpu_proc[self.gpu_dev_keys[0]])
        if n:

            report = {
                'CPU_proc': sum(self.cpu_proc) / n,
                'CPU_mem':  sum(self.cpu_mem) / n}

            for d in self.gpu_proc:
                report[f'GPU{d}_proc'] = sum(self.gpu_proc[d]) / n
                report[f'GPU{d}_mem'] = sum(self.gpu_mem[d]) / n / self.gpu_mem_size[d]
                report[f'GPU{d}_mem_max'] = max(self.gpu_mem[d]) / self.gpu_mem_size[d]

            self.cpu_proc = []
            self.cpu_mem = []
            for d in self.gpu_proc:
                self.gpu_proc[d] = []
                self.gpu_mem[d] = []

            return report
        return {}

    def get_report(self) -> Dict[str,float]:
        msg = QMessage(type='send_report', data=None)
        self.ique.put(msg)
        return self.oque.get().data

    def stop(self):
        msg = QMessage(type='stop', data=None)
        self.ique.put(msg)