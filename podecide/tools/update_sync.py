from pypaq.mpython.mptools import ExSubprocess, Que, QMessage
import time
from torchness.tbwr import TBwr
from typing import List, Dict, Optional

from envy import DMK_MODELS_FD
from podecide.dmk import NeurDMK


class UpdSync(ExSubprocess):
    """ Update Synchronizer for DMK
    build by GM while starting a game """

    def __init__(
            self,
            dmkL:List[NeurDMK],
            tb_name: Optional[str]= None,
            **kwargs):

        ExSubprocess.__init__(self, ique=Que(), **kwargs)

        self.dmk_device = {dmk.name: dmk.device for dmk in dmkL}        # {dmk_name: device}
        devices = set(self.dmk_device.values())
        self.oqueD: Dict[str,Que] = {dmk.name: Que() for dmk in dmkL}   # here UpdSync puts ticket for waiting DMK
        self.ticket = {d: True for d in devices}                        # tickets, per device
        self.dmks_waiting_for_ticket = {d: [] for d in devices}         # DMK waiting list per device

        # activate DMKs for requested updates
        for dmk in dmkL:
            dmk.set_upd_sync(
                que_out= self.ique,                                     # here DMKs put requests and return tickets
                que_in=  self.oqueD[dmk.name])

        self.tb_name = tb_name
        self._stime = {d: None for d in devices}
        self._stime_log = {d: {'update':[], 'idle':[]} for d in devices}
        self._tb_counter = {d: 0 for d in devices}
        self._tb_freq = 10

        self.logger.info(f'*** UpdSync *** initialized and started for {len(dmkL)} DMKs')
        self.start()

    def subprocess_method(self):

        self.logger.debug(f'UpdSync stats process loop')

        tbwr = TBwr(logdir=f'{DMK_MODELS_FD}/{self.tb_name}') if self.tb_name else None

        while True:

            msg = self.ique.get()

            if msg.type == 'stop':
                break

            # received ticket from DMK
            if msg.type == 'ticket':

                dev = self.dmk_device[msg.data]
                self.ticket[dev] = True
                self.logger.debug(f'UpdSync received ticket from {msg.data}')

                ctime = time.time()
                if self._stime[dev] is not None:
                    self._stime_log[dev]['update'].append(ctime - self._stime[dev])
                self._stime[dev] = ctime

            # received update request from DMK
            if msg.type == 'update_request':

                dev = self.dmk_device[msg.data]
                self.dmks_waiting_for_ticket[dev].append(msg.data)
                self.logger.debug(f'UpdSync received ticket request for dev:{dev} from {msg.data}, dmks_waiting: {self.dmks_waiting_for_ticket}')

            for dev in self.ticket:
                if self.ticket[dev] and self.dmks_waiting_for_ticket[dev]:

                    dmk_name = self.dmks_waiting_for_ticket[dev].pop(0)
                    msg = QMessage(type='ticket', data=None)
                    self.oqueD[dmk_name].put(msg)
                    self.ticket[dev] = False
                    self.logger.debug(f'UpdSync sent dev:{dev} ticket to {dmk_name}')

                    ctime = time.time()
                    if self._stime[dev] is not None:
                        self._stime_log[dev]['idle'].append(ctime - self._stime[dev])
                    self._stime[dev] = ctime

                    if len(self._stime_log[dev]['idle']) == self._tb_freq:

                        if tbwr:

                            idl = sum(self._stime_log[dev]['idle']) / self._tb_freq
                            upd = sum(self._stime_log[dev]['update']) / self._tb_freq
                            all = idl + upd
                            tbwr.add(value=idl,     tag=f'UpdSync/GPU{dev}_time_idle',  step=self._tb_counter[dev])
                            tbwr.add(value=upd,     tag=f'UpdSync/GPU{dev}_time_upd',   step=self._tb_counter[dev])
                            tbwr.add(value=upd/all, tag=f'UpdSync/GPU{dev}_factor_upd', step=self._tb_counter[dev])

                            waiting = len(self.dmks_waiting_for_ticket[dev])
                            tbwr.add(value=waiting, tag=f'UpdSync/GPU{dev}_waiting',    step=self._tb_counter[dev])

                            self._tb_counter[dev] += 1

                        self._stime_log[dev]['idle'] = []
                        self._stime_log[dev]['update'] = []

        self.logger.debug(f'UpdSync stopped process loop')

    def stop(self):
        msg = QMessage(type='stop', data=None)
        self.ique.put(msg)