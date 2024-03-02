from torchness.tbwr import TBwr
from typing import List, Optional, Dict


class TBwr_DMK(TBwr):
    """ it is a customized (for DMK) version of TBwr that supports loop_stats """

    def __init__(self, collect_loop_stats:bool, **kwargs):
        super().__init__(**kwargs)
        self.collect_loop_stats = collect_loop_stats
        self.loop_stats: Dict[str, List] = {}

    def add_force(self, *args, **kwargs):
        super().add(*args, **kwargs)

    def add(self,
            value,
            tag: str,
            step: Optional[int]=    None):
        if self.collect_loop_stats:
            if tag not in self.loop_stats:
                self.loop_stats[tag] = []
            self.loop_stats[tag].append(value)
        else:
            self.add_force(value=value, tag=tag, step=step)

    def add_histogram(self, *args, **kwargs):
        if not self.collect_loop_stats:
            super().add_histogram(*args, **kwargs)

    def add_text(self, *args, **kwargs):
        if not self.collect_loop_stats:
            super().add_text(*args, **kwargs)

    def publish_loop_stats(self, step):
        for k in self.loop_stats:
            val = sum(self.loop_stats[k]) / len(self.loop_stats[k])
            self.add_force(value=val, tag=k, step=step)
        self.loop_stats = {}