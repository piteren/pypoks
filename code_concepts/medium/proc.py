from multiprocessing import Process
from functools import partial

def proc(f):
    def new_f(*args, **kwargs):
        Process(target=partial(f, *args, **kwargs)).start()
    return new_f