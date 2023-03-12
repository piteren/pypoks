from ptools.lipytools.decorators import autoinit
from ptools.pms.subscriptable import Subscriptable


class SomeClass(Subscriptable):

    @autoinit
    def __init__(self, a, b, c:int=1, d:int=2):
        #self.a = 'replaced'
        self.dna = {}

    def __getitem__(self, key):
        return self.dna[key]

    def __setitem__(self, key, value):
        self.dna[key] = value

    def __contains__(self, key):
        return key in self.dna


so = SomeClass(5,6,7)
print(vars(so))

so = SomeClass(5,6,d=7)
print(vars(so))

so['ala'] = 9
print(so['ala'])
print('ala' in so)

print(vars(so))