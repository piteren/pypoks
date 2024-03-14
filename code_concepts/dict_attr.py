"""

 2020 (c) piteren

"""

class TE:

    def __init__(self):
        self.pa = 3
        self.pb = 'text'

    def __str__(self):
        return f'pa:{self.pa} pb:{self.pb}'

t = TE()
print(t.__dict__)
t.__dict__['pa'] = 4
print(t.__dict__)
print(t)