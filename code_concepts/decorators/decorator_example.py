"""

 2020 (c) piteren

"""

class my_decorator(object):

    def __init__(self, f):
        print("inside my_decorator.__init__()")
        self.f = f
        self.f('init') # Prove that function definition has completed

    def __call__(self):
        print("inside my_decorator.__call__()")
        self.f('call')

@my_decorator
def aFunction(str):
    print(f"aFunction({str})")

print("Finished decorating aFunction()")

aFunction()
