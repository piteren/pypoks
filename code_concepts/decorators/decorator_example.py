"""

 2020 (c) piteren

 https://python-3-patterns-idioms-test.readthedocs.io/en/latest/PythonDecorators.html

"""

class my_decorator(object):

    def __init__(self, f): # constructor is executed only once, at the point of decoration of the function
        print("inside my_decorator.__init__()")
        self.f = f
        self.f('init') # Prove that function creation is complete (..but not finally decorated)

    def __call__(self, *args): # decorator as a class must implement __call__
        print("inside my_decorator.__call__()")
        self.f(*args)

@my_decorator
def a_fun(str):
    print(f"aFunction({str})")

print("Finished decorating aFunction()")

a_fun('piotrek')
