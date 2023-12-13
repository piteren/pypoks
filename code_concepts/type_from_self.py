class A:

    def __init__(self):

        self.some_type = int
        self.obj: self.some_type = 1
        print(type(self.obj))

a = A()