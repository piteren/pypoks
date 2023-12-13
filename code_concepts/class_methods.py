class A:

    @classmethod
    def ma(cls):
        print('ma of A')

    def mc(self):
        A.ma()
        self.ma()

class B(A):

    @classmethod
    def ma(cls):
        print('ma of B')


b = B()
b.mc()
