"""

 2020 (c) piteren

"""

class A:
    def __init__(self):
        print('.a init')
        pass

class B:
    def __init__(self):
        print('.b init')
        pass

class C(A,B): # python looks for parents left to right
    def __init__(self):
        print('C init')
        super().__init__()

class D(B,A):
    def __init__(self):
        print('D init')
        super().__init__()

class E(A,B):
    def __init__(self):
        print('E init')
        A.__init__(self)
        B.__init__(self)


oc = C()
od = D()
oe = E()


class F:
    def m(self):
        print("m of F called")

class G(F):
    def m(self):
        print("m of G called")

class H(F):
    def m(self):
        print("m of H called")

class I(G,H):
    def m(self):
        print("m of I called")

x = I()
G.m(x)