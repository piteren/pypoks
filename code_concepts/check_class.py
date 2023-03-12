class AnyThing:

    def __init__(self, num:int):
        self.num = num

    def gx(self, ob: "AnyThing") -> "AnyThing":
        return AnyThing(self.num + ob.num)

    def __str__(self):
        return str(f'{type(self)}: {self.num}')

oa = AnyThing(1)
ob = AnyThing(2)
oc = oa.gx(ob)
print(oc)