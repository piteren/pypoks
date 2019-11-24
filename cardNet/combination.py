"""

 2019 (c) piteren

"""

import itertools

cardList = [x for x in range(52)]
combList = list(itertools.combinations(cardList,5))

print(len(combList))