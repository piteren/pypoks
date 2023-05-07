from pypaq.lipytools.files import r_json, w_json

all_results = {0: 'a', 1:'b'}
w_json(all_results, 'test.json')

all_results = r_json('test.json')
print(all_results)
# keys are str type now