from pypaq.lipytools.files import list_dir

from pologic.game_config import GameConfig
from pologic.potable import PTable


if __name__ == "__main__":

    test_dir = 'hg_hands'

    hh_files = list_dir(test_dir)['files']
    hh_files = sorted([fn for fn in hh_files if fn.endswith('.txt')])
    print(f'got {len(hh_files)} HH files')

    for hh_file in hh_files:

        print(f'checking {hh_file}..')

        with open(f'{test_dir}/{hh_file}') as file:
            hh_strL = [l[:-1] for l in file]

        table = PTable(
            name=           'test_table',
            game_config=    GameConfig.from_name(name=hh_strL[0].split()[1]),
            pl_ids=         [l.split()[1] for l in hh_strL if l.startswith('POS:')],
            loglevel=       30)
        table.run_hand(hh_strL)