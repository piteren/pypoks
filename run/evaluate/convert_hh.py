from pypaq.lipytools.files import list_dir, r_json

from envy import TR_RESULTS_FP
from pologic.hand_history import HHistory, states2HHtexts


if __name__ == "__main__":

    hand_dir = 'hg_hands'
    hh_files = sorted([fn for fn in list_dir(hand_dir)['files'] if fn.endswith('jsonl')])

    loop_results = r_json(TR_RESULTS_FP)
    key = 'refs_ranked' if 'refs_ranked' in loop_results else 'dmk_ranked'
    dmks_ranked = loop_results[key]
    dmk_agent_name = dmks_ranked[0]

    for fn in hh_files:
        hh = HHistory.from_file(f'{hand_dir}/{fn}')
        texts = states2HHtexts(
            states=     hh.events,
            add_probs=  True,
            rename=     {dmk_agent_name:'agent'},
        )

        with open(f'{hand_dir}/{fn[:-5]}txt', 'w') as file:
            file.write('\n'.join(texts))