import numpy as np
from pypaq.lipytools.plots import histogram
from pypaq.lipytools.files import r_json

from pologic.podeck import PDeck
from podecide.cardNet.cardNet_module import CardNet_MOTorch
from podecide.cardNet.cardNet_batcher import get_test_batch



if __name__ == "__main__":

    #CardNet_MOTorch.copy_saved('cn96_0601_2044','cardNet96')

    card_net = CardNet_MOTorch(
        name='cn96_0601_2044',
        #cards_emb_width=24,
        device=-1, loglevel=10)

    test_batch, _ = get_test_batch(batch_size=2000, n_monte=10000000)
    test_batch_conv = {k: card_net.convert(data=test_batch[k]) for k in test_batch}
    eq_py_test = r_json('_cache/eq_py_test.json')

    out = card_net.loss(**test_batch_conv)

    deck = PDeck()

    res = []
    for cardsA, eq_pred, eq_mc in zip(
            test_batch['cards_A'],
            out['reg_won_A'],
            test_batch['prob_won_A']):
        cardsA = [c for c in cardsA if c != 52] # remove pad card
        cardsA_str = ''.join([PDeck.cts(c) for c in cardsA])
        eq_pred = float(eq_pred)
        eq_mc = float(eq_mc)
        eq_py = eq_py_test[cardsA_str]
        res.append((cardsA_str, eq_pred, eq_mc, eq_py))

    res.sort(key=lambda x:abs(x[1]-x[3]), reverse=True)
    d_p_mc =  [abs(e[1]-e[2]) for e in res]
    d_p_py =  [abs(e[1]-e[3]) for e in res]
    d_mc_py = [abs(e[2]-e[3]) for e in res]
    print(histogram(d_p_mc, name='dEQ_pred_mc'))
    print(histogram(d_p_py, name='dEQ_pred_py'))
    print(histogram(d_mc_py, name='dEQ_mc_py'))

    for ix,(r,d) in enumerate(zip(res,d_p_py)):
        print(f'{ix:4} {r[0]:21} {r[1]:.3f} {r[2]:.3f} {r[3]:.3f}[{d:.6f}]')

    is_ok = True
    print('checking rank pred ..')
    for cardsA, cardsB, logits_rankA, logits_rankB, lA, lB in zip(
            test_batch['cards_A'],
            test_batch['cards_B'],
            out['logits_rank_A'],
            out['logits_rank_B'],
            test_batch['label_rank_A'],
            test_batch['label_rank_B']):

        cardsA = [c for c in cardsA if c != 52]  # remove pad card

        if len(cardsA) == 7:
            pred = int(np.argmax(logits_rankA.detach().cpu().numpy()))
            if pred != lA:
                is_ok = False
                print(pred, lA, cardsA)

        pred = int(np.argmax(logits_rankB.detach().cpu().numpy()))
        if pred != lB:
            is_ok = False
            print(pred, lB, cardsB)
    if is_ok:
        print('Ok!')

    is_ok = True
    print('checking won pred ..')
    for cardsA, cardsB, logits_winner, lW in zip(
            test_batch['cards_A'],
            test_batch['cards_B'],
            out['logits_winner'],
            test_batch['label_won']):

        cardsA = [c for c in cardsA if c != 52]  # remove pad card

        if len(cardsA) == 7:
            pred = int(np.argmax(logits_winner.detach().cpu().numpy()))
            if pred != lW:
                is_ok = False
                print(pred, lW, cardsA, cardsB)
    if is_ok:
        print('Ok!')