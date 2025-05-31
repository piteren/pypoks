import numpy as np
from pypaq.lipytools.plots import histogram

from pologic.podeck import PDeck
from podecide.cardNet.cardNet_module import CardNet_MOTorch
from podecide.cardNet.cardNet_batcher import get_test_batch



if __name__ == "__main__":

    # CardNet_MOTorch.copy_saved('cn1210_1335','cardNet12')

    card_net = CardNet_MOTorch(
        #name='cn1210_1159',
        cards_emb_width=24, device=-1, loglevel=10)

    test_batch, _ = get_test_batch(batch_size=2000, n_monte=10000000)
    test_batch_conv = {k: card_net.convert(data=test_batch[k]) for k in test_batch}

    out = card_net.loss(**test_batch_conv)

    deck = PDeck()

    res = []
    for cardsA, pred, true in zip(
            test_batch['cards_A'],
            out['reg_won_A'],
            test_batch['prob_won_A']):
        cardsA = [c for c in cardsA if c != 52] # remove pad card
        cardsA_str = ' '.join([PDeck.cts(c) for c in cardsA])
        pred = float(pred)
        true = float(true)
        diff = abs(pred-true)
        res.append((cardsA_str, pred, true, diff))

    res.sort(key=lambda x:x[-1], reverse=True)
    for ix,r in enumerate(res):
        print(f'{ix:4} {r[0]:21} {r[1]:.3f} {r[2]:.3f} {r[3]:.6f}')

    diffs = [e[-1] for e in res]
    print(histogram(diffs, name='diffs'))

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

