import numpy as np
from pologic.podeck import PDeck
from podecide.cardNet.cardNet_module import CardNet_MOTorch
from podecide.cardNet.cardNet_batcher import get_test_batch



if __name__ == "__main__":

    #CardNet_MOTorch.copy_saved('cn1210_1335','cardNet12')

    card_net = CardNet_MOTorch(
        #name=               'cn1210_1159',
        cards_emb_width=    12,
        device=             -1,
        loglevel=           10,
    )

    test_batch, _ = get_test_batch(batch_size=2000, n_monte=10000000)
    test_batch_conv = {k: card_net.convert(data=test_batch[k]) for k in test_batch}

    #print(test_batch.keys()) # ['cards_A', 'cards_B', 'label_won', 'label_rank_A', 'label_rank_B', 'prob_won_A']

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
        res.append({
            #'cardsA':       cardsA,
            'cardsA_str':   cardsA_str,
            'true':         true,
            'pred':         pred,
            'diff':         diff,
        })

    res.sort(key=lambda x:x['diff'], reverse=True)
    for r in res:
        print(f'{r["cardsA_str"]:21} {r["pred"]:.3f} {r["true"]:.3f} {r["diff"]:.3f}')

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
                print(pred, lA, cardsA)

        pred = int(np.argmax(logits_rankB.detach().cpu().numpy()))
        if pred != lB:
            print(pred, lB, cardsB)

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
                print(pred, lW, cardsA, cardsB)

