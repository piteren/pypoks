import torch
from torchness.types import TNS, DTNS, ACT
from torchness.base_elements import my_initializer
from torchness.layers import TF_Dropout, LayDense
from torchness.encoders import EncTNS, EncDRT
from torchness.motorch import Module, MOTorch
from typing import Optional

from envy import CN_MODELS_FD, get_cardNet_name


class CardEnc(Module):

    def __init__(
            self,
            emb_width: int=         24,             # cards embedding width, makes encoder width x7
            time_drop: float=       0.0,
            feat_drop: float=       0.0,
            in_proj: Optional[int]= None,
            n_layers: int=          8,
            dense_mul: int=         4,              # transformer dense multiplication
            dropout: float=         0.0,            # transformer dropout
            activation: ACT=        torch.nn.ReLU):

        Module.__init__(self)

        self.emb_width = emb_width

        self.cards_emb = torch.nn.Parameter(torch.empty(size=(53,self.emb_width))) # 52 + one card for 'no_card'
        my_initializer(self.cards_emb)

        self.mycards_emb = torch.nn.Parameter(torch.empty(size=(2,self.emb_width)))# my cards
        my_initializer(self.mycards_emb)

        self.tf_drop = TF_Dropout(
            time_drop=  time_drop,
            feat_drop=  feat_drop)

        self.in_proj = LayDense(
            in_features=    self.emb_width,
            out_features=   in_proj,
            activation=     None,
            bias=           False,
            initializer=    my_initializer) if in_proj else None

        self.enc_tns = EncTNS(
            num_layers=     n_layers,
            num_layers_TAT= 0,
            d_model=        self.emb_width if not in_proj else in_proj,
            nhead=          1,
            dns_scale=      dense_mul,
            dropout=        dropout,
            activation=     activation)

    def forward(
            self,
            cards: TNS # seven cards (ids)
    ) -> DTNS:

        my_cards_indexes = [0,0,1,1,1,1,1] # adds info which cards are mine (EncTNS does not recognize order)

        input = self.cards_emb[cards] + self.mycards_emb[my_cards_indexes]
        input = self.tf_drop(input)

        if self.in_proj: input = self.in_proj(input)

        enc_out = self.enc_tns(input)
        output = enc_out['out']
        zsL = enc_out['zsL']

        output = output.view(list(output.shape)[:-2] + [-1]) # flatten last two dim

        return {
            'out':  output,
            'zsL':  zsL}

    def loss(self, *args, **kwargs) -> DTNS:
        raise NotImplementedError

    @property
    def enc_width(self) -> int:
        return 7 * self.enc_tns.d_model

# CardNet is used only to train CardEnc
class CardNet(Module):

    def __init__(
            self,
            # CardEnc
            emb_width: int=                 24,             # cards embedding width
            time_drop: float=               0.0,
            feat_drop: float=               0.0,
            in_proj: Optional[int]=         None,
            n_layers: int=                  8,
            dense_mul: int=                 4,              # transformer dense multiplication
            dropout: float=                 0.0,            # transformer dropout
            activation: ACT=                torch.nn.ReLU,
            # EncDRT
            drt_dense_proj: Optional[int]=  None,
            drt_layers: Optional[int]=      2,
            drt_scale: int=                 6,
            drt_dropout: float=             0.0,
            use_huber: bool=                False,           # uses Huber loss for regression
            opt_class=                      torch.optim.Adam,
            opt_betas=                      (0.7,0.7),
            baseLR=                         1e-3,
            warm_up=                        10000,
            ann_base=                       0.999,
            ann_step=                       0.04,
            n_wup_off=                      1,
            gc_factor=                      0.01,
            do_clip=                        False,
    ):

        Module.__init__(self)

        self.card_enc = CardEnc(
            emb_width=  emb_width,
            time_drop=  time_drop,
            feat_drop=  feat_drop,
            in_proj=    in_proj,
            n_layers=   n_layers,
            dense_mul=  dense_mul,
            dropout=    dropout,
            activation= activation)

        self.enc_drt = EncDRT(
            in_width=       2*self.card_enc.enc_width,
            lay_width=      drt_dense_proj,
            n_layers=       drt_layers,
            do_scaled_dns=  True,
            dns_scale=      drt_scale,
            activation=     activation,
            lay_dropout=    drt_dropout,
            initializer=    my_initializer) if drt_layers else None

        # rank classifier
        self.rank = LayDense(
            in_features=    self.card_enc.enc_width,
            out_features=   9,
            activation=     None,
            bias=           False,
            initializer=    my_initializer)

        self.use_huber = use_huber

        # probability of A winning (regression)
        self.wonA_prob = LayDense(
            in_features=    self.card_enc.enc_width,
            out_features=   1,
            activation=     activation,
            bias=           False,
            initializer=    my_initializer)

        # winner classifier (on concatenated representations)
        self.winner = LayDense(
            in_features=    2*self.card_enc.enc_width,
            out_features=   3,
            activation=     None,
            bias=           False,
            initializer=    my_initializer)

    def forward(
            self,
            cards_A: TNS,
            cards_B: TNS) -> DTNS:

        enc_out_A = self.card_enc(cards_A)
        enc_out_B = self.card_enc(cards_B)
        zsL = enc_out_A['zsL'] + enc_out_B['zsL']

        logits_rank_A = self.rank(enc_out_A['out'])
        logits_rank_B = self.rank(enc_out_B['out'])

        reg_won_A= self.wonA_prob(enc_out_A['out'])
        reg_won_A = torch.squeeze(reg_won_A) # reduce last dimension

        conc_out = torch.concat([enc_out_A['out'], enc_out_B['out']], dim=-1)
        if self.enc_drt:
            drt_out = self.enc_drt(conc_out)
            conc_out = drt_out['out']
            zsL += drt_out['zsL']

        logits_winner = self.winner(conc_out)

        return {
            'logits_rank_A':    logits_rank_A,
            'logits_rank_B':    logits_rank_B,
            'reg_won_A':        reg_won_A,
            'logits_winner':    logits_winner,
            'zsL':              zsL}

    def loss(
            self,
            cards_A: TNS,       # seven cards A (ids)
            cards_B: TNS,       # seven cards B (ids)
                # true
            label_won: TNS,     # won label 0-A, 1-B, 2-draw
            label_rank_A: TNS,  # <0;8>
            label_rank_B: TNS,  # <0;8>
            prob_won_A: TNS,    # probability of winning A
    ) -> DTNS:

        out = self(cards_A=cards_A, cards_B=cards_B)

        # where all cards of A are known (there is no 52 in c_idsA)
        where_all_cards_A = torch.max(cards_A, dim=-1)[0]
        where_all_cards_A = torch.where(
            condition=  where_all_cards_A < 52,
            self=       1.0,
            other=      0.0)

        loss_rank_A = torch.nn.functional.cross_entropy(
            input=      out['logits_rank_A'],
            target=     label_rank_A,
            reduction= 'none')
        loss_rank_A = torch.mean(loss_rank_A * where_all_cards_A) # masked where all cards of A are known

        loss_rank_B = torch.nn.functional.cross_entropy(
            input=      out['logits_rank_B'],
            target=     label_rank_B,
            reduction=  'mean') # cards of B are all known always

        loss_rank = loss_rank_A + loss_rank_B

        # rank (B) metrics
        pred_rank = torch.argmax(out['logits_rank_B'], dim=-1)
        correct_pred_rank = torch.eq(pred_rank, label_rank_B).to(torch.float)
        accuracy_rank = torch.mean(correct_pred_rank)

        loss_reg = torch.nn.functional.huber_loss if self.use_huber else torch.nn.functional.mse_loss
        loss_won_A = loss_reg(
            input=      out['reg_won_A'],
            target=     prob_won_A,
            reduction=  'mean') # for any A cards configuration (known/not known)

        # difference in probabilities
        diff_won_prob = torch.abs(prob_won_A - out['reg_won_A'])
        diff_won_prob_mean = torch.mean(diff_won_prob)
        diff_won_prob_max = torch.max(diff_won_prob)

        loss_winner = torch.nn.functional.cross_entropy(
            input=      out['logits_winner'],
            target=     label_won,
            reduction= 'none')
        loss_winner = torch.mean(loss_winner * where_all_cards_A) # masked..

        loss = loss_winner + loss_rank + loss_won_A

        # winner classifier metrics
        pred_winner = torch.argmax(out['logits_winner'], dim=-1)
        correct_pred_winner = torch.eq(pred_winner, label_won)
        correct_pred_winner_where = correct_pred_winner * where_all_cards_A
        accuracy_winner = torch.sum(correct_pred_winner_where) / torch.sum(where_all_cards_A)

        out.update({
            'loss':                 loss,
            'loss_winner':          loss_winner,
            'loss_rank':            loss_rank,
            'loss_won_A':           loss_won_A,
            'accuracy_winner':      accuracy_winner,
            'accuracy_rank':        accuracy_rank,
            'diff_won_prob_mean':   diff_won_prob_mean,
            'diff_won_prob_max':    diff_won_prob_max,})
        return out

# MOTorch for CardNet, overrides save & load for checkpoint of CardEnc only
class CardNet_MOTorch(MOTorch):

    def __init__(
            self,
            cards_emb_width: int,
            module_type=    CardNet,
            name=           None,
            save_topdir=    CN_MODELS_FD,
            read_only=      True,
            **kwargs):

        MOTorch.__init__(
            self,
            module_type=    module_type,
            name=           name or get_cardNet_name(cards_emb_width),
            save_topdir=    save_topdir,
            emb_width=      cards_emb_width,
            read_only=      read_only,
            **kwargs)