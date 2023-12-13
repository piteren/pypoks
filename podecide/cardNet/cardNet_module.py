import torch
from torchness.types import TNS, DTNS, ACT
from torchness.base_elements import my_initializer
from torchness.layers import TF_Dropout, LayDense
from torchness.encoders import EncTNS, EncDRT
from torchness.motorch import Module, MOTorch
from typing import Optional, Tuple, Dict

from envy import CN_MODELS_FD, get_cardNet_name, PyPoksException


# Cards Encoder, TNS based
class CardEnc(Module):

    def __init__(
            self,
            cards_emb_width: int=   12,             # cards embedding width, makes encoder width x7
            time_drop: float=       0.0,
            feat_drop: float=       0.0,
            in_proj: Optional[int]= None,
            n_layers: int=          8,
            dense_mul: int=         4,              # transformer dense multiplication
            dropout: float=         0.0,            # transformer dropout
            activation: ACT=        torch.nn.ReLU,
            **kwargs):

        Module.__init__(self, **kwargs)

        self.cards_emb_width = cards_emb_width

        self.cards_emb = torch.nn.Parameter(torch.empty(size=(53,self.cards_emb_width))) # 52 + one card for 'no_card'
        my_initializer(self.cards_emb)

        self.mycards_emb = torch.nn.Parameter(torch.empty(size=(2,self.cards_emb_width)))# my cards
        my_initializer(self.mycards_emb)

        self.tf_drop = TF_Dropout(
            time_drop=  time_drop,
            feat_drop=  feat_drop)

        self.in_proj = LayDense(
            in_features=    self.cards_emb_width,
            out_features=   in_proj,
            activation=     None,
            bias=           False,
            initializer=    my_initializer) if in_proj else None

        self.enc_tns = EncTNS(
            num_layers=     n_layers,
            num_layers_TAT= 0,
            d_model=        self.cards_emb_width if not in_proj else in_proj,
            nhead=          1,
            dns_scale=      dense_mul,
            dropout=        dropout,
            activation=     activation)

        self.logger.info(f'*** CardEnc *** initialized, cards_emb_width:{self.cards_emb_width}, enc_width:{self.enc_width}')

    def forward(
            self,
            cards: TNS # seven cards (ids)
    ) -> DTNS:

        self.logger.debug(f'CardEnc forward called with cards: {cards} {cards.shape}')
        my_cards_indexes = [0,0,1,1,1,1,1] # adds info which cards are mine (EncTNS does not recognize order)

        input = self.cards_emb[cards] + self.mycards_emb[my_cards_indexes]
        self.logger.debug(f'input shape (7 cards embedded): {input.shape}')
        input = self.tf_drop(input)

        if self.in_proj: input = self.in_proj(input)

        enc_out = self.enc_tns(input)
        output = enc_out['out']

        self.logger.debug(f'output shape (TNS encoder): {output.shape}')
        output = output.view(list(output.shape)[:-2] + [-1]) # flatten last two dim
        self.logger.debug(f'output shape (flattened): {output.shape}')

        return {'out':output, 'zeroes':enc_out['zeroes']}

    def loss(self, *args, **kwargs) -> DTNS:
        raise NotImplementedError

    @property
    def enc_width(self) -> int:
        return 7 * self.enc_tns.d_model


class CardNet(Module):
    """ CardNet is used to train CardEnc """

    def __init__(
            self,
            # CardEnc
            cards_emb_width: int=           12,             # cards embedding width
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
            use_huber: bool=                False,           # use Huber loss for regression
            opt_class=                      torch.optim.Adam,
            opt_alpha=                      0.7,
            opt_beta=                       0.5,
            baseLR=                         1e-3,
            warm_up=                        10000,
            n_wup_off=                      5,
            ann_base=                       0.999,
            ann_step=                       0.04, # for training longer than 200K batches 0.02 gives better result
            gc_do_clip=                     False,
            gc_start_val=                   0.1,
            gc_factor=                      0.01,
            gc_max_clip=                    1.0,
            gc_max_upd=                     1.1,
            loss_winner_coef=               0.3,
            loss_rank_coef=                 0.2,
            **kwargs):

        Module.__init__(self, **kwargs)

        self.card_enc = CardEnc(
            cards_emb_width=    cards_emb_width,
            time_drop=          time_drop,
            feat_drop=          feat_drop,
            in_proj=            in_proj,
            n_layers=           n_layers,
            dense_mul=          dense_mul,
            dropout=            dropout,
            activation=         activation,
            logger=             self.logger)

        self.enc_drt = EncDRT(
            in_width=       2*self.card_enc.enc_width,
            lay_width=      drt_dense_proj,
            n_layers=       drt_layers,
            do_scaled_dns=  True,
            dns_scale=      drt_scale,
            activation=     activation,
            lay_dropout=    drt_dropout,
            initializer=    my_initializer) if drt_layers else None

        # probability of winning (regression)
        self.won_prob = LayDense(
            in_features=    self.card_enc.enc_width,
            out_features=   1,
            activation=     None,
            bias=           False,
            initializer=    my_initializer)

        # rank classifier
        self.rank = LayDense(
            in_features=    self.card_enc.enc_width,
            out_features=   9,
            activation=     None,
            bias=           False,
            initializer=    my_initializer)

        self.use_huber = use_huber

        # winner classifier (on concatenated representations)
        self.winner = LayDense(
            in_features=    2*self.card_enc.enc_width,
            out_features=   3,
            activation=     None,
            bias=           False,
            initializer=    my_initializer)

        self.opt_class = opt_class
        self.opt_alpha = opt_alpha
        self.opt_beta = opt_beta
        self.loss_winner_coef = loss_winner_coef
        self.loss_rank_coef = loss_rank_coef

    def forward(
            self,
            cards_A: TNS,
            cards_B: TNS) -> DTNS:

        enc_out_A = self.card_enc(cards_A)
        enc_out_B = self.card_enc(cards_B)
        zsL = [enc_out_A['zeroes'], enc_out_B['zeroes']]

        logits_rank_A = self.rank(enc_out_A['out'])
        logits_rank_B = self.rank(enc_out_B['out'])

        reg_won_A = self.won_prob(enc_out_A['out'])

        conc_out = torch.concat([enc_out_A['out'], enc_out_B['out']], dim=-1)
        if self.enc_drt:
            drt_out = self.enc_drt(conc_out)
            conc_out = drt_out['out']
            zsL.append(drt_out['zeroes'])

        logits_winner = self.winner(conc_out)

        return {
            'logits_rank_A':    logits_rank_A,
            'logits_rank_B':    logits_rank_B,
            'reg_won_A':        reg_won_A,
            'logits_winner':    logits_winner,
            'zeroes':           torch.cat(zsL)}

    def get_optimizer_def(self) -> Tuple[type(torch.optim.Optimizer), Dict]:
        return self.opt_class, {'betas': (self.opt_alpha, self.opt_beta)}

    def loss(
            self,
            cards_A: TNS,       # seven cards A
            cards_B: TNS,       # seven cards B
            label_won: TNS,     # won label 0-A, 1-B, 2-draw
            label_rank_A: TNS,  # <0;8>
            label_rank_B: TNS,  # <0;8>
            prob_won_A: TNS,    # probability of winning A
    ) -> DTNS:

        out = self(cards_A=cards_A, cards_B=cards_B)

        # where all cards of A are known (there is no 52 in 7)
        cards_A_max_of7 = torch.max(cards_A, dim=-1)[0]
        where_all_cards_A = cards_A_max_of7 < 52

        loss_rank_A = torch.nn.functional.cross_entropy(
            input=      out['logits_rank_A'],
            target=     label_rank_A,
            reduction= 'none')
        loss_rank_A = (loss_rank_A * where_all_cards_A).mean() # masked where all cards of A are known

        loss_rank_B = torch.nn.functional.cross_entropy(
            input=      out['logits_rank_B'],
            target=     label_rank_B,
            reduction=  'mean') # cards of B are all known always

        loss_rank = loss_rank_A + loss_rank_B

        # rank (B) metrics
        pred_rank = torch.argmax(out['logits_rank_B'], dim=-1)
        correct_pred_rank = torch.eq(pred_rank, label_rank_B).to(torch.float)
        accuracy_rank = torch.mean(correct_pred_rank)

        # loss of estimating probability of winning for any A cards configuration (known/not known)
        loss_reg = torch.nn.functional.huber_loss if self.use_huber else torch.nn.functional.mse_loss
        reg_won_A_reduced = torch.squeeze(out['reg_won_A'], dim=-1)
        loss_won_A = loss_reg(
            input=      reg_won_A_reduced,
            target=     prob_won_A,
            reduction=  'mean')

        # difference in probabilities
        diff_won_prob = torch.abs(prob_won_A - reg_won_A_reduced)
        diff_won_prob_mean = torch.mean(diff_won_prob)
        diff_won_prob_max = torch.max(diff_won_prob)

        loss_winner = torch.nn.functional.cross_entropy(
            input=      out['logits_winner'],
            target=     label_won,
            reduction= 'none')
        loss_winner = (loss_winner * where_all_cards_A).mean() # masked..

        loss = self.loss_winner_coef*loss_winner + self.loss_rank_coef*loss_rank + loss_won_A

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
            'diff_won_prob':        diff_won_prob,
            'diff_won_prob_mean':   diff_won_prob_mean,
            'diff_won_prob_max':    diff_won_prob_max})
        return out

# MOTorch for CardNet
class CardNet_MOTorch(MOTorch):

    SAVE_TOPDIR = CN_MODELS_FD

    def __init__(
            self,
            module_type=                    CardNet,
            name: Optional[str]=            None,
            cards_emb_width: Optional[int]= None,
            read_only=                      True,
            **kwargs):

        if name is None and cards_emb_width is None:
            raise PyPoksException('name or cards_emb_width mus be given')

        if not name:
            name = get_cardNet_name(cards_emb_width)

        MOTorch.__init__(
            self,
            module_type=        module_type,
            name=               name,
            cards_emb_width=    cards_emb_width,
            read_only=          read_only,
            **kwargs)