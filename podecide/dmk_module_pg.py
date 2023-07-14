import torch
from torchness.types import TNS, DTNS
from torchness.motorch import Module
from torchness.base_elements import my_initializer
from torchness.layers import LayDense
from torchness.encoders import EncCNN
from typing import Optional

from envy import N_TABLE_PLAYERS, TBL_MOV
from podecide.cardNet.cardNet_module import CardNet_MOTorch


# CardNet + EncCNN, Policy Gradient based
class ProCNN_DMK_PG(Module):

    def __init__(
            self,
            train_ce :bool=         True,           # enable training of cards encoder (CardEnc)
            cards_emb_width: int=   12,             # card emb width >> makes network width (x7)
            width=                  None,           # representation width (number of filters), for None uses cards_encoded_width
            n_lay=                  12,             # number of CNN layers >> makes network deep ( >> context length)
            cnn_ldrt_scale=         0,#3,           # TODO: test it
            activation=             torch.nn.ReLU,
            opt_class=              torch.optim.Adam,
            opt_betas=              (0.7, 0.7),
            baseLR=                 3e-6,
            warm_up=                100,            # num of steps has to be small (since we do rare updates)
            gc_factor=              0.05,
            do_clip=                True,
            device=                 None,
            dtype=                  None,
            logger=                 None):

        Module.__init__(self)

        self.train_ce = train_ce
        self.card_net = CardNet_MOTorch(
            cards_emb_width=    cards_emb_width,
            device=             device,
            dtype=              dtype,
            bypass_data_conv=   True,
            try_load_ckpt=      False,
            read_only=          True,
            logger=             logger)
        enc_width = self.card_net.module.card_enc.enc_width # just alias

        # event embeddings
        n_events = 1 + N_TABLE_PLAYERS + len(TBL_MOV) * (N_TABLE_PLAYERS - 1)
        self.event_emb = torch.nn.Parameter(
            data=   torch.empty(
                size=   (n_events, enc_width)))
        my_initializer(self.event_emb)

        self.enc_cnn = EncCNN(
            in_features=    enc_width,
            time_drop=      0.0,
            feat_drop=      0.0,
            shared_lays=    False,
            n_layers=       n_lay,
            n_filters=      width or enc_width,
            activation=     activation,
            do_ldrt=        bool(cnn_ldrt_scale),
            ldrt_dns_scale= cnn_ldrt_scale,
            initializer=    my_initializer)

        self.logits = LayDense(
            in_features=    width or enc_width,
            out_features=   len(TBL_MOV),
            activation=     None,
            bias=           False,
            initializer=    my_initializer)

    def forward(
            self,
            cards: TNS,                             # 7 cards ids tensor
            event: TNS,                             # event id tensor
            switch: TNS,                            # 1 for cads 0 for event
            enc_cnn_state: Optional[TNS]=   None,   # state tensor
    ) -> DTNS:

        card_enc_module = self.card_net.module.card_enc # just alias
        if self.train_ce:
            card_enc_out = card_enc_module(cards)
        else:
            with torch.no_grad():
                card_enc_out = card_enc_module(cards)
        ce = card_enc_out['out']
        zsL_enc = card_enc_out['zsL']

        switch = switch.to(torch.float32)

        output = switch * ce + (1-switch) * self.event_emb[event]

        enc_cnn_out = self.enc_cnn(inp=output, history=enc_cnn_state)
        output = enc_cnn_out['out']
        fin_state = enc_cnn_out['state']
        zsL_cnn = enc_cnn_out['zsL']

        logits = self.logits(output)

        return {
            'logits':       logits,
            'probs':        torch.nn.functional.softmax(input=logits, dim=-1),
            'fin_state':    fin_state,
            'zsL_enc':      zsL_enc,
            'zsL_cnn':      zsL_cnn}

    def loss(
            self,
            cards: TNS,
            event: TNS,
            switch: TNS,
            move: TNS,                              # move (action) taken
            reward: TNS,                            # (dreturns)
            enc_cnn_state: Optional[TNS]=   None,
    ) -> DTNS:

        out = self(
            cards=          cards,
            event=          event,
            switch=         switch,
            enc_cnn_state=  enc_cnn_state)

        logits = out['logits']

        # INFO: loss for reshaped tensors since torch does not support higher dim here
        orig_shape = list(logits.shape)
        loss = torch.nn.functional.cross_entropy(
            input=      logits.view(-1,orig_shape[-1]),
            target=     move.view(-1),
            reduction=  'none')
        loss = loss.view(orig_shape[:-1])
        loss = loss * reward

        probs = out['probs']
        max_probs = torch.max(probs, dim=-1)[0] # max probs
        min_probs = torch.min(probs, dim=-1)[0] # min probs
        max_probs_mean = torch.mean(max_probs)  # mean of max probs
        min_probs_mean = torch.mean(min_probs)  # mean of min probs

        out.update({
            'loss':             torch.mean(loss),
            'max_probs_mean':   max_probs_mean,
            'min_probs_mean':   min_probs_mean})
        return out