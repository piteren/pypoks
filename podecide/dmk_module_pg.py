import torch
from torchness.types import TNS, DTNS
from torchness.motorch import Module
from torchness.base_elements import my_initializer
from torchness.layers import LayDense
from torchness.encoders import EncCNN
from typing import Optional

from envy import N_TABLE_PLAYERS, TBL_MOV, TABLE_CASH_START
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

        # event embeddings # TODO: by now extended * players
        n_events = 1 + N_TABLE_PLAYERS + len(TBL_MOV) * N_TABLE_PLAYERS
        self.event_emb = torch.nn.Parameter(data=torch.empty(size=(n_events,enc_width)))
        my_initializer(self.event_emb)

        # player embeddings (..used only by MRGPL)
        self.player_emb = torch.nn.Parameter(data=torch.empty(size=(N_TABLE_PLAYERS,enc_width)))
        my_initializer(self.player_emb)

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
            cards: TNS,                             # 7 cards ids tensor (int)
            switch: TNS,                            # 1 for cards 0 for event (int)
            event: TNS,                             # event id tensor (int)
            player: TNS,                            # player id, 0 is me (int)
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

        event = event + player * len(TBL_MOV)  # hides player in event

        output = switch * ce + (1-switch) * self.event_emb[event]

        enc_cnn_out = self.enc_cnn(inp=output, history=enc_cnn_state)
        output = enc_cnn_out['out']
        fin_state = enc_cnn_out['state']
        zsL_cnn = enc_cnn_out['zsL']

        logits = self.logits(output)

        return {
            'enc_cnn_out':  enc_cnn_out,
            'logits':       logits,
            'probs':        torch.nn.functional.softmax(input=logits, dim=-1),
            'fin_state':    fin_state,
            'zsL_enc':      zsL_enc,
            'zsL_cnn':      zsL_cnn}

    def loss(
            self,
            cards: TNS,
            switch: TNS,
            event: TNS,
            player: TNS,
            move: TNS,                              # move (action) taken
            reward: TNS,                            # (dreturns)
            enc_cnn_state: Optional[TNS]=   None,
    ) -> DTNS:

        out = self(
            cards=          cards,
            switch=         switch,
            event=          event,
            player=         player,
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
        loss = torch.mean(loss)

        out['loss'] = loss
        out.update(self.min_max_probs(out['probs']))
        return out

    @staticmethod
    def min_max_probs(probs) -> DTNS:
        max_probs = torch.max(probs, dim=-1)[0] # max probs
        min_probs = torch.min(probs, dim=-1)[0] # min probs
        max_probs_mean = torch.mean(max_probs)  # mean of max probs
        min_probs_mean = torch.mean(min_probs)  # mean of min probs
        return {
            'max_probs_mean':   max_probs_mean,
            'min_probs_mean':   min_probs_mean}


class ProCNN_DMK_PG_MRG(ProCNN_DMK_PG):

    def forward(
            self,
            cards: TNS,                             # 7 cards ids tensor (int)
            switch: TNS,                            # 1 for cards 0 for event (int)
            event: TNS,                             # event id tensor (int)
            player: TNS,                            # player id, 0 is me (int)
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

        event = event + player * len(TBL_MOV) # hides player in event

        output = ce + self.event_emb[event]

        enc_cnn_out = self.enc_cnn(inp=output, history=enc_cnn_state)
        output = enc_cnn_out['out']
        fin_state = enc_cnn_out['state']
        zsL_cnn = enc_cnn_out['zsL']

        logits = self.logits(output)

        return {
            'enc_cnn_out':  enc_cnn_out,
            'logits':       logits,
            'probs':        torch.nn.functional.softmax(input=logits, dim=-1),
            'fin_state':    fin_state,
            'zsL_enc':      zsL_enc,
            'zsL_cnn':      zsL_cnn}


class ProCNN_DMK_PG_MRGPL(ProCNN_DMK_PG):

    def forward(
            self,
            cards: TNS,                             # 7 cards ids tensor (int)
            switch: TNS,                            # 1 for cards 0 for event (int)
            event: TNS,                             # event id tensor (int)
            player: TNS,                            # player id, 0 is me (int)
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

        output = ce + self.event_emb[event] + self.player_emb[player]

        enc_cnn_out = self.enc_cnn(inp=output, history=enc_cnn_state)
        output = enc_cnn_out['out']
        fin_state = enc_cnn_out['state']
        zsL_cnn = enc_cnn_out['zsL']

        logits = self.logits(output)

        return {
            'enc_cnn_out':  enc_cnn_out,
            'logits':       logits,
            'probs':        torch.nn.functional.softmax(input=logits, dim=-1),
            'fin_state':    fin_state,
            'zsL_enc':      zsL_enc,
            'zsL_cnn':      zsL_cnn}

# CardNet + EncCNN, Policy Gradient based, evo version
class ProCNN_DMK_PGevo(Module):

    def __init__(
            self,
            train_ce :bool=             True,           # enable training of cards encoder (CardEnc)
            cards_emb_width: int=       12,             # card emb width
            event_emb_width: int=       12,
            float_feat_size: int=       8,
            player_id_emb_width:int=    12,
            player_pos_emb_width: int=  12,
            cnn_width=                  None,           # CNN representation width (number of filters), for None uses CNN input width
            n_lay=                      12,             # number of CNN layers >> makes network deep ( >> context length)
            cnn_ldrt_scale=             0,#3,           # TODO: test it
            activation=                 torch.nn.ReLU,
            opt_class=                  torch.optim.Adam,
            opt_betas=                  (0.7, 0.7),
            baseLR=                     3e-6,
            warm_up=                    100,            # num of steps has to be small (since we do rare updates)
            gc_factor=                  0.05,
            do_clip=                    True,
            device=                     None,
            dtype=                      None,
            logger=                     None):

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
        cn_enc_width = self.card_net.module.card_enc.enc_width

        # event embeddings
        n_events = 1 + len(TBL_MOV)
        self.event_id_emb = torch.nn.Parameter(data=torch.empty(size=(n_events, event_emb_width)))
        my_initializer(self.event_id_emb)

        # player id embeddings
        self.player_id_emb = torch.nn.Parameter(data=torch.empty(size=(N_TABLE_PLAYERS, player_id_emb_width)))
        my_initializer(self.player_id_emb)

        # player pos embeddings
        self.player_pos_emb = torch.nn.Parameter(data=torch.empty(size=(N_TABLE_PLAYERS, player_pos_emb_width)))
        my_initializer(self.player_pos_emb)

        # INFO: stats temporary disabled
       #cnn_in_width =  cn_enc_width + event_emb_width + player_id_emb_width + player_pos_emb_width + 7 + 12 # 7 cash + 12 stats
       #cnn_out_width = cn_enc_width + event_emb_width + player_id_emb_width + player_pos_emb_width + (7 + 12) * float_feat_size
        cnn_in_width =  cn_enc_width + event_emb_width + player_id_emb_width + player_pos_emb_width + 7
        cnn_out_width = cn_enc_width + event_emb_width + player_id_emb_width + player_pos_emb_width + 7        * float_feat_size
        cnn_out_width = cnn_width or cnn_out_width

        self.enc_cnn = EncCNN(
            in_features=    cnn_in_width,
            time_drop=      0.0,
            feat_drop=      0.0,
            shared_lays=    False,
            n_layers=       n_lay,
            n_filters=      cnn_out_width,
            activation=     activation,
            do_ldrt=        bool(cnn_ldrt_scale),
            ldrt_dns_scale= cnn_ldrt_scale,
            initializer=    my_initializer)

        self.logits = LayDense(
            in_features=    cnn_out_width,
            out_features=   len(TBL_MOV),
            activation=     None,
            bias=           False,
            initializer=    my_initializer)

    def forward(
            self,
            cards: TNS,         # cards ids tensor (7 x int)        <- emb
            event_id: TNS,      # event id (int)                    <- emb
            cash: TNS,          # cash values (7 x float) move, 3x player, 3x table
            pl_id: TNS,         # player id, 0 is me (int)          <- emb
            pl_pos: TNS,        # player pos, 0 is SB (int)         <- emb
            pl_stats: TNS,      # player stats (float,..)
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

        feats_to_cat = [
            ce,
            self.event_id_emb[event_id],
            cash / TABLE_CASH_START,
            self.player_id_emb[pl_id],
            self.player_pos_emb[pl_pos],
            # pl_stats, # INFO: temporary disabled
        ]
        inp = torch.cat(feats_to_cat, dim=-1)

        enc_cnn_out = self.enc_cnn(
            inp=        inp,
            history=    enc_cnn_state)
        output = enc_cnn_out['out']
        fin_state = enc_cnn_out['state']
        zsL_cnn = enc_cnn_out['zsL']

        logits = self.logits(output)

        return {
            'enc_cnn_out':  enc_cnn_out,
            'logits':       logits,
            'probs':        torch.nn.functional.softmax(input=logits, dim=-1),
            'fin_state':    fin_state,
            'zsL_enc':      zsL_enc,
            'zsL_cnn':      zsL_cnn}

    def loss(
            self,
            cards: TNS,
            event_id: TNS,
            cash: TNS,
            pl_id: TNS,
            pl_pos: TNS,
            pl_stats: TNS,
            move: TNS,                              # move (action) taken
            reward: TNS,                            # (dreturns)
            enc_cnn_state: Optional[TNS]=   None,
    ) -> DTNS:

        out = self(
            cards=          cards,
            event_id=       event_id,
            cash=           cash,
            pl_id=          pl_id,
            pl_pos=         pl_pos,
            pl_stats=       pl_stats,
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
        loss = torch.mean(loss)

        out['loss'] = loss
        out.update(self.min_max_probs(out['probs']))
        return out

    @staticmethod
    def min_max_probs(probs) -> DTNS:
        max_probs = torch.max(probs, dim=-1)[0] # max probs
        min_probs = torch.min(probs, dim=-1)[0] # min probs
        max_probs_mean = torch.mean(max_probs)  # mean of max probs
        min_probs_mean = torch.mean(min_probs)  # mean of min probs
        return {
            'max_probs_mean':   max_probs_mean,
            'min_probs_mean':   min_probs_mean}