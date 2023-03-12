import torch
from torchness.types import TNS, DTNS
from torchness.motorch import Module, MOTorch
from torchness.base_elements import my_initializer
from torchness.layers import LayDense
from torchness.encoders import EncCNN, EncDRT
from typing import Optional

from pypoks_envy import N_TABLE_PLAYERS, TBL_MOV, DMK_MODELS_FD
from podecide.cardNet.cardNet_module import CardNet_MOTorch


# CNN + RES + CE/move Module, added DRT after CNN
class ProCNN_DMK(Module):

    def __init__(
            self,
            train_ce :bool=         True,           # enable training of cards encoder (CardEnc)
            cards_emb_width: int=   12,             # card emb width >> makes network width (x7)
            width=                  None,           # representation width (number of filters), for None uses cards_encoded_width
            n_lay=                  12,             # number of CNN layers >> makes network deep ( >> context length)
            cnn_ldrt_scale=         0,#3,           # TODO: test it
            n_lay_drt=              0,#2,           # number of drt (after cnn) layers TODO: test it
            enc_drt_scale=          6,              # TODO: test it
            activation=             torch.nn.ReLU,
            opt_class=              torch.optim.Adam,
            opt_betas=              (0.7, 0.7),
            baseLR=                 3e-6,           # TODO: check it - has renamed
            warm_up=                100,            # num of steps has to be small (since we do rare updates)
            gc_factor=              0.05,
            do_clip=                True,
            empower_exp=            1,              # empower exploration loss
            limit_cert=             None,#0.3,
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

        # TODO: maybe remove: in_proj + ln_in since both are implemented in EncCNN
        # projection without activation and bias
        self.in_proj = LayDense(
            in_features=    enc_width,
            out_features=   width,
            activation=     None,
            bias=           False,
            initializer=    my_initializer) if width and width != enc_width else None

        new_width = width if self.in_proj is not None else enc_width

        # layer_norm TODO: may be removed since enc_CNN has LN @input
        self.ln_in = torch.nn.LayerNorm(normalized_shape=new_width)

        self.enc_cnn = EncCNN(
            in_features=    new_width,
            time_drop=      0.0,
            feat_drop=      0.0,
            shared_lays=    False,
            n_layers=       n_lay,
            n_filters=      None,
            activation=     activation,
            do_ldrt=        bool(cnn_ldrt_scale),
            ldrt_dns_scale= cnn_ldrt_scale,
            initializer=    my_initializer)

        self.enc_drt = EncDRT(
            in_width=       new_width,
            n_layers=       n_lay_drt,
            do_scaled_dns=  True,
            dns_scale=      enc_drt_scale,
            activation=     activation,
            initializer=    my_initializer) if n_lay_drt else None

        self.logits = LayDense(
            in_features=    new_width,
            out_features=   len(TBL_MOV),
            activation=     None,
            bias=           False,
            initializer=    my_initializer)

    def forward(
            self,
            cards: TNS,     # 7 cards ids tensor
            event: TNS,     # event id tensor
            switch: TNS,    # 1 for cads 0 for event
            enc_cnn_state: Optional[TNS]=   None,
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

        if self.in_proj:
            output = self.in_proj(output)

        output = self.ln_in(output)

        enc_cnn_out = self.enc_cnn(inp=output, history=enc_cnn_state)
        output = enc_cnn_out['out']
        fin_state = enc_cnn_out['state']
        zsL_cnn = enc_cnn_out['zsL']

        zsL_drt = []
        if self.enc_drt:
            enc_drt_out = self.enc_drt(output)
            output = enc_drt_out['out']
            zsL_drt = enc_drt_out['zsL']

        logits = self.logits(output)

        return {
            'logits':       logits,
            'probs':        torch.nn.functional.softmax(input=logits, dim=-1),
            'fin_state':    fin_state,
            'zsL_enc':      zsL_enc,
            'zsL_cnn':      zsL_cnn,
            'zsL_drt':      zsL_drt}

    def loss(
            self,
            cards: TNS,
            event: TNS,
            switch: TNS,
            move: TNS,
            reward: TNS,
            enc_cnn_state: Optional[TNS]=   None,
    ) -> DTNS:

        out = self(
            cards=          cards,
            event=          event,
            switch=         switch,
            enc_cnn_state=  enc_cnn_state)
        logits = out['logits']
        probs = out['probs']

        max_probs = torch.max(probs, dim=-1)[0] # max probs
        min_probs = torch.min(probs, dim=-1)[0] # min probs
        max_probs_mean = torch.mean(max_probs)  # mean of max probs
        min_probs_mean = torch.mean(min_probs)  # mean of min probs

        orig_shape = list(logits.shape)
        # INFO: loss for reshaped tensors since torch does not support higher dim here
        loss = torch.nn.functional.cross_entropy(
            input=      logits.view(-1,orig_shape[-1]),
            target=     move.view(-1),
            reduction=  'none')
        loss = loss.view(orig_shape[:-1])
        loss = loss * reward

        # TODO: consider implementing, ..BUT never was used till now
        """
        # empower exploration loss
        # INFO: experimental
        if empower_exp > 1:
            argmax = tf.cast(tf.argmax(probs, axis=-1), dtype=tf.int32)
            loss = tf.where( # empower loss for wrong prediction
                condition=  tf.math.equal(argmax, move_PH),
                x=          loss,
                y=          empower_exp*loss)

        # limit certainty
        # INFO: experimental
        if limit_cert:
            max_probs_mm = tf.where( # reduce max_probs to 0 where loss<0
                condition=  tf.math.less(loss, 0),
                x=          tf.zeros_like(max_probs),
                y=          max_probs)
            loss = tf.where( # reduce loss to 0 where max_probs_mm above limit
                condition=  tf.greater(max_probs_mm, limit_cert),
                x=          tf.zeros_like(loss),
                y=          loss)
            loss = tf.reduce_mean(loss)
        """

        loss = torch.mean(loss)

        out.update({
            'loss':             loss,
            'probs':            probs,
            'max_probs_mean':   max_probs_mean,
            'min_probs_mean':   min_probs_mean})
        return out


# adds possibility to load cardNet ckpt while init
class DMK_MOTorch(MOTorch):

    def __init__(
            self,
            module_type=                ProCNN_DMK,
            save_topdir=                DMK_MODELS_FD,
            load_cardnet_pretrained=    False,
            **kwargs):

        # INFO: load_cardnet_pretrained will not be saved with POINT of MOTorch, but it is intended
        MOTorch.__init__(
            self,
            module_type=    module_type,
            save_topdir=    save_topdir,
            **kwargs)

        if load_cardnet_pretrained:
            self.load_cardnet_pretrained()

    def load_cardnet_pretrained(self):
        self.module.card_net.load_ckpt()
        self._log.info(f'{self.name} loaded card_net pretrained checkpoint')