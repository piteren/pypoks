import torch
from torchness.types import TNS, DTNS
from torchness.base_elements import my_initializer
from torchness.layers import LayDense
from typing import Optional

from podecide.dmk_module_pg import ProCNN_DMK_PG


# CNN + RES + CE/move Module, added DRT after CNN, Policy Gradient based
class ProCNN_DMK_A2C(ProCNN_DMK_PG):

    def __init__(self, width=None, **kwargs):

        ProCNN_DMK_PG.__init__(self, width=width, **kwargs)

        self.value = LayDense(
            in_features=    width or self.card_net.module.card_enc.enc_width,
            out_features=   1,
            activation=     None,
            bias=           False,
            initializer=    my_initializer)

    def forward(self, **kwargs) -> DTNS:

        s_out = super().forward(**kwargs)

        output = s_out['enc_cnn_out']['out']

        value = self.value(output) # baseline architecture, where value comes from common A+C tower
        value = torch.reshape(value, (value.shape[:-1]))  # remove last dim

        s_out['value'] = value
        return s_out

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
        value = out['value']

        advantage = reward - value
        advantage_nograd = advantage.detach()  # to prevent flow of Actor loss gradients to Critic network

        # INFO: loss for reshaped tensors since torch does not support higher dim here
        orig_shape = list(logits.shape)
        loss_actor = torch.nn.functional.cross_entropy(
            input=      logits.view(-1,orig_shape[-1]),
            target=     move.view(-1),
            reduction=  'none')
        loss_actor = loss_actor.view(orig_shape[:-1])
        loss_actor = loss_actor * advantage_nograd
        loss_actor = torch.mean(loss_actor)

        loss_critic = torch.nn.functional.huber_loss(value, advantage, reduction='none')
        loss_critic = torch.mean(loss_critic)

        out.update({
            'loss':             loss_actor + loss_critic,
            'loss_actor':       loss_actor,
            'loss_critic':      loss_critic})
        out.update(self.min_max_probs(out['probs']))
        return out