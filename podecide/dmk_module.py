import torch
from torchness.types import TNS, DTNS
from torchness.motorch import Module, MOTorchException
from torchness.base_elements import my_initializer, select_with_indices
from torchness.layers import LayDense
from torchness.encoders import EncCNN
from typing import Optional, Tuple, Dict, List

from podecide.cardNet.cardNet_module import CardNet_MOTorch


class ProCNN_DMK_PG(Module):
    """ Policy Gradient based DMK Module
    CardNet + EncCNN """

    def __init__(
            self,
            table_size: int,                            # number of table players / opponents
            table_moves: List,                          # moves supported by DMK Module
            train_ce :bool=                 False,      # enable training of cards encoder (CardEnc)
            cards_emb_width: int=           12,         # card embedding width
            event_emb_width: int=           12,         # event embedding width
            float_feat_size: int=           8,
            player_id_emb_width:int=        12,
            player_pos_emb_width: int=      12,
            cnn_width=                      None,       # CNN representation width (number of filters), for None uses CNN input width
            n_lay=                          12,         # number of CNN layers >> makes network deep ( >> context length)
            cnn_ldrt_scale=                 0,
            activation=                     torch.nn.ReLU,
            opt_class=                      torch.optim.Adam,
            opt_alpha=                      0.7,
            opt_beta=                       0.7,
            opt_amsgrad=                    False,
            baseLR=                         5e-6,
            warm_up=                        100,        # num of steps has to be small (since updates are rare)
            gc_do_clip=                     True,
            gc_factor=                      0.05,
            reward_norm: bool=              True,       # apply normalization to rewards
            clip_coef: float=               0.2,        # PPO policy clipping coefficient, set here for monitoring PG update
            nam_loss_coef: float=           20.0,       # not allowed moves loss coefficient
            entropy_coef: float=            0.02,       # entropy loss coefficient
            device=                         None,
            dtype=                          None,
            **kwargs):
        """ Notes about hyper-parameters:

        Many training loops has been run, and some values of hpms worked better than other.
        Most loops were run with PG 12 (cards_emb_width).

        - train_ce -> False

        many optimizer configuration were tested, like amsgrad, rmsprop, but finally Adam was performing best with:
        - opt_alpha -> 0.7
        - opt_beta -> 0.7
        - baseLR -> <1e-6;1e-5>
        - gc_do_clip -> True
        - reward_norm -> True """

        Module.__init__(self, **kwargs)

        self.train_ce = train_ce

        card_net_MOTorch = CardNet_MOTorch(
            cards_emb_width=    cards_emb_width,
            device=             device,
            dtype=              dtype,
            bypass_data_conv=   True,
            try_load_ckpt=      False,
            read_only=          True,
            logger=             self.logger)
        self.card_net = card_net_MOTorch.module
        cn_enc_width = self.card_net.card_enc.enc_width

        # event embeddings
        n_events = 1 + len(table_moves) # POS + all MOVes
        self.event_id_emb = torch.nn.Parameter(data=torch.empty(size=(n_events, event_emb_width)))
        my_initializer(self.event_id_emb)

        # player id embeddings
        self.player_id_emb = torch.nn.Parameter(data=torch.empty(size=(table_size, player_id_emb_width)))
        my_initializer(self.player_id_emb)

        # player pos embeddings
        self.player_pos_emb = torch.nn.Parameter(data=torch.empty(size=(table_size, player_pos_emb_width)))
        my_initializer(self.player_pos_emb)

        n_st = 0#len(PLAYER_STATS_USED) # number of stats floats # INFO: since stats are temporary disabled
        n_floats = 1 + 8 + n_st # cn_prob_win + 8 cash + stats
        cnn_in_width =  cn_enc_width + event_emb_width + player_id_emb_width + player_pos_emb_width + n_floats
        cnn_out_width = cn_enc_width + event_emb_width + player_id_emb_width + player_pos_emb_width + n_floats * float_feat_size
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
            out_features=   len(table_moves),
            activation=     None,
            bias=           False,
            initializer=    my_initializer)

        self.opt_class = opt_class
        self.opt_alpha = opt_alpha
        self.opt_beta = opt_beta
        self.opt_amsgrad = opt_amsgrad

        self.reward_norm = reward_norm
        self.clip_coef = clip_coef
        self.nam_loss_coef = nam_loss_coef
        self.entropy_coef = entropy_coef

    def forward(
            self,
            cards: TNS,         # cards ids tensor (7 x int)        <- emb
            event_id: TNS,      # event id (int)                    <- emb
            cash: TNS,          # cash values (8 x float) move, 3x player, 4x table
            pl_id: TNS,         # player id, 0 is me (int)          <- emb
            pl_pos: TNS,        # player pos, 0 is SB (int)         <- emb
            pl_stats: TNS,      # player stats (float,..)
            enc_cnn_state: Optional[TNS]=   None,   # state tensor
    ) -> DTNS:

        if self.train_ce:
            card_enc_out = self.card_net.card_enc(cards)
            won_prob = self.card_net.won_prob(card_enc_out['out'])
        else:
            with torch.no_grad():
                card_enc_out = self.card_net.card_enc(cards)
                won_prob = self.card_net.won_prob(card_enc_out['out'])

        feats = [
            card_enc_out['out'],
            won_prob,
            self.event_id_emb[event_id],
            cash,
            self.player_id_emb[pl_id],
            self.player_pos_emb[pl_pos],
            # pl_stats, # INFO: temporary disabled
        ]
        inp = torch.cat(feats, dim=-1)

        enc_cnn_out = self.enc_cnn(
            inp=        inp,
            history=    enc_cnn_state)
        output = enc_cnn_out['out']
        logits = self.logits(output)

        dist = torch.distributions.Categorical(logits=logits)

        return {
            'enc_cnn_output':   output,
            'logits':           logits,
            'probs':            dist.probs,
            'entropy':          dist.entropy().mean(),
            'fin_state':        enc_cnn_out['state'],
            'zeroes_enc':       card_enc_out['zeroes'],
            'zeroes_cnn':       enc_cnn_out['zeroes']}

    def fwd_logprob(self, move:TNS, **kwargs) -> DTNS:
        """ FWD
        + preparation of logprob (ln(prob) of selected move)
        this method comes from PPO """
        out = self(**kwargs)
        prob_move = select_with_indices(source=out['probs'], indices=move)
        out['logprob'] = torch.log(prob_move)
        return out

    def fwd_logprob_ratio(self, old_logprob:TNS, **kwargs) -> DTNS:
        """ FWD
        + logprob (current)
        + ratio of current logprob vs given old
        + prepares additional metrics
        this method comes from PPO """

        logrpob_out = self.fwd_logprob(**kwargs)
        new_logrpob = logrpob_out.pop('logprob')
        logratio = new_logrpob - old_logprob
        ratio = logratio.exp()

        out = logrpob_out
        out['ratio'] = ratio

        # stats
        with torch.no_grad():
            out.update({
                'approx_kl': ((ratio - 1) - logratio).mean(),
                'clipfracs': ((ratio - 1.0).abs() > self.clip_coef).float().mean()})

        return out

    def get_optimizer_def(self) -> Tuple[type(torch.optim.Optimizer), Dict]:

        if self.opt_class not in [
            torch.optim.Adam,
            torch.optim.RAdam,
            torch.optim.RMSprop,
            torch.optim.SGD,
        ]:
            err = f'{self.__class__.__name__} got not supported optimizer: {self.opt_class.__name__}'
            self.logger.error(err)
            raise MOTorchException(err)

        opt_kwargs = {}

        if self.opt_class == torch.optim.Adam:
            opt_kwargs['betas'] = (self.opt_alpha, self.opt_beta)
            opt_kwargs['amsgrad'] = self.opt_amsgrad
            opt_kwargs['eps'] = 1e-5 # from PPO, original: 1e-8

        if self.opt_class == torch.optim.RAdam:
            opt_kwargs['betas'] = (self.opt_alpha, self.opt_beta)

        if self.opt_class == torch.optim.RMSprop:
            opt_kwargs['alpha'] = self.opt_alpha

        return self.opt_class, opt_kwargs

    @staticmethod
    def loss_nam(logits:TNS, allowed_moves:TNS) -> TNS:
        return torch.sum(torch.softmax(logits, dim=-1) * ~allowed_moves, dim=-1) ** 2

    def loss(
            self,
            cards: TNS,
            event_id: TNS,
            cash: TNS,
            pl_id: TNS,
            pl_pos: TNS,
            pl_stats: TNS,
            move: TNS,           # move (action) taken
            reward: TNS,         # (dreturns)
            allowed_moves: TNS,  # OH tensor
            enc_cnn_state: Optional[TNS]=   None,
            old_logprob: Optional[TNS]=     None, # not used by PG, added for compatibility with PPO
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

        reward_norm = self.norm(reward)
        deciding_state = torch.sum(allowed_moves, dim=-1) > 0  # bool tensor, True where state is deciding one (OD in MSOD)

        # INFO: loss for reshaped tensors since torch does not support higher dim here
        orig_shape = list(logits.shape)
        loss_actor = torch.nn.functional.cross_entropy(
            input=      logits.view(-1, orig_shape[-1]),
            target=     move.view(-1),
            reduction=  'none')
        loss_actor = loss_actor.view(orig_shape[:-1])
        loss_actor = loss_actor * (reward_norm if self.reward_norm else reward)

        # multiply by deciding_state for zero loss in non-deciding states,
        # non-deciding states should have reward == 0, BUT after reward normalization
        # reward_norm (-> reward_selected) may be != 0 for those
        loss_actor *= deciding_state
        loss_actor = torch.mean(loss_actor)

        entropy = out['entropy']
        loss_entropy_factor = self.entropy_coef * entropy

        loss_nam = self.loss_nam(
            logits=         logits,
            allowed_moves=  allowed_moves)
        loss_nam *= deciding_state
        loss_nam = torch.mean(loss_nam)
        loss_nam_factor = self.nam_loss_coef * loss_nam

        loss = loss_actor - loss_entropy_factor + loss_nam_factor

        out.update({
            'reward':       reward,
            'reward_norm':  reward_norm,
            'loss':         loss,
            'loss_actor':   loss_actor,
            'loss_nam':     loss_nam})
        return out

    @staticmethod
    def norm(tns:TNS) -> TNS:
        """ normalizes tensor """
        return (tns - tns.mean()) / (tns.std() + 1e-8)


class ProCNN_DMK_A2C(ProCNN_DMK_PG):
    """ Actor + Critic based (in one tower) DMK Module """

    def __init__(self, loss_critic_factor:float=0.5, **kwargs):
        ProCNN_DMK_PG.__init__(self, **kwargs)
        self.loss_critic_factor = loss_critic_factor

        self.value = LayDense(
            in_features=    self.enc_cnn.n_filters,
            out_features=   1,
            activation=     None,
            bias=           False,
            initializer=    my_initializer)

    def forward(self, **kwargs) -> DTNS:

        s_out = super().forward(**kwargs)

        enc_cnn_output = s_out['enc_cnn_output']

        value = self.value(enc_cnn_output) # baseline architecture, where value and policy share same NN
        value = torch.reshape(value, (value.shape[:-1]))  # remove last dim

        s_out['value'] = value
        return s_out

    def loss_critic(self, value:TNS, dreturn:TNS) -> TNS:
        loss_critic = torch.nn.functional.huber_loss(input=value, target=dreturn)
        return loss_critic * self.loss_critic_factor

    def loss(
            self,
            move: TNS,
            reward: TNS,
            allowed_moves: TNS,  # OH tensor
            old_logprob: Optional[TNS]= None,  # not used by A2C (similar to PG), added for compatibility with PPO
            **kwargs
    ) -> DTNS:
        """ A2C loss overrides PG loss:
        - replaces reward (dreturn) with advantage, reward_norm replaced by advantage_norm
        - adds Critic loss """

        out = self(**kwargs)

        value = out['value']
        logits = out['logits']

        deciding_state = torch.sum(allowed_moves, dim=-1) > 0  # bool tensor, True where state is deciding one (OD in MSOD)

        advantage = reward - value
        advantage_nograd = advantage.detach()  # to prevent flow of Actor loss gradients to Critic network
        advantage_norm = self.norm(advantage_nograd)

        # INFO: loss for reshaped tensors since torch does not support higher dim here
        orig_shape = list(logits.shape)
        loss_actor = torch.nn.functional.cross_entropy(
            input=      logits.view(-1, orig_shape[-1]),
            target=     move.view(-1),
            reduction=  'none')
        loss_actor = loss_actor.view(orig_shape[:-1])
        loss_actor = loss_actor * (advantage_nograd if self.reward_norm else advantage_nograd)

        # multiply by deciding_state for zero loss in non-deciding states,
        # non-deciding states should have reward == 0, BUT after reward normalization
        # reward_norm (-> reward_selected) may be != 0 for those
        loss_actor *= deciding_state
        loss_actor = torch.mean(loss_actor)

        loss_critic = self.loss_critic(
            value=      value,
            dreturn=    reward)

        entropy = out['entropy']
        loss_entropy_factor = self.entropy_coef * entropy

        loss_nam = self.loss_nam(
            logits=         logits,
            allowed_moves=  allowed_moves)
        loss_nam *= deciding_state
        loss_nam = torch.mean(loss_nam)
        loss_nam_factor = self.nam_loss_coef * loss_nam

        loss = loss_actor + loss_critic - loss_entropy_factor + loss_nam_factor

        out.update({
            'reward':           reward,
            'advantage':        advantage_nograd,
            'advantage_norm':   advantage_norm,
            'loss':             loss,
            'loss_actor':       loss_actor,
            'loss_critic':      loss_critic,
            'loss_nam':         loss_nam})
        return out


class ProCNN_DMK_PPO(ProCNN_DMK_PG):
    """ PPO based DMK Module """

    def __init__(
            self,
            gc_do_clip=         True,
            gc_factor=          0.01,
            gc_max_clip=        0.5,
            gc_max_upd=         1.1,
            minibatch_num: int= 5,
            n_epochs_ppo: int=  1,
            **kwargs
    ):
        ProCNN_DMK_PG.__init__(self, **kwargs)

    def loss_actor(self, advantage:TNS, ratio:TNS) -> TNS:
        """ actor (policy) loss, clipped """
        pg_loss1 = -advantage * ratio
        pg_loss2 = -advantage * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
        return torch.max(pg_loss1, pg_loss2)

    def loss(
            self,
            reward: TNS,
            allowed_moves: TNS,
            **kwargs,
    ) -> DTNS:
        """ PPO loss
        modified: no critic & advantages """

        old_logprob = kwargs.pop('old_logprob')
        ratio_out = self.fwd_logprob_ratio(old_logprob=old_logprob, **kwargs)

        reward_norm = self.norm(reward)
        deciding_state = torch.sum(allowed_moves, dim=-1) > 0

        loss_actor = self.loss_actor(
            advantage=  reward_norm if self.reward_norm else reward,
            ratio=      ratio_out['ratio'])
        loss_actor *= deciding_state
        loss_actor = torch.mean(loss_actor)

        entropy = ratio_out['entropy']
        loss_entropy_factor = self.entropy_coef * entropy

        loss_nam = self.loss_nam(
            logits=         ratio_out['logits'],
            allowed_moves=  allowed_moves)
        loss_nam *= deciding_state
        loss_nam = torch.mean(loss_nam)
        loss_nam_factor = self.nam_loss_coef * loss_nam

        loss = loss_actor - loss_entropy_factor + loss_nam_factor

        out = ratio_out
        out.update({
            'reward':       reward,
            'reward_norm':  reward_norm,
            'entropy':      entropy,
            'loss':         loss,
            'loss_actor':   loss_actor,
            'loss_nam':     loss_nam})
        return out