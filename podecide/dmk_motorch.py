import os
import random
import shutil
import torch
from torchness.types import TNS, DTNS
from torchness.motorch import MOTorch, Module
from torchness.comoneural.zeroes_processor import ZeroesProcessor
from typing import Dict, Union, Optional, List, Iterable

from envy import CN_MODELS_FD, DMK_MODELS_FD, get_cardNet_name, PyPoksException
from podecide.dmk_module import ProCNN_DMK_PG, ProCNN_DMK_A2C, ProCNN_DMK_PPO
from podecide.game_state import GameState



class DMK_MOTorch(MOTorch):

    SAVE_TOPDIR = DMK_MODELS_FD

    def __init__(
            self,
            module_type: Optional[type(Module)]=        None,
            name: Optional[str]=                        None,
            name_timestamp=                             False,
            player_ids: Optional[Iterable[str]]=        ('pl0',),
            save_topdir: Optional[str]=                 None,
            save_fn_pfx: Optional[str]=                 None,
            load_cardnet_pretrained: Union[bool, str]=  'auto',  # for 'auto' loads if was not saved before
            **kwargs):
        """ INFO: load_cardnet_pretrained will not be saved with POINT of DMK_MOTorch
        it is on purpose to not load (use default 'auto') CN for trained DMK
        loading CN pretrained is used in general for fresh DMK (or GX without ckpt) """

        name = self._get_name(
            module_type=    module_type,
            name=           name,
            name_timestamp= name_timestamp)

        saved_already = self.is_saved(
            name=           name,
            save_topdir=    save_topdir,
            save_fn_pfx=    save_fn_pfx)

        MOTorch.__init__(
            self,
            module_type=    module_type,
            name=           name,
            name_timestamp= False,
            save_topdir=    save_topdir,
            save_fn_pfx=    save_fn_pfx,
            **kwargs)

        self._player_ids = player_ids

        if (load_cardnet_pretrained == 'auto' and not saved_already) or load_cardnet_pretrained is True:
            self._log.info(f'{self.name} going to load pretrained CN checkpoint (DMK saved_already:{saved_already}, load_cardnet_pretrained:{load_cardnet_pretrained})..')
            self.load_cardnet_pretrained()
        else:
            self._log.info(f'{self.name} has not loaded pretrained CN checkpoint (DMK saved_already:{saved_already}, load_cardnet_pretrained:{load_cardnet_pretrained})')

        self._zero_state = self.convert(self.module.enc_cnn.get_zero_history())
        self._last_fwd_state: Dict[str,TNS] = {pa: self._zero_state for pa in self._player_ids}  # state after last fwd
        self._last_upd_state: Dict[str,TNS] = {pa: self._zero_state for pa in self._player_ids}  # state after last upd

        self._ze_pro_enc = ZeroesProcessor(
            intervals=      (5,20),
            tag_pfx=        'nane_enc',
            tbwr=           self._TBwr) if self._TBwr else None
        self._ze_pro_cnn = ZeroesProcessor(
            intervals=      (5,20),
            tag_pfx=        'nane_cnn',
            tbwr=           self._TBwr) if self._TBwr else None

    def load_cardnet_pretrained(self):
        """ loads CN checkpoint from pretrained """

        cn_model_name = get_cardNet_name(self.cards_emb_width)
        ckpt_path = self._get_ckpt_path(model_name=cn_model_name, save_topdir=CN_MODELS_FD)

        self._log.info(f'> trying to load {cn_model_name} (CN) pretrained checkpoint from {ckpt_path}')

        try:
            save_obj = torch.load(f=ckpt_path, map_location=self.device)
            self.module.card_net.load_state_dict(save_obj.pop('model_state_dict'))
            self._log.info(f'> {cn_model_name} (CN) checkpoint loaded from {ckpt_path}')
        except Exception as e:
            self._log.info(f'> {cn_model_name} (CN) checkpoint NOT loaded because of exception: {e}')

    def reset_fwd_state(self):
        """ resets agent FWD state,
        designed for inference mode <- single player game """
        if len(self._last_fwd_state) > 1:
            raise PyPoksException('reset_fwd_state is valid only for one player')
        pid = list(self._last_fwd_state.keys())[0]
        self._last_fwd_state[pid] = self._zero_state

    def build_batch(
            self,
            player_ids: List[str],                  # list of player_id
            game_statesL: List[List[GameState]],    # list of GameState lists (for each player list of his GameStates)
            for_training=   True,
    ) -> DTNS:
        raise NotImplementedError

    def run_policy(self, player_ids:List[str], batch:DTNS):
        """ runs policy and returns numpy array with probs """
        out = self(bypass_data_conv=True, **batch)

        # save FWD states
        for pid, state in zip(player_ids, out['fin_state']):
            self._last_fwd_state[pid] = state

        return out['probs'].cpu().detach().numpy()

    def update_policy(self, player_ids:List[str], batch:DTNS) -> None:
        """ backward + save states (baseline) """
        out = self.backward(bypass_data_conv=True, **batch)

        # save UPD states
        for pid, state in zip(player_ids, out['fin_state']):
            self._last_upd_state[pid] = state

    # overriden here to allow load_cardnet_pretrained when not do_gx_ckpt
    @classmethod
    def gx_saved(
            cls,
            name_parentA: str,
            name_parentB: Optional[str],
            name_child: str,
            save_topdir_parentA: Optional[str]= None,
            save_topdir_parentB: Optional[str]= None,
            save_topdir_child: Optional[str]=   None,
            save_fn_pfx: Optional[str]=         None,
            device=                             None,
            do_gx_ckpt=                         True,
            ratio: float=                       0.5,
            noise: float=                       0.03,
            logger=                             None,
            loglevel=                           30,
    ) -> None:
        """ performs GX on saved MOTorch """

        if not save_topdir_parentA: save_topdir_parentA = cls.SAVE_TOPDIR
        if not save_fn_pfx: save_fn_pfx = cls.SAVE_FN_PFX

        cls.gx_saved_point(
            name_parentA=           name_parentA,
            name_parentB=           name_parentB,
            name_child=             name_child,
            save_topdir_parentA=    save_topdir_parentA,
            save_topdir_parentB=    save_topdir_parentB,
            save_topdir_child=      save_topdir_child,
            save_fn_pfx=            save_fn_pfx,
            logger=                 logger,
            loglevel=               loglevel)

        if do_gx_ckpt:
            cls.gx_ckpt(
                nameA=              name_parentA,
                nameB=              name_parentB or name_parentA,
                name_child=         name_child,
                save_topdirA=       save_topdir_parentA,
                save_topdirB=       save_topdir_parentB,
                save_topdir_child=  save_topdir_child,
                ratio=              ratio,
                noise=              noise)
        # build and save to have checkpoint saved
        else:
            child = cls(
                name=                       name_child,
                save_topdir=                save_topdir_child or save_topdir_parentA,
                save_fn_pfx=                save_fn_pfx,
                load_cardnet_pretrained=    not do_gx_ckpt, # only this line has changed vs super()
                device=                     device,
                logger=                     logger,
                loglevel=                   loglevel)
            child.save()

    @classmethod
    def save_checkpoint_backup(cls, model_name:str, save_topdir:str):
        """ saves checkpoint backup """
        ckpt_path = cls._get_ckpt_path(model_name=model_name, save_topdir=save_topdir)
        ckpt_path_backup = f'{ckpt_path}.backup'
        if not os.path.isfile(ckpt_path):
            msg = 'cannot save backup, checkpoint does not exist!'
            raise PyPoksException(msg)
        shutil.copyfile(src=ckpt_path, dst=ckpt_path_backup)

    @classmethod
    def restore_checkpoint_backup(cls, model_name:str, save_topdir:str):
        """ restores backup checkpoint """
        ckpt_path = cls._get_ckpt_path(model_name=model_name, save_topdir=save_topdir)
        ckpt_path_backup = f'{ckpt_path}.backup'
        if not os.path.isfile(ckpt_path_backup):
            msg = 'cannot restore backup, backup checkpoint does not exist!'
            raise PyPoksException(msg)
        shutil.copyfile(src=ckpt_path_backup, dst=ckpt_path)


class DMK_MOTorch_PG(DMK_MOTorch):

    def __init__(self, module_type=ProCNN_DMK_PG, **kwargs):
        DMK_MOTorch.__init__(self, module_type=module_type, **kwargs)

    def fwd_logprob(
            self,
            *args,
            move: TNS,                  # move (action) taken
            set_training: bool= None,
            no_grad: bool=      True,
            **kwargs
    ) -> DTNS:
        """ FWD + logprob <- check module fwd_logprob()
        defaults of this method (set_training, no_grad) set it to be used in inference mode """

        if set_training is not None:
            self.train(set_training)

        if no_grad:
            with torch.no_grad():
                out = self.module.fwd_logprob(*args, move=move, **kwargs)
        else:
            out = self.module.fwd_logprob(*args, move=move, **kwargs)

        # eventually roll back to default
        if set_training:
            self.train(False)

        return out

    def fwd_logprob_ratio(
            self,
            *args,
            move: TNS,                  # move (action) taken
            old_logprob: TNS,
            set_training: bool= None,
            no_grad: bool=      True,
            **kwargs
    ) -> DTNS:
        """ FWD + logprob + ratio <- check module fwd_logprob_ratio()
        defaults of this method (set_training, no_grad) set it to be used in inference mode """

        if set_training is not None:
            self.train(set_training)

        if no_grad:
            with torch.no_grad():
                out = self.module.fwd_logprob_ratio(*args, move=move, old_logprob=old_logprob, **kwargs)
        else:
            out = self.module.fwd_logprob_ratio(*args, move=move, old_logprob=old_logprob, **kwargs)

        # eventually roll back to default
        if set_training:
            self.train(False)

        return out

    def build_batch(
            self,
            player_ids: List[str],
            game_statesL: List[List[GameState]],
            for_training=   True,
    ) -> DTNS:

        n_moves = len(self.table_moves)

        fwd_keys = ['cards','event_id','cash','pl_id','pl_pos','pl_stats']
        bwd_keys = ['move','reward','allowed_moves']
        batch_keys = [] + fwd_keys if not for_training else [] + fwd_keys + bwd_keys
        batch_keys.append('enc_cnn_state')

        batch = {k: [] for k in batch_keys}

        # for every player
        for pid, game_states in zip(player_ids,game_statesL):

            # build seqs
            seqs = {k: [] for k in batch_keys[:-1]}
            for gs in game_states:

                val = gs.state_orig_data

                # pad cards
                cards = val['cards']
                cards += [52]*(7-len(cards))
                seqs['cards'].append(cards)

                for k in fwd_keys[1:]:
                    seqs[k].append(val[k])

                if for_training:
                    move = gs.move
                    seqs['move'].append(move if move is not None else 0)
                    seqs['reward'].append(gs.reward_sh if gs.reward_sh is not None else 0)
                    allowed_moves = gs.allowed_moves if move is not None else [False] * n_moves
                    seqs['allowed_moves'].append(allowed_moves)

            for k in seqs:
                batch[k].append(seqs[k])
            enc_cnn_state = self._last_upd_state[pid] if for_training else self._last_fwd_state[pid]
            batch['enc_cnn_state'].append(enc_cnn_state)

        # convert data for torch
        batch_conv_torch = {}
        for k in batch_keys[:-1]:
            batch_conv_torch[k] = self.convert(batch[k])
        batch_conv_torch['enc_cnn_state'] = torch.stack(batch['enc_cnn_state'])

        return batch_conv_torch

    def update_policy(self, player_ids:List[str], batch:DTNS) -> None:
        """ + compute logprob + publish """

        # compute old_logprob before update
        batchFWD = {}
        batchFWD.update(batch)
        batchFWD.pop('reward')
        batchFWD.pop('allowed_moves')
        pre_logprob_out = self.fwd_logprob(**batchFWD)
        batch['old_logprob'] = pre_logprob_out['logprob']

        out = self.backward(bypass_data_conv=True, **batch)

        # save UPD states
        for pid,state in zip(player_ids, out['fin_state']):
            self._last_upd_state[pid] = state

        if self._TBwr:

            self._ze_pro_enc.process(zeroes=out['zeroes_enc'], step=self.train_step)
            self._ze_pro_cnn.process(zeroes=out['zeroes_cnn'], step=self.train_step)

            out['batchsize'] = batch['cards'].shape[1] * batch['cards'].shape[0]
            pkeys = [
                'batchsize',
                'currentLR',
                'entropy',
                'loss',
                'loss_actor',
                'loss_critic',
                'loss_nam',
                'gg_norm',
                'gg_norm_clip']
            for l,k in zip('abcdefghijklmnopqrstuvwxyz', pkeys):
                if k in out:
                    self.log_TB(value=out[k], tag=f'backprop/{l}.{k}', step=self.train_step)

            # INFO: stats below may be NOT computed for PG, those are only for info / monitoring / debug
            ### ratio stats + policy histograms

            ratio_out = self.fwd_logprob_ratio(old_logprob=batch['old_logprob'], **batchFWD)
            for l,k in zip('ab', ['approx_kl','clipfracs']):
                self.log_TB(value=ratio_out[k], tag=f'backprop.ratio_full/{l}.{k}', step=self.train_step) # ratio stats "after full batch"
                if k in out: # PPO case
                    self.log_TB(value=out[k], tag=f'backprop.ratio_in/{l}.{k}', step=self.train_step) # average ratio stats "in batch" / while UPD

            self.log_histogram_TB(values=ratio_out['ratio'], tag=f'policy/a.ratio')
            self.log_histogram_TB(values=out['probs'], tag=f'policy/b.probs')
            for l,k in zip('cdef',['reward', 'reward_norm', 'advantage', 'advantage_norm']):
                if k in out:
                    self.log_histogram_TB(values=out[k], tag=f'policy/{l}.{k}', step=self.train_step)

        torch.cuda.empty_cache()


class DMK_MOTorch_A2C(DMK_MOTorch_PG):

    def __init__(self, module_type=ProCNN_DMK_A2C, **kwargs):
        DMK_MOTorch_PG.__init__(self, module_type=module_type, **kwargs)


class DMK_MOTorch_PPO(DMK_MOTorch_PG):

    def __init__(self, module_type=ProCNN_DMK_PPO, **kwargs):

        # INFO: for large PPO updates disables strange CUDA error
        # those backends turn on / off implementations of SDP (scaled dot product attention)
        torch.backends.cuda.enable_mem_efficient_sdp(False) # enables or disables Memory-Efficient Attention
        torch.backends.cuda.enable_flash_sdp(False) # enables or disables FlashAttention
        torch.backends.cuda.enable_math_sdp(True) # enables or disables PyTorch C++ implementation

        DMK_MOTorch_PG.__init__(self, module_type=module_type, **kwargs)

    def backward(
            self,
            bypass_data_conv=   True,
            set_training: bool= True,
            empty_cuda_cache=   True,
            **kwargs
    ) -> DTNS:
        """ backward in PPO mode """

        batch = kwargs

        # until now batch is a dict {key: TNS}, where TNS is a rectangle [len(upd_pid), n_states_upd, feats]
        batch_width = batch['old_logprob'].shape[0]
        mb_size = batch_width // self.minibatch_num
        batch_spl = {k: torch.split(batch[k], mb_size, dim=0) for k in batch} # split along 0 axis into chunks of mb_size
        minibatches = [
            {k: batch_spl[k][ix] for k in batch}            # list of dicts {key: TNS}, where TNS is a minibatch rectangle
            for ix in range(len(batch_spl['old_logprob']))] # num of minibatches

        if self.n_epochs_ppo > 1:
            mb_more = minibatches * (self.n_epochs_ppo - 1)
            random.shuffle(mb_more)
            minibatches = mb_more + minibatches # put more (shuffled) first

        res = {}
        for mb in minibatches:

            out = self.loss(
                bypass_data_conv=   bypass_data_conv,
                set_training=       set_training,
                **mb)
            self.logger.debug(f'> loss() returned: {list(out.keys())}')

            for k in out:
                if k not in res:
                    res[k] = []
                res[k].append(out[k])

            self._opt.zero_grad()           # clear gradients
            out['loss'].backward()          # build gradients

            gnD = self._grad_clipper.clip() # clip gradients, adds 'gg_norm' & 'gg_norm_clip' to out
            for k in gnD:
                if k not in res:
                    res[k] = []
                res[k].append(gnD[k])

            self._opt.step()                # apply optimizer

        #print(f'---batch_width: {batch_width}')
        #for k in res:
        #    print(k, len(res[k]), type(res[k][0]), res[k][0].shape if type(res[k][0]) is torch.Tensor else '')

        ### merge outputs

        res_prep = {}
        for k in [
            'reward',
            'reward_norm',
            'probs',
            'fin_state',
            'zeroes_enc',
            'zeroes_cnn']:
            res_prep[k] = torch.cat(res[k], dim=0)

        for k in [
            'entropy',
            'loss',
            'loss_actor',
            'loss_nam',
            'gg_norm',
            'gg_norm_clip',
            'approx_kl',
            'clipfracs']:
            res_prep[k] = torch.Tensor(res[k]).mean()

        # trim to batch width (from the end.. not shuffled)
        if self.n_epochs_ppo > 1:
            res_prep['fin_state'] = res_prep['fin_state'][-batch_width:]

        self._scheduler.step()  # apply LR scheduler
        self.train_step += 1    # update step

        res_prep['currentLR'] = self._scheduler.get_last_lr()[0]  # INFO: currentLR of the first group is taken

        if empty_cuda_cache:
            torch.cuda.empty_cache()

        return res_prep