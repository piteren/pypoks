from torchness.motorch import MOTorch

from pypoks_envy import DMK_MODELS_FD



# adds possibility to load cardNet ckpt while init
class DMK_MOTorch(MOTorch):

    def __init__(
            self,
            save_topdir=                DMK_MODELS_FD,
            load_cardnet_pretrained=    False,
            **kwargs):

        # INFO: load_cardnet_pretrained will not be saved with POINT of MOTorch, but it is intended
        MOTorch.__init__(self, save_topdir=save_topdir, **kwargs)

        if load_cardnet_pretrained:
            self.load_cardnet_pretrained()

    def load_cardnet_pretrained(self):
        self.module.card_net.load_ckpt()
        self._log.info(f'{self.name} loaded card_net pretrained checkpoint')