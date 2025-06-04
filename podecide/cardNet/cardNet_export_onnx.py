import torch
from torchness.onnx_export import get_tested_model, export, play

from podecide.cardNet.cardNet_module import CardNet_MOTorch


if __name__ == "__main__":

    onnx_model_path = '_models/cardNet/cn96.onnx'

    card_net = CardNet_MOTorch(name='cn96_0601_2044')
    card_net_module = card_net.module

    cards = [
        [1,2,3,4,5,6,7],
        [1,2,3,4,5,52,52],
    ]

    inputs = {'cards_A':torch.tensor(cards),'cards_B':torch.tensor(cards)}
    output_names = ['logits_rank','_unusedA','equity','_unusedB','_unusedC']

    get_tested_model(
        model=card_net_module, inputs=inputs,
        model_class=None, model_ckpt_fp=None)

    export(
        model=card_net_module, inputs=inputs, output_names=output_names,
        onnx_model_path=onnx_model_path,
        model_class=None, model_ckpt_fp=None)

    play(onnx_model_path=onnx_model_path, inputs=inputs)