import logging
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import sys

from .audiomae_wrapper import AudioMAE

sys.path.append("..")
from factory import QformerPretrainOutput, unwrap_model, is_master, tensor_move_to

sys.path.append("../../src/lavis")
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from lavis.common.registry import registry


@registry.register_model("lam")
@registry.register_model("lam_feature_extractor")
class LamQformer(Blip2Base):

    def __init__(
        self,
        # audio encoder
        vit_model=None,
        target_length=1024,
        encoder_hidden_size=768,
        drop_path_rate=0.1,
        # vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        freeze_qformer=True,
        device='cpu',
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        if vit_model != None:
            self.audio_encoder, self.ln_audio = self.init_audio_encoder(
                model_name=vit_model,  # vit_base_patch16 by default
                hidden_size=encoder_hidden_size,
                num_classes=527,
                drop_path_rate=drop_path_rate,
                global_pool=True,
                mask_2d=True,
                target_length=target_length,
                use_custom_patch=False,
                device=device)
        if vit_model != None and freeze_vit:
            for name, param in self.audio_encoder.model.named_parameters():
                param.requires_grad = False
            self.audio_encoder.model = self.audio_encoder.model.eval()
            self.audio_encoder.model.train = disabled_train
            logging.info("Freeze audio encoder")
        self.vit_model = vit_model

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, encoder_hidden_size, cross_attention_freq)
        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            logging.info("Freeze Qformer.")
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.Qformer.to(device)
        self.query_tokens = tensor_move_to(self.query_tokens, device)
        self._device = device  # `device` is a predefined method in `Blip2Base`

    @classmethod
    def init_audio_encoder(cls,
                           model_name="vit_base_patch16",
                           hidden_size=768,
                           num_classes=527,
                           drop_path_rate=0.1,
                           global_pool=True,
                           mask_2d=True,
                           target_length=1024,
                           use_custom_patch=False,
                           device='cpu'):
        audio_encoder = AudioMAE(
            model_name=model_name,
            hidden_size=hidden_size,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
            global_pool=global_pool,
            mask_2d=mask_2d,
            target_length=target_length,
            use_custom_patch=use_custom_patch,
            device=device,
        )
        ln_audio = torch.nn.LayerNorm(audio_encoder.hidden_size).to(device)

        return audio_encoder, ln_audio

    def forward(self, batch):
        pass
