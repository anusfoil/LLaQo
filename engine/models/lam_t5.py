import os
import time
import math
import logging
import json
import torch
import wandb
import torchaudio
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, tensor, Tensor
from contextlib import suppress
from typing import Union, List
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast

import sys

from .lam_qformer import LamQformer

sys.path.append("..")
from factory import tensor_move_to, is_master, QformerPretrainOutput

sys.path.append("../train")
from train_utils import gather_features, ContrastiveLoss, get_metrics

sys.path.append("../../src/laion_clap/src/laion_clap")
from training.train import AverageMeter

sys.path.append("../../src/lavis")
from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration

# def load_model(model_name: str, is_eval: bool = True, device=device("cpu")):
#     PATH = "/data/scratch/acw630/clap_logs/2023_05_15-18_44_45-model_ViT_Qformer-lr_0.0001-b_150-j_1-p_fp32/checkpoints/epoch_10.pt"

#     if model_name == "lam":
#         model = LamQformer(
#             embed_dim=256,
#             freeze_qformer=False,
#             device=device,
#         )

#     model.load_state_dict(torch.load(PATH, map_location=device))

#     if is_eval:
#         model = model.eval()

#     return model


class LamT5(LamQformer):
    """
    LAM w/ T5 model.
    Supported model types:
        - pretrain_flant5xl: pretrained model with FlanT5-XL
        - pretrain_flant5xxl: pretrained model with FlanT5-XXL
    """

    PRETRAINED_MODEL_CONFIG_DICT = {}

    def __init__(
        self,
        pretrained_weights=None,
        use_grad_checkpoint=False,
        t5_model="google/flan-t5-xl",
        prompt="",
        max_txt_len=32,
        apply_lemmatizer=False,
        **kwargs,
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__(**kwargs)

        self.tokenizer = self.init_tokenizer()

        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)
        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = "gelu"
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model, config=t5_config)

        for name, param in self.t5_model.named_parameters():
            param.requires_grad = False
            param.data = param.data.bfloat16()

        self.t5_proj = nn.Linear(self.Qformer.config.hidden_size,
                                 self.t5_model.config.hidden_size).to(
                                     self._device)

        if pretrained_weights is not None:
            ckpt = torch.load(pretrained_weights, map_location='cpu')
            self.Qformer.load_state_dict(ckpt["Qformer.state_dict"])
            self.t5_proj.load_state_dict(ckpt["t5_proj.state_dict"])
            self.Qformer = self.Qformer.eval()
            self.t5_proj = self.t5_proj.eval()

        self.max_txt_len = max_txt_len
        self.prompt = prompt

        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None

        self.Qformer.to(self._device)
        self.t5_proj.to(self._device)
        self.t5_model.to(self._device)

    @torch.no_grad()
    def audio_prompt(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        fbank = samples["audio"].to(self._device)
        B = fbank.size(dim=0)

        with self.maybe_autocast():
            audio_embeds = self.ln_audio(
                self.audio_encoder(fbank)["audio_embedding"])
        audio_embeds = audio_embeds.float()
        audio_atts = torch.ones(audio_embeds.size()[:-1],
                                dtype=torch.long).to(self._device)

        query_tokens = self.query_tokens.expand(audio_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=audio_embeds,
            encoder_attention_mask=audio_atts,
            return_dict=True,
        )
        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1],
                             dtype=torch.long).to(self._device)
        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = self.prompt

        if isinstance(prompt, str):
            prompt = [prompt] * B
        else:
            assert len(
                prompt
            ) == B, "The number of prompts must be equal to the batch size."

        input_tokens = self.t5_tokenizer(prompt,
                                         padding="longest",
                                         return_tensors="pt").to(self._device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        with self.maybe_autocast(dtype=torch.bfloat16):
            inputs_embeds = self.t5_model.encoder.embed_tokens(
                input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)
            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True)

        return [
            output_text[id * num_captions:(id + 1) * num_captions]
            for id in range(B)
        ]

    def preprocess_audio(self, audio_path: Union[str, List]) -> Tensor:
        if isinstance(audio_path, list):
            audios = [
                self.audio_encoder.preprocess_audio(p) for p in audio_path
            ]
            return torch.stack(audios, dim=0)
        else:
            return self.audio_encoder.preprocess_audio(audio_path).unsqueeze(0)
