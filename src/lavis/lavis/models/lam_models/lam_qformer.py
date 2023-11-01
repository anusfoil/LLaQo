"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from typing import Optional

from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.lam_models.lam import (
    LAMBase,
    compute_sim_matrix,
    disabled_train,
)
from lavis.models.blip_models.blip_outputs import BlipIntermediateOutput, BlipSimilarity

# FIXME: reletive path instead
import sys

sys.path.append("/data/home/acw630/WORKPLACE/LAM/engine/models")
from audiomae_wrapper import AudioMAE


@dataclass
class LAMOutput(ModelOutput):
    # some finetuned models (e.g. BlipVQA) do not compute similarity, thus optional.
    sims: Optional[BlipSimilarity] = None

    intermediate_output: BlipIntermediateOutput = None

    loss: Optional[torch.FloatTensor] = None

    loss_atc: Optional[torch.FloatTensor] = None

    loss_atm: Optional[torch.FloatTensor] = None

    loss_lm: Optional[torch.FloatTensor] = None


@dataclass
class LAMOutputFeatures(ModelOutput):
    """
    Data class of features from BlipFeatureExtractor.

    Args:
        audio_embeds: (torch.FloatTensor) of shape (batch_size, num_patches+1, embed_dim), optional
        audio_features: (torch.FloatTensor) of shape (batch_size, num_patches+1, feature_dim), optional
        text_embeds: (torch.FloatTensor) of shape (batch_size, sequence_length+1, embed_dim), optional
        text_features: (torch.FloatTensor) of shape (batch_size, sequence_length+1, feature_dim), optional

        The first embedding or feature is for the [CLS] token.

        Features are obtained by projecting the corresponding embedding into a normalized low-dimensional space.
    """

    audio_embeds: Optional[torch.FloatTensor] = None
    audio_embeds_proj: Optional[torch.FloatTensor] = None

    text_embeds: Optional[torch.FloatTensor] = None
    text_embeds_proj: Optional[torch.FloatTensor] = None

    multimodal_embeds: Optional[torch.FloatTensor] = None


@registry.register_model("lam")
@registry.register_model("lam_feature_extractor")
class LamQformer(LAMBase):
    """
    LAM first-stage model with Q-former and ViT.
    Supported model types:
        - FIXME
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("lam", "pretrain")
    """

    # FIXME
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/lam/lam_pretrain.yaml",
    }
    _VIT_CHECKPOINT_PATH_ = "/data/EECS-MachineListeningLab/huan/pretrained_models/finetuned.pth"

    def __init__(
        self,
        # audio encoder
        vit_model="vit_base_patch16",
        target_length=1024,
        encoder_hidden_size=768,
        drop_path_rate=0.1,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
        freeze_qformer=True,
        device='cpu',
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.audio_encoder, self.ln_audio = self.init_audio_encoder(
            model_name=vit_model,  # vit_base_patch16 by default
            ckpt_path=self._VIT_CHECKPOINT_PATH_,
            hidden_size=encoder_hidden_size,
            num_classes=527,
            drop_path_rate=drop_path_rate,
            global_pool=True,
            mask_2d=True,
            target_length=target_length,
            use_custom_patch=False,
            vit_precision=vit_precision,
            device=device,
        )

        if freeze_vit:
            for name, param in self.audio_encoder.model.named_parameters():
                param.requires_grad = False
            self.audio_encoder.model = self.audio_encoder.model.eval()
            self.audio_encoder.model.train = disabled_train
            logging.info("Freeze audio encoder.")

        self.vit_model = vit_model

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, encoder_hidden_size, cross_attention_freq)
        self.Qformer.resize_token_embeddings(len(self.tokenizer))

        # we do not need to freeze qformer in the stage 1.
        # if freeze_qformer:
        # for name, param in self.Qformer.named_parameters():
        #     param.requires_grad = False
        # self.Qformer = self.Qformer.eval()
        # logging.info("Freeze Qformer.")

        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        # FIXME: if we need this in BLIP flamework
        # self.Qformer.to(device)
        # self.query_tokens = tensor_move_to(self.query_tokens, device)
        # self._device = device  # `device` is a predefined method in `LAMBase`

        self.audio_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len

    @classmethod
    def init_audio_encoder(
        cls,
        model_name="vit_base_patch16",
        ckpt_path=None,
        hidden_size=768,
        num_classes=527,
        drop_path_rate=0.1,
        global_pool=True,
        mask_2d=True,
        target_length=1024,
        use_custom_patch=False,
        vit_precision="fp16",
        device='cpu',
    ):
        assert model_name == "vit_base_patch16"
        audio_encoder = AudioMAE.create_audiomae(
            target_length=target_length,
            drop_path_rate=drop_path_rate,
            ckpt_path=ckpt_path,
            precision=vit_precision,
            device=device,
        )
        ln_audio = torch.nn.LayerNorm(audio_encoder.hidden_size).to(device)
        audio_encoder.vit_name = model_name

        return audio_encoder, ln_audio

    def forward(self, samples):
        audio = samples["audio"].to(torch.float16)
        text = samples["text_input"]

        with self.maybe_autocast():
            audio_embeds = self.ln_audio(
                self.audio_encoder(audio)["patch_embedding"])

        audio_atts = torch.ones(audio_embeds.size()[:-1],
                                dtype=torch.long).to(audio.device)

        query_tokens = self.query_tokens.expand(audio_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=audio_embeds,
            encoder_attention_mask=audio_atts,
            use_cache=True,
            return_dict=True,
        )

        audio_feats = F.normalize(self.audio_proj(
            query_output.last_hidden_state),
                                  dim=-1)

        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(audio.device)
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        text_feat = F.normalize(self.text_proj(
            text_output.last_hidden_state[:, 0, :]),
                                dim=-1)

        ###============== Audio-text Contrastive ===================###
        audio_feats_all = concat_all_gather(
            audio_feats)  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        text_feat_all = concat_all_gather(
            text_feat)  # [batch_size*num_gpu, embed_dim]

        sim_q2t = torch.matmul(audio_feats.unsqueeze(1),
                               text_feat_all.unsqueeze(-1)).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        # audio-text similarity: aggregate across all query tokens
        sim_a2t, _ = sim_q2t.max(-1)
        sim_a2t = sim_a2t / self.temp

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1),
            audio_feats_all.permute(0, 2, 1)).squeeze()

        # text-audio similarity: aggregate across all query tokens
        sim_t2a, _ = sim_t2q.max(-1)
        sim_t2a = sim_t2a / self.temp  # [batch_size, batch_size*num_gpu]

        rank = dist.get_rank()
        bs = audio.size(0)
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs,
                                 dtype=int).to(audio.device)

        loss_atc = (F.cross_entropy(sim_a2t, targets, label_smoothing=0.1) +
                    F.cross_entropy(sim_t2a, targets, label_smoothing=0.1)) / 2

        ###============== Audio-text Matching ===================###
        text_input_ids_world = concat_all_gather(text_tokens.input_ids)
        text_attention_mask_world = concat_all_gather(
            text_tokens.attention_mask)
        audio_embeds_world = all_gather_with_grad(audio_embeds)
        with torch.no_grad():
            weights_t2i = F.softmax(sim_t2a, dim=1) + 1e-4
            weights_t2i[:, rank * bs:rank * bs + bs].fill_diagonal_(0)
            weights_i2t = F.softmax(sim_a2t, dim=1) + 1e-4
            weights_i2t[:, rank * bs:rank * bs + bs].fill_diagonal_(0)

        # select a negative audio for each text
        audio_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            audio_embeds_neg.append(audio_embeds_world[neg_idx])
        audio_embeds_neg = torch.stack(audio_embeds_neg, dim=0)

        # select a negative text for each audio
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat(
            [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg],
            dim=0)  # pos, pos, neg
        text_atts_all = torch.cat(
            [
                text_tokens.attention_mask, text_tokens.attention_mask,
                text_atts_neg
            ],
            dim=0,
        )

        query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1,
                                                    -1)
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1],
                                    dtype=torch.long).to(audio.device)
        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        audio_embeds_all = torch.cat(
            [audio_embeds, audio_embeds_neg, audio_embeds],
            dim=0)  # pos, neg, pos
        audio_atts_all = torch.ones(audio_embeds_all.size()[:-1],
                                    dtype=torch.long).to(audio.device)

        output_itm = self.Qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=audio_embeds_all,
            encoder_attention_mask=audio_atts_all,
            return_dict=True,
        )

        vl_embeddings = output_itm.last_hidden_state[:, :query_tokens_itm.
                                                     size(1), :]
        vl_output = self.itm_head(vl_embeddings)
        logits = vl_output.mean(dim=1)

        itm_labels = torch.cat(
            [
                torch.ones(bs, dtype=torch.long),
                torch.zeros(2 * bs, dtype=torch.long)
            ],
            dim=0,
        ).to(audio.device)
        loss_atm = F.cross_entropy(logits, itm_labels)

        ##================= Image Captioning ========================##
        decoder_input_ids = text_tokens.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100)

        query_atts = torch.ones(query_tokens.size()[:-1],
                                dtype=torch.long).to(audio.device)
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask],
                                   dim=1)
        lm_output = self.Qformer(
            decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
        )

        loss_lm = lm_output.loss

        return LAMOutput(
            loss=loss_atc + loss_atm + loss_lm,
            loss_atc=loss_atc,
            loss_atm=loss_atm,
            loss_lm=loss_lm,
        )

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - audio (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each audio.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        audio = samples["audio"]
        with torch.cuda.amp.autocast(enabled=(device != torch.device("cpu"))):
            audio_embeds = self.ln_audio(
                self.audio_encoder(audio)["patch_embedding"])

        if not use_nucleus_sampling:
            audio_embeds = audio_embeds.repeat_interleave(num_beams, dim=0)
        else:
            num_beams = 1
        audio_atts = torch.ones(audio_embeds.size()[:-1],
                                dtype=torch.long).to(audio.device)

        model_kwargs = {
            "encoder_hidden_states": audio_embeds,
            "encoder_attention_mask": audio_atts,
        }

        input_ids = (torch.LongTensor(audio.size(0), 1).fill_(
            self.tokenizer.bos_token_id).to(audio.device))
        query_tokens = self.query_tokens.expand(audio_embeds.shape[0], -1, -1)

        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs)
        captions = self.tokenizer.batch_decode(outputs,
                                               skip_special_tokens=True)
        return captions

    def forward_audio(self, audio):
        with torch.cuda.amp.autocast(enabled=(device != torch.device("cpu"))):
            audio_embeds = self.ln_audio(
                self.audio_encoder(audio)["patch_embedding"])

        audio_atts = torch.ones(audio_embeds.size()[:-1],
                                dtype=torch.long).to(audio.device)

        query_tokens = self.query_tokens.expand(audio_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=audio_embeds,
            encoder_attention_mask=audio_atts,
            return_dict=True,
        )
        return query_output.last_hidden_state, audio_embeds

    def forward_text(self, text_tokens):
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return text_output.last_hidden_state[:, 0, :]

    def compute_itm(self, audio_inputs, text_ids, text_atts):
        audio_atts = torch.ones(audio_inputs.size()[:-1],
                                dtype=torch.long).to(audio_inputs.device)
        query_tokens = self.query_tokens.expand(audio_inputs.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1],
                                dtype=torch.long).to(audio_inputs.device)
        attention_mask = torch.cat([query_atts, text_atts], dim=1)
        output_itm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=audio_inputs,
            encoder_attention_mask=audio_atts,
            return_dict=True,
        )
        vl_embeddings = output_itm.last_hidden_state[:, :query_tokens.
                                                     size(1), :]
        itm_logit = self.itm_head(vl_embeddings)
        itm_logit = itm_logit[:, :, 1].mean(dim=1)
        return itm_logit

    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - audio (torch.Tensor): A tensor of shape (B, T) containing the audio.
                    Raw audios should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "audio".
                If "multimodal", return audio features and multimodal features;
                if "text", return text features;
                if "audio", return audio features.
                Default: "multimodal".
        Returns:
            LAMOutputFeatures: A LAMOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """
        audio = samples.get("audio")
        caption = samples.get("text_input")

        # assert mode is one of "audio", "text", "multimodal"
        assert mode in [
            "audio",
            "text",
            "multimodal",
        ], "mode must be one of 'audio', 'text', 'multimodal'"

        # initalize output
        audio_embeds, text_embeds, multimodal_embeds = None, None, None
        audio_features, text_features = None, None

        if mode == "audio":
            assert (audio is not None
                    ), "Image is not provided for mode 'audio' or 'multimodal'"
            # return query features
            with self.maybe_autocast():
                audio_embeds_frozen = self.ln_audio(self.audio_encoder(audio))
            audio_embeds_frozen = audio_embeds_frozen.float()
            audio_atts = torch.ones(audio_embeds_frozen.size()[:-1],
                                    dtype=torch.long).to(self.device)
            query_tokens = self.query_tokens.expand(
                audio_embeds_frozen.shape[0], -1, -1)

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=audio_embeds_frozen,
                encoder_attention_mask=audio_atts,
                return_dict=True,
            )
            audio_embeds = query_output.last_hidden_state
            audio_features = F.normalize(self.audio_proj(audio_embeds), dim=-1)

        elif mode == "text":
            assert (caption is not None
                    ), "text input is None for mode 'text' or 'multimodal'"

            # return text features
            text = self.tokenizer(caption, return_tensors="pt",
                                  padding=True).to(self.device)

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel query features
            with self.maybe_autocast():
                audio_embeds_frozen = self.ln_audio(self.audio_encoder(audio))
            audio_embeds_frozen = audio_embeds_frozen.float()
            audio_atts = torch.ones(audio_embeds_frozen.size()[:-1],
                                    dtype=torch.long).to(self.device)
            query_tokens = self.query_tokens.expand(
                audio_embeds_frozen.shape[0], -1, -1)
            query_atts = torch.ones(query_tokens.size()[:-1],
                                    dtype=torch.long).to(self.device)

            text = self.tokenizer(caption, return_tensors="pt",
                                  padding=True).to(self.device)
            attention_mask = torch.cat([query_atts, text.attention_mask],
                                       dim=1)

            output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=audio_embeds_frozen,
                encoder_attention_mask=audio_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state[:, :query_tokens.
                                                         size(1), :]

        return LAMOutputFeatures(
            audio_embeds=audio_embeds,
            audio_embeds_proj=audio_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "vit_base_patch16")
        target_length = cfg.get("target_length", 1024)
        encoder_hidden_size = cfg.get("encoder_hidden_size", 768)
        num_query_token = cfg.get("num_query_token", 32)
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        # freeze_qformer = cfg.get("freeze_qformer", False)
        # device = cfg.get("device", 'cpu')
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = cls(
            vit_model=vit_model,
            target_length=target_length,
            encoder_hidden_size=encoder_hidden_size,
            drop_path_rate=drop_path_rate,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
            # freeze_qformer=freeze_qformer,
            device=device,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self,
                                  data_loader=data_loader,
                                  k_test=k_test)
