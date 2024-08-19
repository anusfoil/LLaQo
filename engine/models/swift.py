import logging
import contextlib

import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast

import transformers

# FIXME: reletive path instead
from utilities import set_logger
import sys

sys.path.append("/data/home/acw630/WORKPLACE/LAM/engine/models")
from audiomae_wrapper import AudioMAE

sys.path.append("/data/home/acw630/WORKPLACE/LAM/src/lavis")
from lavis.models.lam_models.lam import LAMBase, disabled_train

log = set_logger(__name__)
# class SWIFT(LAMBase):
#     r"""An extension of (multi-modal) llm for hearing."""

#     _VIT_CHECKPOINT_PATH_ = "/data/EECS-MachineListeningLab/jinhua/pretrained_models/finetuned.pth"

#     def init_swift(
#         self,
#         # audio encoder config
#         audio_encoder_name="vit_base_patch16",
#         target_length=1024,
#         encoder_hidden_size=768,
#         drop_path_rate=0,
#         vit_precision="fp16",
#         freeze_audio_encoder=True,
#         # Qformer config
#         num_query_token=32,
#         enable_text_input=True,
#         prompt="",
#         max_txt_len=128,
#         # llm projection config
#         map_to_llm=False,
#         llm_hidden_dim=4096,
#         device='cpu',
#     ):
#         r"""Init Qformer only in this case."""
#         LAMBase.__init__(self)

#         self.swift_tokenizer = self.init_tokenizer(truncation_side="left")

#         self.swift_audio_encoder, self.swift_ln_audio = self._init_audio_encoder(
#             model_name=audio_encoder_name,  # vit_base_patch16 by default
#             ckpt_path=self._VIT_CHECKPOINT_PATH_,
#             hidden_size=encoder_hidden_size,
#             num_classes=527,
#             drop_path_rate=drop_path_rate,
#             global_pool=True,
#             mask_2d=True,
#             target_length=target_length,
#             use_custom_patch=False,
#             vit_precision=vit_precision,
#             device=device,
#         )

#         if freeze_audio_encoder:
#             for _, param in self.swift_audio_encoder.model.named_parameters():
#                 param.requires_grad = False
#             self.swift_audio_encoder.model = self.swift_audio_encoder.model.eval(
#             )
#             self.swift_audio_encoder.model.train = disabled_train
#             logging.info("Freeze audio encoder.")

#         self.swift_Qformer, self.swift_query_tokens = self.init_Qformer(
#             num_query_token,
#             encoder_hidden_size,
#         )

#         if not enable_text_input:
#             self.swift_Qformer.bert.embeddings.word_embeddings = None
#             self.swift_Qformer.bert.embeddings.position_embeddings = None
#             for layer in self.swift_Qformer.bert.encoder.layer:
#                 layer.output = None
#                 layer.intermediate = None
#         else:
#             self.swift_Qformer.resize_token_embeddings(
#                 len(self.swift_tokenizer))

#         self.swift_Qformer.cls = None

#         self.prompt = prompt
#         self.max_txt_len = max_txt_len
#         self.enable_text_input = enable_text_input

#         if map_to_llm:
#             self.audio_indicator_token = self._init_audio_indicator_token(
#                 # model_obj.llm_tokenizer,
#                 # model_obj.llm_model,
#             )
#             self.swift_llm_proj = self._init_llm_projection_layer(
#                 self.swift_Qformer.config.hidden_size, llm_hidden_dim)

#         self.to(device)

#     def get_audio_embedding(
#         self,
#         audio,
#         prompt="",
#     ):
#         if prompt == "":
#             prompt = self.prompt

#         bs = audio.size(0)

#         if isinstance(prompt, str):
#             prompt = [prompt] * bs
#         else:
#             assert len(
#                 prompt) == bs, f"The number of prompts not equal to {bs}."
#         query_tokens = self.swift_query_tokens.expand(bs, -1, -1)
#         # FIXME: +1 if audio_indicator_token exists
#         audio_token_length = query_tokens.size(1)

#         if self.enable_text_input:
#             text_Qformer = self.swift_tokenizer(
#                 prompt,
#                 padding='longest',
#                 truncation=True,
#                 max_length=self.max_txt_len,
#                 return_tensors="pt",
#             ).to(audio.device)
#             query_atts = torch.ones(query_tokens.size()[:-1],
#                                     dtype=torch.long).to(audio.device)
#             Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],
#                                      dim=1)

#         with self.maybe_autocast():
#             audio_embeds = self.swift_ln_audio(
#                 self.swift_audio_encoder(audio)["patch_embedding"])
#             audio_atts = torch.ones(audio_embeds.size()[:-1],
#                                     dtype=torch.long).to(audio.device)

#             if self.enable_text_input:
#                 query_output = self.swift_Qformer.bert(
#                     text_Qformer.input_ids,
#                     attention_mask=Qformer_atts,
#                     query_embeds=query_tokens,
#                     encoder_hidden_states=audio_embeds,
#                     encoder_attention_mask=audio_atts,
#                     return_dict=True,
#                 )
#             else:
#                 query_output = self.swift_Qformer.bert(
#                     query_embeds=query_tokens,
#                     encoder_hidden_states=audio_embeds,
#                     encoder_attention_mask=audio_atts,
#                     return_dict=True,
#                 )
#         return query_output.last_hidden_state[:, :audio_token_length, :]

#     def get_text_embedding(
#         self,
#         text,
#         padding="max_length",
#     ):
#         r"""
#             e.g.,
#             text_emb = self.get_text_embedding(text)[:, 0, :]
#         """
#         text_tokens = self.swift_tokenizer(
#             text,
#             padding=padding,
#             truncation=True,
#             max_length=self.max_txt_len,
#             return_tensors="pt",
#         )

#         text_output = self.swift_Qformer.bert(
#             text_tokens.input_ids,
#             attention_mask=text_tokens.attention_mask,
#             return_dict=True,
#         )

#         return {
#             "class_embedding": text_output.last_hidden_state[:, 0, :],
#             "text_embedding": text_output.last_hidden_state[:, 1:-1, :],
#             "all_embedding": text_output.last_hidden_state,
#         }

#     @classmethod
#     def _init_audio_encoder(
#         cls,
#         model_name="vit_base_patch16",
#         ckpt_path=None,
#         hidden_size=768,
#         num_classes=527,
#         drop_path_rate=0.1,
#         global_pool=True,
#         mask_2d=True,
#         target_length=1024,
#         use_custom_patch=False,
#         vit_precision="fp16",
#         device='cpu',
#     ):
#         assert model_name == "vit_base_patch16"
#         audio_encoder = AudioMAE.create_audiomae(
#             target_length=target_length,
#             drop_path_rate=drop_path_rate,
#             ckpt_path=ckpt_path,
#             precision=vit_precision,
#             device=device,
#         )
#         ln_audio = torch.nn.LayerNorm(audio_encoder.hidden_size).to(device)
#         audio_encoder.vit_name = model_name

#         return audio_encoder, ln_audio

#     @classmethod
#     def _init_audio_indicator_token(
#         cls,
#         # llm_tokenizer,
#         # llm_model,
#         llm_hidden_size=4096,
#     ):
#         # assert llm_tokenizer.cls_token == None, "`cls_token` is occupied, try another token in this case."

#         # llm_tokenizer.add_special_tokens({'cls_token': '<AUDIO>'})
#         # audio_indicator_id = llm_tokenizer.cls_token_id

#         # llm_model.resize_token_embeddings(len(llm_tokenizer))

#         # audio_indicator_token = nn.Parameter(
#         #     torch.zeros(1, 1, llm_model.config.hidden_size))
#         # audio_indicator_token.data.normal_(
#         #     mean=0.0, std=llm_model.config.initializer_range)

#         # llm_model.audio_indicator_id = audio_indicator_id
#         # llm_model.audio_indicator_token = audio_indicator_token

#         # return llm_tokenizer, llm_model
#         audio_indicator_token = nn.Parameter(torch.zeros(
#             1, 1, llm_hidden_size))
#         audio_indicator_token.data.normal_(mean=0.0, std=0)
#         return audio_indicator_token

#     @classmethod
#     def _init_llm_projection_layer(cls, input_dim, llm_hidden_dim):
#         return nn.Linear(input_dim, llm_hidden_dim)

#     def load_checkpoint(self, local_ckpt_path):
#         """
#         Load from a finetuned checkpoint.

#         This should expect no mismatch in the model keys and the checkpoint keys.
#         """
#         checkpoint = torch.load(local_ckpt_path, map_location="cpu")

#         if "model" in checkpoint.keys():
#             _state_dict = checkpoint["model"]
#         else:
#             _state_dict = checkpoint

#         state_dict = {
#             f"swift_{name}": param
#             for name, param in _state_dict.items()
#         }

#         msg = self.load_state_dict(state_dict, strict=False)

#         log.info("Missing keys {}".format(msg.missing_keys))
#         log.info("load checkpoint from %s" % local_ckpt_path)

#         return msg

#     def maybe_autocast(self, dtype=torch.float16):
#         # if on cpu, don't use autocast
#         # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
#         enable_autocast = (self.device != torch.device("cpu")
#                            or self.device != "cpu")

#         if enable_autocast:
#             return torch.cuda.amp.autocast(dtype=dtype)
#         else:
#             return contextlib.nullcontext()

#     def compute_similarity(
#         self,
#         emb: torch.Tensor,
#         emb2: torch.Tensor,
#         normalization: bool = True,
#     ):
#         r"""Compute similarity between text and audio embeddings."""

#         if normalization:
#             emb = F.normalize(emb, dim=-1)
#             emb2 = F.normalize(emb2, dim=-1)

#         return emb @ emb2.T

#     @classmethod
#     def extend(cls, model_obj, ckpt_path):
#         r"""Extend a built llm for hearing."""
#         from processors import _fbankProcessor

#         def _extend_instance(obj, ext_cls):
#             base_cls = obj.__class__
#             base_cls_name = obj.__class__.__name__
#             obj.__class__ = type(base_cls_name, (ext_cls, base_cls), {})

#         _extend_instance(model_obj, cls)
#         swift = cls()
#         swift.init_swift(map_to_llm=True)

#         swift.load_checkpoint(ckpt_path)
#         print(swift.audio_indicator_token)

#         model_obj.__dict__.update({"swift": swift})
#         model_obj.swift.audio_processor = _fbankProcessor.build_processor({})


class SWIFT(LAMBase):
    r"""An extension of (multi-modal) llm for hearing."""

    _VIT_CHECKPOINT_PATH_ = "/data/EECS-MachineListeningLab/jinhua/pretrained_models/finetuned.pth"

    def init_swift(
        self,
        # audio encoder config
        audio_encoder_name="vit_base_patch16",
        target_length=1024,
        encoder_hidden_size=768,
        drop_path_rate=0,
        vit_precision="fp16",
        freeze_audio_encoder=True,
        # Qformer config
        num_query_token=32,
        enable_text_input=True,
        prompt="",
        max_txt_len=128,
        # llm projection config
        map_to_llm=False,
        llm_hidden_dim=4096,
        device='cpu',
    ):
        r"""Init Qformer only in this case."""
        LAMBase.__init__(self)

        self.tokenizer = self.init_tokenizer(truncation_side="left")

        self.audio_encoder, self.ln_audio = self._init_audio_encoder(
            model_name=audio_encoder_name,  # vit_base_patch16 by default
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

        if freeze_audio_encoder:
            for _, param in self.audio_encoder.model.named_parameters():
                param.requires_grad = False
            self.audio_encoder.model = self.audio_encoder.model.eval()
            self.audio_encoder.model.train = disabled_train
            logging.info("Freeze audio encoder.")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token,
            encoder_hidden_size,
        )

        if not enable_text_input:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.Qformer.resize_token_embeddings(len(self.tokenizer))

        self.Qformer.cls = None

        self.prompt = prompt
        self.max_txt_len = max_txt_len
        self.enable_text_input = enable_text_input

        if map_to_llm:
            self.audio_indicator_token = self._init_audio_indicator_token(
                # model_obj.llm_tokenizer,
                # model_obj.llm_model,
            )
            self.llm_proj = self._init_llm_projection_layer(
                self.Qformer.config.hidden_size, llm_hidden_dim)

        self.to(device)

    def get_audio_embedding(
        self,
        audio,
        prompt="",
    ):
        if prompt == "":
            prompt = self.prompt

        bs = audio.size(0)
        # audio.to(self.device)

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(
                prompt) == bs, f"The number of prompts not equal to {bs}."
        query_tokens = self.query_tokens.expand(bs, -1, -1)
        # FIXME: +1 if audio_indicator_token exists
        audio_token_length = query_tokens.size(1)

        if self.enable_text_input:
            text_Qformer = self.tokenizer(
                prompt,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(audio.device)
            query_atts = torch.ones(query_tokens.size()[:-1],
                                    dtype=torch.long).to(audio.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],
                                     dim=1)

        with self.maybe_autocast():
            audio_embeds = self.ln_audio(
                self.audio_encoder(audio)["patch_embedding"])
            audio_atts = torch.ones(audio_embeds.size()[:-1],
                                    dtype=torch.long).to(audio.device)

            if self.enable_text_input:
                query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=audio_embeds,
                    encoder_attention_mask=audio_atts,
                    return_dict=True,
                )
            else:
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=audio_embeds,
                    encoder_attention_mask=audio_atts,
                    return_dict=True,
                )
        return query_output.last_hidden_state[:, :audio_token_length, :]

    def get_text_embedding(
        self,
        text,
        padding="max_length",
    ):
        r"""
            e.g.,
            text_emb = self.get_text_embedding(text)[:, 0, :]
        """
        text_tokens = self.tokenizer(
            text,
            padding=padding,
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)

        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )

        return {
            "class_embedding": text_output.last_hidden_state[:, 0, :],
            "text_embedding": text_output.last_hidden_state[:, 1:, :],
            "all_embedding": text_output.last_hidden_state,
        }

    @classmethod
    def _init_audio_encoder(
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

    @classmethod
    def _init_audio_indicator_token(
        cls,
        # llm_tokenizer,
        # llm_model,
        llm_hidden_size=4096,
    ):
        # assert llm_tokenizer.cls_token == None, "`cls_token` is occupied, try another token in this case."

        # llm_tokenizer.add_special_tokens({'cls_token': '<AUDIO>'})
        # audio_indicator_id = llm_tokenizer.cls_token_id

        # llm_model.resize_token_embeddings(len(llm_tokenizer))

        # audio_indicator_token = nn.Parameter(
        #     torch.zeros(1, 1, llm_model.config.hidden_size))
        # audio_indicator_token.data.normal_(
        #     mean=0.0, std=llm_model.config.initializer_range)

        # llm_model.audio_indicator_id = audio_indicator_id
        # llm_model.audio_indicator_token = audio_indicator_token

        # return llm_tokenizer, llm_model
        audio_indicator_token = nn.Parameter(torch.zeros(
            1, 1, llm_hidden_size))
        audio_indicator_token.data.normal_(mean=0.0, std=0)
        return audio_indicator_token

    @classmethod
    def _init_llm_projection_layer(cls, input_dim, llm_hidden_dim):
        return nn.Linear(input_dim, llm_hidden_dim)

    def load_checkpoint(self, local_ckpt_path):
        """
        Load from a finetuned checkpoint.

        This should expect no mismatch in the model keys and the checkpoint keys.
        """
        checkpoint = torch.load(local_ckpt_path, map_location="cpu")

        # if "model" in checkpoint.keys():
        #     _state_dict = checkpoint["model"]
        # else:
        #     _state_dict = checkpoint

        # state_dict = {
        #     f"swift_{name}": param
        #     for name, param in _state_dict.items()
        # }

        state_dict = checkpoint["model"] if "model" in checkpoint.keys(
        ) else checkpoint
        msg = self.load_state_dict(state_dict, strict=False)

        log.info("Missing keys {}".format(msg.missing_keys))
        log.info("load checkpoint from %s" % local_ckpt_path)

        return msg

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = (self.device != torch.device("cpu")
                           or self.device != "cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def compute_similarity(
        self,
        emb: torch.Tensor,
        emb2: torch.Tensor,
        normalization: bool = True,
    ):
        r"""Compute similarity between text and audio embeddings."""

        if normalization:
            emb = F.normalize(emb, dim=-1)
            emb2 = F.normalize(emb2, dim=-1)

        return torch.matmul(emb, emb2.mT)

    @classmethod
    def extend(cls, model_obj, ckpt_path):
        r"""Extend a built llm for hearing."""
        from processors import _fbankProcessor

        def _extend_instance(obj, ext_cls):
            base_cls = obj.__class__
            base_cls_name = obj.__class__.__name__
            obj.__class__ = type(base_cls_name, (ext_cls, base_cls), {})

        _extend_instance(model_obj, cls)
        swift = cls()
        swift.init_swift(map_to_llm=True)

        swift.load_checkpoint(ckpt_path)

        model_obj.__dict__.update({"swift": swift})
        model_obj.swift.audio_processor = _fbankProcessor.build_processor({})


if __name__ == "__main__":
    ckpt_path = "/data/EECS-MachineListeningLab/jinhua/lam/check_point/Pretrain_stage1_audioset/20230728062/checkpoint_270000.pth"
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    swift = SWIFT(map_to_llm=True, device=device)
    swift.load_checkpoint(ckpt_path)
