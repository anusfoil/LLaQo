"""
Requires Transformer 4.28 and above, implementation may change according the Llama implementation
"""
import os
import logging
import string
from packaging import version

import torch
from torch.nn.functional import pad
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

import transformers

from lavis.common.registry import registry
from lavis.models.lam_models.lam import LAMBase, disabled_train
from lavis.common.dist_utils import download_cached_file
from lavis.common.utils import is_url

# FIXME: reletive path instead
# from utilities import set_logger
import sys

# sys.path.append("/data/home/eey340/WORKPLACE/LAM/engine/models")
sys.path.append("../../../../..")
from audiomae_wrapper import AudioMAE

# log = set_logger(__name__)


@registry.register_model("lam_vicuna_instruct")
class LAMVicunaInstruct(LAMBase):
    """
    LAM Vicuna model.
    Supported model types:
        - vicuna7b
        - vicuna7b-ft
        - vicuna1.5_7b
        - vicuna1.5_7b-ft
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("lam_vicuna_instruct", "vicuna7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "vicuna7b": "configs/models/lam/lam_instruct_vicuna7b.yaml",
        "vicuna7b-ft": "configs/models/lam/lam_instruct_vicuna7b-ft.yaml",
        "llama2_7b": "configs/models/lam/lam_instruct_llama2_7b.yaml",
        "llama2_7b-ft": "configs/models/lam/lam_instruct_llama2_7b-ft.yaml",
        "llama3_8b": "configs/models/lam/lam_instruct_llama3_8b.yaml",
        "vicuna1.5_7b": "configs/models/lam/lam_instruct_vicuna1.5_7b.yaml",
        "vicuna1.5_7b-ft": "configs/models/lam/lam_instruct_vicuna1.5_7b-ft.yaml",
        # "vicuna13b": "configs/models/lam/lam_instruct_vicuna13b.yaml",
    }
    # FIXME
    _VIT_CHECKPOINT_PATH_ = (
        "ckpts/vit.pth"
    )

    def __init__(
        self,
        # audio encoder
        vit_model="vit_base_patch16",
        target_length=1024,
        encoder_hidden_size=768,
        drop_path_rate=0,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        llm_model="",
        prompt="",
        max_txt_len=128,  # 128
        max_output_txt_len=256,
        apply_lemmatizer=False,
        qformer_text_input=True,
        device="cpu",
    ):
        super().__init__()
        transformers_version = version.parse(transformers.__version__)
        assert transformers_version >= version.parse(
            "4.28"
        ), "BLIP-2 Vicuna requires transformers>=4.28"
        from transformers import LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
        from lavis.models.lam_models.modeling_llama4lam import LlamaForCausalLM

        self.tokenizer = self.init_tokenizer(truncation_side="left")

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
            vit_precision=vit_precision if freeze_vit else "fp32",
            device=device,
        )

        if freeze_vit:
            for name, param in self.audio_encoder.model.named_parameters():
                param.requires_grad = False
            self.audio_encoder.model = self.audio_encoder.model.eval()
            self.audio_encoder.model.train = disabled_train
            logging.info("Freeze audio encoder.")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, encoder_hidden_size
        )

        if not qformer_text_input:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            # log.info("Enable Qformer text input.")
            print("Enable Qformer text input.")
            self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None

        print(llm_model)
        if "llama" in llm_model: # loading for llama3
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                llm_model, use_fast=False, truncation_side="left"
            )
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                llm_model, torch_dtype=torch.float16,
            )
        else: # loading for vicuna
            self.llm_tokenizer = LlamaTokenizer.from_pretrained(
                llm_model, use_fast=False, truncation_side="left"
            )
            self.llm_model = LlamaForCausalLM.from_pretrained(
                llm_model, torch_dtype=torch.float16
            )
        self.llm_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.llm_tokenizer.add_special_tokens({"bos_token": "</s>"})
        self.llm_tokenizer.add_special_tokens({"eos_token": "</s>"})
        self.llm_tokenizer.add_special_tokens({"unk_token": "</s>"})
        # NOTE: add <AUDIO> token to indicate the occurance of audio embeddings
        self.llm_tokenizer.add_special_tokens({"cls_token": "<AUDIO>"})
        self.audio_indicator_id = self.llm_tokenizer.cls_token_id

        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        for _, param in self.llm_model.named_parameters():
            param.requires_grad = False

        # NOTE: Using the frozen embedding as placeholder
        self.llm_model.model.embed_tokens.weight[self.audio_indicator_id].zero_()

        self.audio_indicator_token = nn.Parameter(
            torch.zeros(1, 1, self.llm_model.config.hidden_size)
        )
        self.audio_indicator_token.data.normal_(
            mean=0.0, std=self.llm_model.config.initializer_range
        )

        self.llm_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llm_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len  # max text length for LLM
        self.Qformer_max_text_len = 128
        self.max_output_txt_len = max_output_txt_len
        self.prompt = prompt
        prompt_tokens = self.llm_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

        self._lemmatizer = None

        self.qformer_text_input = qformer_text_input

    @staticmethod
    def _parse_input(input_ids, target_id):
        r"""Return the indices in input_ids as per target_id."""
        _input = torch.tensor(input_ids)  # convert list to tensor

        return (_input == target_id).nonzero(as_tuple=True)[0]

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
        device="cpu",
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

    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens["input_ids"].append(
                torch.cat(
                    [
                        input_ids[i][:this_input_ones],
                        output_ids[i][1:],
                        input_ids[i][this_input_ones:],
                    ]
                )
            )
            llm_tokens["attention_mask"].append(
                torch.cat(
                    [
                        input_atts[i][:this_input_ones],
                        output_atts[i][1:],
                        input_atts[i][this_input_ones:],
                    ]
                )
            )
        llm_tokens["input_ids"] = torch.stack(llm_tokens["input_ids"])
        llm_tokens["attention_mask"] = torch.stack(llm_tokens["attention_mask"])
        return llm_tokens, input_part_targets_len

    def _single_forward(self, samples):
        audio = samples["audio"]
        prompt = samples.get("text_input", "")

        aligner_output = self.get_audio_embedding(audio, prompt=prompt)

        inputs_llm = self.llm_proj(aligner_output[:, : self.query_tokens.size(0), :])
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(audio.device)

        # NOTE: add audio indicator token before query embedding pos
        audio_indicator_tokens = self.audio_indicator_token.expand(
            inputs_llm.shape[0], -1, -1
        ).to(audio.device)
        audio_indicator_atts = torch.ones(
            audio_indicator_tokens.size()[:-1], dtype=torch.long
        ).to(audio.device)
        inputs_llm = torch.cat([audio_indicator_tokens, inputs_llm], dim=1)
        atts_llm = torch.cat([audio_indicator_atts, atts_llm], dim=1)

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = "left"
        text_input_tokens = self.llm_tokenizer(
            samples["text_input"],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(audio.device)

        return inputs_llm, atts_llm, text_input_tokens

    def forward(self, samples):
        # For single audio: audio.shape = (B, C, T, F)
        # For few-shot learning: audio.shape = (B, n_audios_per_episode, C, T, F)
        audio = samples["audio"]
        if audio.dim() == 5:
            inputs_llm, atts_llm, text_input_tokens = [], [], []
            B, n_audios_per_episode, _, _, _ = samples["audio"].size()
            for j in range(n_audios_per_episode):
                (
                    this_inputs_llm,
                    this_atts_llm,
                    this_text_input_tokens,
                ) = self._single_forward(
                    {
                        "audio": samples["audio"][:, j, :, :, :],
                        "text_input": samples["text_input"][j],
                    }
                )
                inputs_llm.append(this_inputs_llm)
                atts_llm.append(this_atts_llm)
                text_input_tokens.append(this_text_input_tokens)

            inputs_llm = torch.stack(inputs_llm, dim=1)
            atts_llm = torch.stack(atts_llm, dim=1)

            self.llm_tokenizer.truncation_side = "right"
            text_output_tokens = self.llm_tokenizer(
                [t + self.llm_tokenizer.eos_token for t in samples["text_output"]],
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_output_txt_len,
            ).to(audio.device)

            max_len = max([t.attention_mask.size(dim=-1) for t in text_input_tokens])
            (
                llm_tokens,
                input_part_targets_len,
                batched_episodic_ids,
            ) = self.concat_text_multi_input_output(
                torch.stack(
                    [
                        pad(
                            t.input_ids,
                            (0, max_len - t.input_ids.size(dim=-1)),  # pad to the right
                            value=self.llm_tokenizer.pad_token_id,
                        )
                        for t in text_input_tokens
                    ],
                    dim=1,
                ),
                torch.stack(
                    [
                        pad(
                            t.attention_mask,
                            (0, max_len - t.attention_mask.size(dim=-1)),
                            value=0,
                        )
                        for t in text_input_tokens
                    ],
                    dim=1,
                ),
                text_output_tokens.input_ids,
                text_output_tokens.attention_mask,
            )
            # do not apply loss to the padding
            targets = llm_tokens["input_ids"].masked_fill(
                llm_tokens["input_ids"] == self.llm_tokenizer.pad_token_id, -100
            )

            # do not apply loss to the text input (i.e., instruction) and query tokens
            for i, l in enumerate(input_part_targets_len):
                targets[i][:l] = -100

            batched_episodic_embeds = []
            for episodic_ids in batched_episodic_ids:
                batched_episodic_embeds.append(
                    [self.llm_model.get_input_embeddings()(ids) for ids in episodic_ids]
                )

            attention_mask = llm_tokens["attention_mask"]
            inputs_embeds = []
            # Embed input ids for llm
            for episodic_idx in range(B):
                this_inputs_embeds = []
                for audio_id in range(n_audios_per_episode):
                    _audio_embeds = inputs_llm[episodic_idx][audio_id]
                    _text_embeds = batched_episodic_embeds[episodic_idx][audio_id]
                    # NOTE: <BOS> is the first token instead of audio embeds
                    if audio_id == 0:
                        this_inputs_embeds.append(
                            torch.cat(
                                [_text_embeds[:1], _audio_embeds, _text_embeds[1:]]
                            )
                        )
                    else:
                        this_inputs_embeds.append(
                            torch.cat([_audio_embeds, _text_embeds])
                        )

                this_inputs_embeds.append(
                    batched_episodic_embeds[episodic_idx][n_audios_per_episode]
                )
                inputs_embeds.append(torch.cat(this_inputs_embeds))

            inputs_embeds = torch.stack(inputs_embeds)
        else:
            inputs_llm, atts_llm, text_input_tokens = self._single_forward(samples)

            self.llm_tokenizer.truncation_side = "right"

            text_output_tokens = self.llm_tokenizer(
                [t + self.llm_tokenizer.eos_token for t in samples["text_output"]],
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_output_txt_len,
            ).to(audio.device)

            llm_tokens, input_part_targets_len = self.concat_text_input_output(
                text_input_tokens.input_ids,
                text_input_tokens.attention_mask,
                text_output_tokens.input_ids,
                text_output_tokens.attention_mask,
            )

            # do not apply loss to the padding
            targets = llm_tokens["input_ids"].masked_fill(
                llm_tokens["input_ids"] == self.llm_tokenizer.pad_token_id, -100
            )

            # do not apply loss to the text input (i.e., instruction)
            for i, l in enumerate(input_part_targets_len):
                targets[i][:l] = -100

            # do not apply loss to the query tokens
            empty_targets = (
                torch.ones(atts_llm.size(), dtype=torch.long)
                .to(audio.device)
                .fill_(-100)
            )
            targets = torch.cat([empty_targets, targets], dim=1)
            inputs_embeds = self.llm_model.get_input_embeddings()(
                llm_tokens["input_ids"]
            )
            # NOTE: put <BOS> before audio embeddings
            inputs_embeds = torch.cat(
                [inputs_embeds[:, :1], inputs_llm, inputs_embeds[:, 1:]], dim=1
            )
            attention_mask = torch.cat(
                [
                    llm_tokens["attention_mask"][:, :1],
                    atts_llm,
                    llm_tokens["attention_mask"][:, 1:],
                ],
                dim=1,
            )

        with self.maybe_autocast():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        loss = outputs.loss

        return {"loss": loss}

    def concat_text_multi_input_output(
        self,
        input_ids,
        input_atts,
        output_ids,
        output_atts,
        # 33 "-100"s as place holder
        placeholder_id=-100,
        placeholder_len=33,
    ):
        r"""Concatenate with placeholder multiple text input with a single text output for each sample.
        Typically, in few-shot learning tasks."""
        result_tokens = {"input_ids": [], "attention_mask": []}
        result_input_ids = []  # need this for embedding text input
        B, n_audios_per_episode, _ = input_atts.size()
        placeholder_att = torch.ones(
            placeholder_len, dtype=torch.long, device=input_ids.device
        )
        placeholder_id = placeholder_att.detach().clone().fill_(-100)

        # get the length of (audio_emb, text_input)
        input_part_targets_len = (input_atts == 1).sum(dim=(1, 2))
        input_part_targets_len -= n_audios_per_episode - 1
        input_part_targets_len += placeholder_len * n_audios_per_episode

        for idx in range(B):
            this_input_ids, this_input_att, res_input_ids = [], [], []
            for i in range(n_audios_per_episode):
                _tmp_ids = input_ids[idx][i][input_atts[idx][i] == 1]
                _tmp_att = torch.ones_like(_tmp_ids, dtype=torch.long)
                # Remove <sos> in the firt position if not the first one.
                if i == 0:
                    this_input_ids.append(
                        torch.cat([_tmp_ids[:1], placeholder_id, _tmp_ids[1:]])
                    )
                    this_input_att.append(
                        torch.cat([_tmp_att[:1], placeholder_att, _tmp_att[1:]])
                    )
                    res_input_ids.append(_tmp_ids)
                else:
                    this_input_ids.append(torch.cat([placeholder_id, _tmp_ids[1:]]))
                    this_input_att.append(torch.cat([placeholder_att, _tmp_att[1:]]))
                    res_input_ids.append(_tmp_ids[1:])

            _tmp_ids = output_ids[idx][output_atts[idx] == 1]
            _tmp_att = torch.ones_like(_tmp_ids, dtype=torch.long)
            this_input_ids.append(_tmp_ids[1:])
            this_input_att.append(_tmp_att[1:])
            res_input_ids.append(_tmp_ids[1:])

            result_tokens["input_ids"].append(torch.cat(this_input_ids))
            result_tokens["attention_mask"].append(torch.cat(this_input_att))
            result_input_ids.append(res_input_ids)

        # Pad to the same length then stack
        result_tokens["input_ids"] = self.pad_before_stack(
            result_tokens["input_ids"], padding_value=self.llm_tokenizer.pad_token_id
        )
        result_tokens["attention_mask"] = self.pad_before_stack(
            result_tokens["attention_mask"], padding_value=0
        )
        result_tokens["input_ids"] = torch.stack(result_tokens["input_ids"])
        result_tokens["attention_mask"] = torch.stack(result_tokens["attention_mask"])

        # Pad result input ids to align dim
        n_pad = (result_tokens["attention_mask"] == 0).sum(dim=1)
        for id, n in enumerate(n_pad):
            result_input_ids[id][-1] = torch.cat(
                [
                    result_input_ids[id][-1],
                    torch.tensor(
                        [self.llm_tokenizer.pad_token_id] * n,
                        dtype=torch.long,
                        device=result_input_ids[id][-1].device,
                    ),
                ]
            )

        return result_tokens, input_part_targets_len.tolist(), result_input_ids

    @staticmethod
    def pad_before_stack(input, max_length=None, padding_value=0):
        if max_length == None:
            max_length = max([i.size(dim=-1) for i in input])

        return [
            pad(i, (0, max_length - i.size(dim=-1)), value=padding_value) for i in input
        ]

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1,
        num_captions=1,
        temperature=1,
    ):
        self.llm_tokenizer.padding_side = "left"

        audio = samples["audio"]

        # For single-audio input
        if audio.dim() == 4:
            prompt = samples.get("prompt", "")
            aligner_output = self.get_audio_embedding(audio, prompt)
            inputs_llm = self.llm_proj(
                aligner_output[:, : self.query_tokens.size(0), :]
            )
            atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(
                audio.device
            )

            # NOTE: add audio indicator token before query embedding pos
            audio_indicator_tokens = self.audio_indicator_token.expand(
                inputs_llm.shape[0], -1, -1
            ).to(audio.device)
            audio_indicator_atts = torch.ones(
                audio_indicator_tokens.size()[:-1], dtype=torch.long
            ).to(audio.device)
            inputs_llm = torch.cat([audio_indicator_tokens, inputs_llm], dim=1)
            atts_llm = torch.cat([audio_indicator_atts, atts_llm], dim=1)

            llm_tokens = self.llm_tokenizer(
                prompt, padding="longest", return_tensors="pt"
            ).to(audio.device)

            with self.maybe_autocast():
                inputs_embeds = self.llm_model.get_input_embeddings()(
                    llm_tokens.input_ids
                )
                inputs_embeds = torch.cat(
                    [inputs_embeds[:, :1], inputs_llm, inputs_embeds[:, 1:]], dim=1
                )
                attention_mask = torch.cat(
                    [
                        llm_tokens.attention_mask[:, :1],
                        atts_llm,
                        llm_tokens.attention_mask[:, 1:],
                    ],
                    dim=1,
                )

        # For multi-audio input
        elif audio.dim() == 5:
            bs, n_audios_per_episode, _, _, _ = audio.size()

            # (psudo) shape of prompt = [n_audios_per_episode, B]
            if "prompt" in samples.keys():
                prompt = samples["prompt"]
            else:
                prompt = self.prompt

            if isinstance(prompt[0], str):
                prompt = [[p] * bs for p in prompt]
            else:
                assert (
                    len(prompt[0]) == bs
                ), "The number of prompts must be equal to the batch size."

            inputs_llm, atts_llm, text_input_tokens = [], [], []
            for j in range(n_audios_per_episode):
                (
                    this_inputs_llm,
                    this_atts_llm,
                    this_text_input_tokens,
                ) = self._single_forward(
                    {
                        "audio": samples["audio"][:, j, :, :, :],
                        "text_input": prompt[j],
                    }
                )
                inputs_llm.append(this_inputs_llm)
                atts_llm.append(this_atts_llm)
                text_input_tokens.append(this_text_input_tokens)

            inputs_llm = torch.stack(inputs_llm, dim=1)
            atts_llm = torch.stack(atts_llm, dim=1)
            episodic_batched_embeds = [
                self.llm_model.get_input_embeddings()(batched_tokens["input_ids"])
                for batched_tokens in text_input_tokens
            ]

            inputs_embeds, attention_mask = [], []
            for idx, (emb, batched_tokens) in enumerate(
                zip(episodic_batched_embeds, text_input_tokens)
            ):
                att = batched_tokens["attention_mask"]
                # <BOS> is the first token instead of audio embeds
                if idx == 0:
                    inputs_embeds.append(
                        torch.cat(
                            [emb[:, :1, :], inputs_llm[:, idx, :, :], emb[:, 1:, :]],
                            dim=1,
                        )
                    )
                    attention_mask.append(
                        torch.cat([att[:, :1], atts_llm[:, idx, :], att[:, 1:]], dim=1)
                    )
                else:
                    inputs_embeds.append(
                        torch.cat([inputs_llm[:, idx, :, :], emb[:, 1:, :]], dim=1)
                    )
                    attention_mask.append(
                        torch.cat([atts_llm[:, idx, :], att[:, 1:]], dim=1)
                    )
            inputs_embeds = torch.cat(inputs_embeds, dim=1)
            attention_mask = torch.cat(attention_mask, dim=1)

        else:
            raise Exception(r"Expect audio dim either 4 or 5.")

        outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds.to(torch.float16),
            attention_mask=attention_mask,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            # eos_token_id=self.eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
        )

        outputs[outputs == 0] = 2  # convert output id 0 to 2 (eos_token_id)
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]

        return output_text

    def get_audio_embedding(self, audio, prompt=""):
        B = audio.size(dim=0)

        # Get audio embedding from encoder
        with self.maybe_autocast():
            audio_embeds = self.ln_audio(
                self.audio_encoder.get_audio_embedding(audio)["patch_embedding"]
            )
            audio_atts = torch.ones(audio_embeds.size()[:-1], dtype=torch.long).to(
                audio.device
            )

        n_audio = audio_embeds.size(dim=0)
        n_seg = n_audio // B

        query_tokens = self.query_tokens.expand(n_audio, -1, -1)
        Qtoken_length = query_tokens.size(dim=1)

        if self.qformer_text_input:
            # Fit input text prompt to the Qformer
            if prompt == "":
                prompt = self.prompt
            if isinstance(prompt, str):
                prompt = [prompt] * n_audio
            else:
                assert len(prompt) == B, f"Prompt number should equal to {B}."
                prompt = prompt * n_seg

            text_Qformer = self.tokenizer(
                prompt,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(audio.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                audio.device
            )
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

        # Get aligner output
        with self.maybe_autocast():
            if self.qformer_text_input:
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

        # Reshape to fit the input audio:
        # (n_audio, Qtoken_length, n_dim) -> (B, Qtoken_length * (n_audio // B), n_dim)
        n_seg = n_audio // B
        output = torch.zeros(
            B,
            Qtoken_length * n_seg,
            query_output.last_hidden_state.size(-1),
            device=audio.device,
        )

        for i in range(0, n_seg):
            output[
                :, Qtoken_length * i : Qtoken_length * (i + 1), :
            ] += query_output.last_hidden_state[B * i : B * (i + 1), :Qtoken_length, :]

        return output

    # def predict_answers(
    #     self,
    #     samples,
    #     num_beams=5,
    #     inference_method="generate",
    #     max_len=10,
    #     min_len=1,
    #     num_ans_candidates=128,
    #     answer_list=None,
    #     prompt="",
    #     length_penalty=0,
    #     **kwargs,
    # ):
    #     if isinstance(samples["text_input"], str):
    #         samples["text_input"] = [samples["text_input"]]

    #     if prompt:
    #         if prompt.count("{}") == 2:
    #             if "ocr_tokens" in samples:
    #                 text_input = [
    #                     prompt.format(
    #                         ", ".join(samples["ocr_tokens"][i][:30]),
    #                         samples["text_input"][i],
    #                     )
    #                     for i in range(len(samples["text_input"]))
    #                 ]
    #             elif "choices" in samples:
    #                 text_input = []
    #                 for i in range(len(samples["text_input"])):
    #                     this_choices = [
    #                         f"({string.ascii_lowercase[j]}) {ch}"
    #                         for j, ch in enumerate(samples["choices"][i])
    #                     ]
    #                     this_choices = " ".join(this_choices)
    #                     text_input.append(
    #                         prompt.format(samples["text_input"][i], this_choices)
    #                     )
    #         else:
    #             text_input = [
    #                 prompt.format(question) for question in samples["text_input"]
    #             ]
    #     else:
    #         text_input = samples["text_input"]

    #     samples["prompt"] = text_input

    #     output_text = self.generate(
    #         samples,
    #         num_beams=num_beams,
    #         max_length=max_len,
    #         min_length=min_len,
    #         length_penalty=length_penalty,
    #     )

    #     if "apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]:
    #         output_text = self._lemmatize(output_text)

    #     return output_text

    # def predict_class(
    #     self,
    #     samples,
    #     candidates,
    #     n_segments=1,
    # ):
    #     self.llm_tokenizer.padding_side = "left"

    #     # If candidates is a list of lists, each sample has its candidates, then we need to iterate one by one
    #     if type(candidates[0]) == list:
    #         results = []

    #         for i in range(samples["audio"].size(0)):
    #             this_sample = {
    #                 "audio": samples["audio"][i].unsqueeze(0),
    #                 "prompt": samples["prompt"],
    #             }

    #             if "text_input" in samples.keys():
    #                 this_sample["text_input"] = [samples["text_input"][i]]

    #             if "context" in samples.keys():
    #                 this_sample["context"] = [samples["context"][i]]

    #             if "history" in samples.keys():
    #                 this_sample["history"] = [samples["history"][i]]

    #             if "caption" in samples.keys():
    #                 this_sample["caption"] = [samples["caption"][i]]

    #             this_result = self._predict_class(
    #                 this_sample, candidates[i], n_segments
    #             )
    #             results.append(this_result)

    #         try:
    #             results = torch.cat(results, dim=0)
    #         except:
    #             results = [res.tolist()[0] for res in results]

    #         return results

    #     return self._predict_class(samples, candidates, n_segments)

    # def _predict_class(
    #     self,
    #     samples,
    #     candidates,
    #     n_segments=1,
    # ):
    #     audio = samples["audio"]
    #     prompt = samples["prompt"]

    #     bs = audio.size(0)

    #     if isinstance(prompt, str):
    #         prompt = [prompt] * bs
    #     else:
    #         assert (
    #             len(prompt) == bs
    #         ), "The number of prompts must be equal to the batch size."

    #     if "text_input" in samples.keys():
    #         if type(samples["text_input"][0]) == list:
    #             prompt = [
    #                 prompt[i].format(*samples["text_input"][i])
    #                 for i in range(len(prompt))
    #             ]
    #         else:
    #             prompt = [
    #                 prompt[i].format(samples["text_input"][i])
    #                 for i in range(len(prompt))
    #             ]

    #     # scienceqa
    #     if "context" in samples.keys() and samples["context"] != "":
    #         prompt = [
    #             f'context: {samples["context"][i]}. {prompt[i]}'
    #             for i in range(len(prompt))
    #         ]

    #     # visual dialog
    #     if "history" in samples.keys() and samples["history"][0] != "":
    #         prompt = [
    #             f'dialog history: {samples["history"][i]}\n{prompt[i]}'
    #             for i in range(len(prompt))
    #         ]

    #     if "caption" in samples.keys() and samples["caption"][0] != "":
    #         prompt = [
    #             f'This audio has the caption "{samples["caption"][i]}". {prompt[i]}'
    #             for i in range(len(prompt))
    #         ]

    #     query_tokens = self.query_tokens.expand(bs, -1, -1)
    #     if self.qformer_text_input:
    #         text_Qformer = self.tokenizer(
    #             prompt,
    #             padding="longest",
    #             truncation=True,
    #             max_length=self.Qformer_max_text_len,
    #             return_tensors="pt",
    #         ).to(audio.device)
    #         query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
    #             audio.device
    #         )
    #         Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

    #     with self.maybe_autocast():
    #         audio_embeds = self.ln_audio(self.audio_encoder(audio)["patch_embedding"])
    #     audio_atts = torch.ones(audio_embeds.size()[:-1], dtype=torch.long).to(
    #         audio.device
    #     )

    #     if self.qformer_text_input:
    #         query_output = self.Qformer.bert(
    #             text_Qformer.input_ids,
    #             attention_mask=Qformer_atts,
    #             query_embeds=query_tokens,
    #             encoder_hidden_states=audio_embeds,
    #             encoder_attention_mask=audio_atts,
    #             return_dict=True,
    #         )
    #     else:
    #         query_output = self.Qformer.bert(
    #             query_embeds=query_tokens,
    #             encoder_hidden_states=audio_embeds,
    #             encoder_attention_mask=audio_atts,
    #             return_dict=True,
    #         )

    #     inputs_llm = self.llm_proj(
    #         query_output.last_hidden_state[:, : query_tokens.size(1), :]
    #     )
    #     atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(audio.device)

    #     self.llm_tokenizer.padding_side = "right"
    #     self.llm_tokenizer.truncation_side = "left"
    #     text_input_tokens = self.llm_tokenizer(
    #         prompt,
    #         return_tensors="pt",
    #         padding="longest",
    #         # truncation=True,
    #         # max_length=self.max_txt_len,
    #     ).to(audio.device)

    #     empty_targets = (
    #         torch.ones(atts_llm.size(), dtype=torch.long).to(audio.device).fill_(-100)
    #     )

    #     # self.llm_tokenizer.padding_side = "right"
    #     self.llm_tokenizer.truncation_side = "right"
    #     n_cands = len(candidates)
    #     with self.maybe_autocast(dtype=torch.bfloat16):
    #         all_losses = []
    #         for n in range(n_segments):
    #             seg_len = n_cands // n_segments
    #             if n == (n_segments - 1):
    #                 seg_len = n_cands - seg_len * (n_segments - 1)

    #             start_i = n * (n_cands // n_segments)
    #             end_i = start_i + seg_len

    #             this_output_tokens = self.llm_tokenizer(
    #                 candidates[start_i:end_i],
    #                 return_tensors="pt",
    #                 padding="longest",
    #                 # truncation=True,
    #                 # max_length=self.max_output_txt_len,
    #             ).to(audio.device)

    #             this_input_tokens_ids = text_input_tokens.input_ids.repeat_interleave(
    #                 seg_len, dim=0
    #             )
    #             this_input_tokens_atts = (
    #                 text_input_tokens.attention_mask.repeat_interleave(seg_len, dim=0)
    #             )

    #             this_output_tokens_ids = this_output_tokens.input_ids.repeat(bs, 1)
    #             this_output_tokens_atts = this_output_tokens.attention_mask.repeat(
    #                 bs, 1
    #             )

    #             this_llm_tokens, this_input_targets_len = self.concat_text_input_output(
    #                 this_input_tokens_ids,
    #                 this_input_tokens_atts,
    #                 this_output_tokens_ids,
    #                 this_output_tokens_atts,
    #             )

    #             this_llm_input_ids = this_llm_tokens["input_ids"]
    #             this_llm_atts = this_llm_tokens["attention_mask"]
    #             # this_llm_input_ids = torch.cat([this_input_tokens_ids, this_output_tokens_ids], dim=1)
    #             # this_llm_atts = torch.cat([this_input_tokens_atts, this_output_tokens_atts], dim=1)

    #             inputs_embeds = self.llm_model.get_input_embeddings()(
    #                 this_llm_input_ids
    #             )
    #             inputs_embeds = torch.cat(
    #                 [inputs_llm.repeat_interleave(seg_len, dim=0), inputs_embeds], dim=1
    #             )
    #             attention_mask = torch.cat(
    #                 [atts_llm.repeat_interleave(seg_len, dim=0), this_llm_atts], dim=1
    #             )

    #             this_targets = this_llm_input_ids.masked_fill(
    #                 this_llm_input_ids == self.llm_tokenizer.pad_token_id, -100
    #             )
    #             # this_targets[:, :this_input_tokens_ids.size(1)] = -100
    #             for i, l in enumerate(this_input_targets_len):
    #                 this_targets[i][:l] = -100

    #             this_targets = torch.cat(
    #                 [empty_targets.repeat_interleave(seg_len, dim=0), this_targets],
    #                 dim=1,
    #             )

    #             outputs = self.llm_model(
    #                 inputs_embeds=inputs_embeds,
    #                 attention_mask=attention_mask,
    #                 return_dict=True,
    #                 labels=this_targets,
    #                 reduction="none",
    #             )

    #             loss = outputs.loss

    #             loss = loss.reshape(bs, seg_len)
    #             # output_class_ranks = torch.argsort(loss, dim=-1)
    #             all_losses.append(loss)

    #         all_losses = torch.cat(all_losses, dim=-1)
    #         output_class_ranks = torch.argsort(all_losses, dim=-1)

    #     return output_class_ranks

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "vit_base_patch16")
        target_length = cfg.get("target_length", 1024)
        encoder_hidden_size = cfg.get("encoder_hidden_size", 768)
        num_query_token = cfg.get("num_query_token")
        llm_model = cfg.get("llm_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 128)
        max_output_txt_len = cfg.get("max_output_txt_len", 256)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        qformer_text_input = cfg.get("qformer_text_input", True)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = cls(
            vit_model=vit_model,
            target_length=target_length,
            encoder_hidden_size=encoder_hidden_size,
            drop_path_rate=drop_path_rate,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            llm_model=llm_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            max_output_txt_len=max_output_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            qformer_text_input=qformer_text_input,
            device=device,
        )

        # if qformer_text_input:
        #     # Hard-coded to load from BLIP-2 stage-1 pre-trained model (not ideal)
        #     model.load_from_pretrained(
        #         url_or_filename="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained.pth"
        #     )

        model.load_checkpoint_from_config(cfg)

        return model

    # def load_from_pretrained(self, url_or_filename):
    #     if is_url(url_or_filename):
    #         cached_file = download_cached_file(url_or_filename,
    #                                            check_hash=False,
    #                                            progress=True)
    #         checkpoint = torch.load(cached_file, map_location="cpu")
    #     elif os.path.isfile(url_or_filename):
    #         checkpoint = torch.load(url_or_filename, map_location="cpu")
    #     else:
    #         raise RuntimeError("checkpoint url or path is invalid")

    #     unmatched_layers = []
    #     for name, param in self.named_parameters():
    #         if "crossattention" not in name:
    #             param.data.copy_(checkpoint[name])
    #         else:
    #             unmatched_layers.append(name)

    #     logging.info("Missing keys {}".format(unmatched_layers))
    #     logging.info("load checkpoint from %s" % url_or_filename)