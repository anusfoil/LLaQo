import os
import time
import math
import logging
import json
import torch
import wandb
import numpy as np
import torch.nn.functional as F
from torch import nn, tensor, Tensor
from contextlib import suppress

import sys

from .lam_qformer import LamQformer

from train_utils import gather_features, ContrastiveLoss, get_metrics

sys.path.append("..")
from factory import tensor_move_to, is_master, QformerPretrainOutput

sys.path.append("../../src/laion_clap/src/laion_clap")
from training.train import AverageMeter


class LamQformerPretrain(LamQformer):

    def __init__(self, embed_dim=256, **kwargs):
        super().__init__(**kwargs)

        self.logit_scale_a = nn.Parameter(
            torch.ones([]) * tensor(1 / 0.07).log().to(self._device))
        self.logit_scale_v = nn.Parameter(
            torch.ones([]) * tensor(1 / 0.07).log().to(self._device))

        # dim of t5 embedding is 2048
        self.t5_proj = nn.Linear(self.Qformer.config.hidden_size,
                                 2048).to(self._device)
        self.audio_proj = nn.Linear(2048, embed_dim).to(self._device)
        self.visual_proj = nn.Linear(2048, embed_dim).to(self._device)

        # self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        # self.temp = nn.Parameter(0.07 * torch.ones([]))
        # self.max_txt_len = max_txt_len

    def feature_forward(self, audio, visual, text=None):
        audio, visual = audio.to(self._device), visual.to(self._device)
        if self.vit_model != None:
            audio_emb = self.ln_audio(self.audio_encoder(audio))
        else:
            # `audio` shape = # (B, 1, 1 + sequence_length, hidden_state)
            audio_emb = audio.squeeze(dim=1)[:, 1:, :]  # remove cls tocken

        audio_atts = torch.ones(audio_emb.size()[:-1],
                                dtype=torch.long).to(self._device)

        query_tokens = self.query_tokens.expand(audio_emb.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=audio_emb,
            encoder_attention_mask=audio_atts,
            use_cache=True,
            return_dict=True,
        )

        input_t5 = self.t5_proj(query_output.last_hidden_state)
        audio_feat = F.normalize(self.audio_proj(input_t5), dim=-1)

        visual_feat = F.normalize(self.visual_proj(visual), dim=-1)

        # text = text.to(self._device)
        # text_tokens = self.tokenizer(
        #     text,
        #     padding="max_length",
        #     truncation=True,
        #     max_length=self.max_txt_len,
        #     return_tensors="pt",
        # ).to(image.device)
        # text_output = self.Qformer.bert(
        #     text_tokens.input_ids,
        #     attention_mask=text_tokens.attention_mask,
        #     return_dict=True,
        # )
        # text_feat = F.normalize(
        #     self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        # )
        return QformerPretrainOutput(
            meta=None,
            audio_feature=audio_feat,
            audio_logit_scale=self.logit_scale_a,
            visual_feature=visual_feat,
            visual_logit_scale=self.logit_scale_v,
            text_feature=None,  # fixit: when adding text for pretraining
        )

    def train_one_epoch(self,
                        data,
                        epoch,
                        optimizer,
                        scaler,
                        scheduler,
                        args,
                        tb_writer=None):
        # device = torch.device(args.device)
        autocast = torch.cuda.amp.autocast if args.precision == "amp" else suppress
        self.Qformer.train()
        loss = ContrastiveLoss(
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod,
            weight_loss_kappa=args.kappa,
        )

        num_batches_per_epoch = data.num_batches
        sample_digits = math.ceil(math.log(data.num_samples + 1, 10))

        loss_m = AverageMeter()
        batch_time_m = AverageMeter()
        data_time_m = AverageMeter()
        end = time.time()

        for i, batch in enumerate(data):
            step = num_batches_per_epoch * epoch + i
            if isinstance(scheduler, dict):
                raise ValueError("scheduler should be a object.")
                for s in scheduler.values():
                    s(step)
            else:
                scheduler(epoch, i)
            meta, audio_emb, visual = batch
            # visual_emb = [v['query_output'] for v in visual]
            visual_emb = [v['input_t5'] for v in visual]
            visual_emb = torch.cat(visual_emb, dim=0)

            data_time_m.update(time.time() - end)
            if isinstance(optimizer, dict):
                for o_ in optimizer.values():
                    o_.zero_grad()
            else:
                optimizer.zero_grad()

            with autocast():
                output = self.feature_forward(
                    audio=audio_emb,
                    visual=visual_emb,
                )

                if args.clap_mlploss:
                    raise TypeError("Do not support mlp loss")
                    # total_loss = loss(audio_features=audio_features,
                    #                   visual_features=visual_features,
                    #                   logit_scale_a=logit_scale_a,
                    #                   logit_scale_t=logit_scale_t,
                    #                   audio_features_mlp=audio_features_mlp,
                    #                   visual_features_mlp=visual_features_mlp)
                else:
                    total_loss = loss(audio_features=output.audio_feature,
                                      visual_features=output.visual_feature,
                                      logit_scale_a=output.audio_logit_scale)

            if isinstance(optimizer, dict):
                if scaler is not None:
                    scaler.scale(total_loss).backward()
                    for o_ in optimizer.values():
                        if args.horovod:
                            o_.synchronize()
                            scaler.unscale_(o_)
                            with o_.skip_synchronize():
                                scaler.step(o_)
                        else:
                            scaler.step(o_)
                    scaler.update()
                else:
                    total_loss.backward()
                    for o_ in optimizer.values():
                        o_.step()
            else:
                if scaler is not None:
                    scaler.scale(total_loss).backward()
                    if args.horovod:
                        optimizer.synchronize()
                        scaler.unscale_(optimizer)
                        with optimizer.skip_synchronize():
                            scaler.step(optimizer)
                    else:
                        scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    optimizer.step()

            # Note: we clamp to 4.6052 = ln(100), as in the original paper.
            with torch.no_grad():
                self.logit_scale_a.clamp_(0, math.log(100))
                # if args.clap_mlploss:
                #     self.logit_scale_t.clamp_(0, math.log(100))

            batch_time_m.update(time.time() - end)
            end = time.time()
            batch_count = i + 1
            if is_master(args) and (i % 100 == 0
                                    or batch_count == num_batches_per_epoch):
                batch_size = len(audio_emb)
                num_samples = batch_count * batch_size * args.world_size
                samples_per_epoch = data.num_samples
                percent_complete = 100.0 * batch_count / num_batches_per_epoch

                # NOTE loss is coarsely sampled, just master node and per log update
                loss_m.update(total_loss.item(), batch_size)
                logit_scale_scalar_a = output.audio_logit_scale.item()
                logit_scale_scalar_t = output.visual_logit_scale.item()
                if isinstance(optimizer, dict):
                    if args.clap_mlploss:
                        logging.info(
                            f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                            f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                            f"Data (t): {data_time_m.avg:.3f} "
                            f"Batch (t): {batch_time_m.avg:.3f} "
                            f"LR: {[o_.param_groups[0]['lr'] for o_ in optimizer.values()]} "
                            f"Logit Scale Audio: {logit_scale_scalar_a:.3f}"
                            f"Logit Scale Text: {logit_scale_scalar_t:.3f}")
                        log_data = {
                            "loss":
                            loss_m.val,
                            "data_time":
                            data_time_m.val,
                            "batch_time":
                            batch_time_m.val,
                            "scale_audio":
                            logit_scale_scalar_a,
                            "scale_text":
                            logit_scale_scalar_t,
                            "lr": [
                                o_.param_groups[0]["lr"]
                                for o_ in optimizer.values()
                            ],
                        }
                    else:
                        logging.info(
                            f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                            f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                            f"Data (t): {data_time_m.avg:.3f} "
                            f"Batch (t): {batch_time_m.avg:.3f} "
                            f"LR: {[o_.param_groups[0]['lr'] for o_ in optimizer.values()]} "
                            f"Logit Scale Audio: {logit_scale_scalar_a:.3f}")
                        log_data = {
                            "loss":
                            loss_m.val,
                            "data_time":
                            data_time_m.val,
                            "batch_time":
                            batch_time_m.val,
                            "scale_audio":
                            logit_scale_scalar_a,
                            "lr": [
                                o_.param_groups[0]["lr"]
                                for o_ in optimizer.values()
                            ],
                        }

                else:
                    if args.clap_mlploss:
                        logging.info(
                            f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                            f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                            f"Data (t): {data_time_m.avg:.3f} "
                            f"Batch (t): {batch_time_m.avg:.3f} "
                            f"LR: {optimizer.param_groups[0]['lr']:5f} "
                            f"Logit Scale Audio: {logit_scale_scalar_a:.3f}"
                            f"Logit Scale Text: {logit_scale_scalar_t:.3f}")

                        # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
                        log_data = {
                            "loss": loss_m.val,
                            "data_time": data_time_m.val,
                            "batch_time": batch_time_m.val,
                            "scale_audio": logit_scale_scalar_a,
                            "scale_text": logit_scale_scalar_t,
                            "lr": optimizer.param_groups[0]["lr"],
                        }
                    else:
                        logging.info(
                            f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                            f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                            f"Data (t): {data_time_m.avg:.3f} "
                            f"Batch (t): {batch_time_m.avg:.3f} "
                            f"LR: {optimizer.param_groups[0]['lr']:5f} "
                            f"Logit Scale Audio: {logit_scale_scalar_a:.3f}")

                        # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
                        log_data = {
                            "loss": loss_m.val,
                            "data_time": data_time_m.val,
                            "batch_time": batch_time_m.val,
                            "scale_audio": logit_scale_scalar_a,
                            "lr": optimizer.param_groups[0]["lr"],
                        }
                for name, val in log_data.items():
                    name = "train/" + name
                    if tb_writer is not None:
                        tb_writer.add_scalar(name, val, step)
                    if args.wandb:
                        assert wandb is not None, "Please install wandb."
                        wandb.log({name: val, "step": step})

                # resetting batch / data time meters per log window
                batch_time_m.reset()
                data_time_m.reset()
        # end for

    @torch.no_grad()
    def evaluate(self, data, epoch, args, tb_writer=None):
        assert not args.clap_mlploss, "Do not support mlp loss."

        metrics = {}

        if not args.parallel_eval:
            if not is_master(args):
                return metrics
        device = torch.device(args.device)
        autocast = torch.cuda.amp.autocast if args.precision == "amp" else suppress
        self.Qformer.eval()

        if is_master(args):
            logging.info('Evaluating...')

        if args.val_data_dir is not None and ((epoch % args.val_frequency) == 0
                                              or epoch == args.epochs):
            num_samples = 0
            samples_per_val = data.num_samples

            # all_audio_features @ all_visual_features will blow up memory and compute very quickly
            eval_info = {}
            if args.clap_mlploss:
                raise ValueError("mlp loss is not supported.")
                # eval_info["all"] = {
                #     "cumulative_loss": 0.0,
                #     "num_samples": 0,
                #     "all_audio_features": [],
                #     "all_visual_features": [],
                #     "all_audio_features_mlp": [],
                #     "all_visual_features_mlp": []
                # }  # cumulative_loss = 0.0
            else:
                eval_info["all"] = {
                    "cumulative_loss": 0.0,
                    "num_samples": 0,
                    "all_audio_features": [],
                    "all_visual_features": []
                }  # cumu
            # all_audio_features, all_visual_features, all_audio_features_mlp, all_visual_features_mlp = [], [], [], []
            for i, batch in enumerate(data):
                meta, audio_emb, visual = batch
                # visual_emb = [v['query_output'] for v in visual]
                visual_emb = [v['input_t5'] for v in visual]
                visual_emb = torch.cat(visual_emb, dim=0)

                # all_names = list(
                #     set([
                #         "-".join(b.split("/")[-3:-1])
                #         for b in batch['__url__']
                #     ]))
                # for name in all_names:
                #     if name not in eval_info.keys():
                #         if args.clap_mlploss:
                #             eval_info[name] = {
                #                 "cumulative_loss": 0.0,
                #                 "num_samples": 0,
                #                 "all_audio_features": [],
                #                 "all_visual_features": [],
                #                 "all_audio_features_mlp": [],
                #                 "all_visual_features_mlp": [],
                #             }
                #         else:
                #             eval_info[name] = {
                #                 "cumulative_loss": 0.0,
                #                 "num_samples": 0,
                #                 "all_audio_features": [],
                #                 "all_visual_features": []
                #             }
                with autocast():
                    output = self.feature_forward(
                        audio=audio_emb,
                        visual=visual_emb,
                    )
                    if args.parallel_eval:
                        # multi-GPU eval
                        if args.clap_mlploss:
                            raise TypeError("MLP loss not supported")
                            # (
                            #     audio_features,
                            #     visual_features,
                            #     audio_features_mlp,
                            #     visual_features_mlp,
                            # ) = gather_features(
                            #     audio_features=audio_features,
                            #     visual_features=visual_features,
                            #     audio_features_mlp=audio_features_mlp,
                            #     visual_features_mlp=visual_features_mlp,
                            #     local_loss=False,
                            #     gather_with_grad=False,
                            #     rank=args.rank,
                            #     world_size=args.world_size,
                            #     use_horovod=args.horovod,
                            #     mlp_loss=args.clap_mlploss)
                        else:
                            (
                                output.audio_feature,
                                output.visual_feature,
                            ) = gather_features(
                                audio_features=output.audio_feature,
                                visual_features=output.visual_feature,
                                local_loss=False,
                                gather_with_grad=False,
                                rank=args.rank,
                                world_size=args.world_size,
                                use_horovod=args.horovod,
                                mlp_loss=args.clap_mlploss)
                    if is_master(args):
                        num_samples += output.audio_feature.shape[0]
                        eval_info["all"]["all_audio_features"].append(
                            output.audio_feature.detach().clone())
                        eval_info["all"]["all_visual_features"].append(
                            output.visual_feature.detach().clone())
                        # for n in [*all_names, "all"]:
                        #     if n == "all":
                        #         eval_info[n]["all_audio_features"].append(
                        #             audio_features.cpu())
                        #         eval_info[n]["all_visual_features"].append(
                        #             visual_features.cpu())
                        # if args.clap_mlploss:
                        #     eval_info[n][
                        #         "all_audio_features_mlp"].append(
                        #             audio_features_mlp.cpu())
                        #     eval_info[n][
                        #         "all_visual_features_mlp"].append(
                        #             visual_features_mlp.cpu())
                        # else:
                        #     idx = np.where(
                        #         np.array([
                        #             "-".join(b.split("/")[-3:-1])
                        #             for b in batch['__url__']
                        #         ]) == n)[0]
                        #     eval_info[n]["all_audio_features"].append(
                        #         audio_features.cpu().index_select(
                        #             0,
                        #             torch.tensor(idx).long()))
                        #     eval_info[n]["all_visual_features"].append(
                        #         visual_features.cpu().index_select(
                        #             0,
                        #             torch.tensor(idx).long()))
                        # if args.clap_mlploss:
                        #     eval_info[n][
                        #         "all_audio_features_mlp"].append(
                        #             audio_features_mlp.cpu(
                        #             ).index_select(
                        #                 0,
                        #                 torch.tensor(idx).long()))
                        #     eval_info[n][
                        #         "all_visual_features_mlp"].append(
                        #             visual_features_mlp.cpu(
                        #             ).index_select(
                        #                 0,
                        #                 torch.tensor(idx).long()))
                        #  print(f'eval step {i}') #  (yusong): for debug

                # cumulative_loss += total_loss * batch_size
                num_samples += args.batch_size
                if is_master(args) and (i % 100) == 0:  # and i != 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]"
                    )
            if is_master(args):
                val_metrics_per_dataset = {}
                for n in eval_info.keys():
                    if args.clap_mlploss:
                        # metrics_single_dataset = get_metrics(
                        #     audio_features=torch.cat(
                        #         eval_info[n]["all_audio_features"]),
                        #     visual_features=torch.cat(
                        #         eval_info[n]["all_visual_features"]),
                        #     logit_scale_a=logit_scale_a.cpu(),
                        #     audio_features_mlp=torch.cat(
                        #         eval_info[n]["all_audio_features_mlp"]),
                        #     visual_features_mlp=torch.cat(
                        #         eval_info[n]["all_visual_features_mlp"]),
                        #     logit_scale_t=logit_scale_t.cpu(),
                        #     mlp_loss=args.clap_mlploss)
                        pass
                    else:
                        metrics_single_dataset = get_metrics(
                            audio_features=torch.cat(
                                eval_info[n]["all_audio_features"]),
                            visual_features=torch.cat(
                                eval_info[n]["all_visual_features"]),
                            logit_scale_a=output.audio_logit_scale.detach(
                            ).clone().to(device),
                            mlp_loss=args.clap_mlploss,
                            batch_size=args.batch_size
                        ) 

                    val_metrics_per_dataset[n] = {
                        n + "/" + k: v
                        for k, v in metrics_single_dataset.items()
                    }
                    metrics.update(val_metrics_per_dataset[n])
                    if "epoch" not in metrics.keys():
                        metrics.update({"epoch": epoch})
        if is_master(args):
            if not metrics:
                return metrics

            logging.info(f"Eval Epoch: {epoch} " + "\n".join([
                "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in m.items()])
                for m in val_metrics_per_dataset.values()
            ]))

            if args.save_logs:
                for name, val in metrics.items():
                    if tb_writer is not None:
                        tb_writer.add_scalar(f"val/{name}", val, epoch)

                with open(os.path.join(args.checkpoint_path, "results.jsonl"),
                          "a+") as f:
                    f.write(json.dumps(metrics))
                    f.write("\n")

            if args.wandb:
                assert wandb is not None, "Please install wandb."
                for name, val in metrics.items():
                    wandb.log({f"val/{name}": val, "epoch": epoch})

            return metrics
        else:
            return metrics
