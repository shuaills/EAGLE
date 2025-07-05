import argparse
from transformers import AutoTokenizer
import re
import gc
import os
import torch
import functools
import json
import safetensors.torch

from accelerate.utils import set_seed
from cnets import Model
from configs import EConfig
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
import wandb
import torch.distributed as dist
from lr_scheduler import CosineAnnealingWarmupLR
from torch.optim import AdamW
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType, ShardingStrategy, FullStateDictConfig
from utils import rank_0_priority

from modeling_llama4_17x16_kv import Llama4ForCausalLM, Llama4TextConfig, Llama4TextDecoderLayer

from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

set_seed(0)

torch.backends.cuda.matmul.allow_tf32 = True

user_template = "<|header_start|>user<|header_end|>"
assistant_template = "<|header_start|>assistant<|header_end|>\n\n"

def parse_args():
    parser = argparse.ArgumentParser(description='eagle3')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--config_path", type=str, default="config.json")
    parser.add_argument('--basepath', type=str, default='/home/lyh/weights/hf/llama31chat/8B/')
    parser.add_argument('--trainpath', type=str,
                        default="/home/lyh/code/nlp/developing/vllmbase/vllm/gedata/l318b.jsonl")
    parser.add_argument('--testpath', type=str,
                        default="/home/lyh/code/nlp/developing/vllmbase/vllm/gedata/0318.json")
    parser.add_argument('--savedir', type=str, default='./output')
    return parser.parse_args()


def init_wandb(args):
    wandb.login()
    wandb.init(project="eagle-llama4-sample", config=args)


def init_distributed():
    from datetime import timedelta
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=3))
    torch.cuda.set_device(dist.get_rank())

def build_dataset_rank(
        tokenizer, datapath
):

    ds = load_dataset('json', data_files=datapath)
    ds = ds['train']
    ds = ds.shuffle(seed=42)
    ds1 = ds
    original_columns1 = ds1.column_names
    num_proc = 8

    def preprocess_function(examples):
        new_examples = {
            "attention_mask": [],
            "input_ids": [],
            "loss_mask": []
        }
        for i in range(len(examples['conversations'])):
            # NOTE(yinfan98): need fix, apply chat template will not help you add system prompt
            messages = [
                {"role": "system",
                 "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
            ]
            convroles = ["user", "assistant"]
            source = examples['conversations'][i]
            if not source:
                continue
            if source[0]["role"] != "user":
                # Skip the first one if it is not from human
                source = source[1:]
            for j, sentence in enumerate(source):
                role = sentence["role"]
                assert role == convroles[j % 2], f"{i}"
                # if sentence["from"]=="gpt":
                #     sentence["value"]=" "+sentence["value"]
                messages.append(
                    {"role": role, "content": sentence["content"]}
                )
            conversation = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            if not tokenizer.pad_token_id:
                tokenizer.pad_token_id = tokenizer.unk_token_id

            input_ids = tokenizer(
                conversation,
                return_tensors="pt",
                max_length=2048,
                add_special_tokens=False,
            ).input_ids[0]
            loss_mask = torch.ones_like(input_ids)

            sep = assistant_template

            total_len = len(input_ids)

            sep2 = user_template
            turns = conversation.split(sep2)

            turns[1] = turns[0] + sep2 + turns[1]
            turns = turns[1:]

            cur_len = 1
            loss_mask[:cur_len] = 0
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids)

                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

                # Ignore the user instructions
                if i == 0:
                    loss_mask[cur_len: cur_len + instruction_len - 2] = 0
                else:
                    loss_mask[cur_len - 3: cur_len + instruction_len + 1] = 0
                cur_len += turn_len
                if i != 0:
                    cur_len += 3
                # cur_len+=2

                # if i != 0 and not tokenizer.legacy:
                #     # The legacy and non-legacy modes handle special tokens differently
                #     cur_len -= 1

            loss_mask[cur_len:] = 0
            attention_mask = torch.ones_like(loss_mask)

            # new_examples["conversation"].append(conversation)
            new_examples["input_ids"].append(input_ids[None, :])
            new_examples["loss_mask"].append(loss_mask[None, :])
            new_examples["attention_mask"].append(attention_mask[None, :])

        return new_examples

    ds1 = ds1.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns1,
        load_from_cache_file=False
    )


    ds1.set_format(type="torch")
    return ds1

class DataCollatorWithPadding:

    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        # padding_tensor = torch.zeros(B, N - n, S,dtype=intensors.dtype)
        padding_tensor = torch.zeros(B, N - n, S, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item['input_ids'].shape[1] for item in features)
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        batch_attention_mask = torch.cat(
            [self.paddingtensor2D(item['attention_mask'], max_length) for item in features])
        batch_loss_mask = torch.cat(
            [self.paddingtensor2D(item['loss_mask'], max_length) for item in features])

        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        return batch

def prepare_dataloaders(args, tokenizer):
    traindataset = build_dataset_rank(tokenizer, args.trainpath)
    testdataset = build_dataset_rank(tokenizer, args.testpath)

    # train
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    train_sampler = DistributedSampler(traindataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(traindataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4,
                              pin_memory=True,
                              collate_fn=DataCollatorWithPadding())

    # test
    test_sampler = DistributedSampler(testdataset, num_replicas=world_size, rank=rank, shuffle=False)
    test_loader = DataLoader(testdataset, batch_size=args.batch_size, sampler=test_sampler, num_workers=4, pin_memory=True,
                             collate_fn=DataCollatorWithPadding())
    return train_loader, test_loader, train_sampler, test_sampler

def get_files_containing_params(checkpoint_folder: str, param_prefix: str="language_model.") -> List[str]:
    """
    根据参数名前缀找到包含这些参数的safetensors文件
    
    Args:
        checkpoint_folder: checkpoint文件夹路径
        param_prefix: 参数名前缀，例如 "language_model."
    
    Returns:
        包含指定参数前缀的safetensors文件的完整路径列表
    """
    index_file = os.path.join(checkpoint_folder, "model.safetensors.index.json")
    
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"未找到 index.json 文件: {index_file}")
    
    with open(index_file, 'r', encoding='utf-8') as f:
        index_data = json.load(f)
    
    if 'weight_map' not in index_data:
        raise ValueError("index.json 中没有 weight_map 字段")
    
    # 找到包含指定参数前缀的文件
    target_files = set()
    matching_params = []
    
    for param_name, file_name in index_data['weight_map'].items():
        if param_name.startswith(param_prefix):
            target_files.add(file_name)
            matching_params.append(param_name)
    
    # 生成完整路径
    full_paths = [os.path.join(checkpoint_folder, f) for f in sorted(target_files)]
    
    # 验证文件是否存在
    existing_paths = []
    for path in full_paths:
        if os.path.exists(path):
            existing_paths.append(path)
        else:
            print(f"警告: 文件不存在 {path}")
    
    return existing_paths


def load_checkpoint(checkpoint_path):
    merged_state_dict: dict[str, torch.Tensor] = {}
    
    checkpoint_folder = get_files_containing_params(checkpoint_path)
    
    for cpt_idx, cpt_path in enumerate(checkpoint_folder):
        state_dict = safetensors.torch.load_file(cpt_path, device="cpu")
        for key, value in state_dict.items():
            # Ensure that the state dict is a flat dict of keys and tensors. Breaking this assumption
            # will break recipe code
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"Expected all values in the state dict to be torch.Tensor. "
                    f"Found {type(value)} instead."
                )
            clean_key = key
            if key.startswith("language_model."):
                clean_key = key[len("language_model."):]
                clean_key = "target_model." + clean_key
                merged_state_dict[clean_key] = value

        # delete the state_dict to free up memory; TODO check if this del is needed
        del state_dict
        gc.collect()
    
    return merged_state_dict

def main():
    args = parse_args()
    print(args)
    init_distributed()

    if dist.get_rank() == 0:
        init_wandb(args)

    # build tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.basepath)

    # build data
    train_loader, test_loader, train_sampler, test_sampler = prepare_dataloaders(args, tokenizer)

    # build model and apply fsdp
    config = EConfig.from_pretrained(args.config_path)
    
    # build target model
    llama4_config = Llama4TextConfig.from_pretrained("/tmp/Llama-4-Scout-17B-16E-Instruct")
    with torch.device("meta"):
        target_model = Llama4ForCausalLM(llama4_config).to(torch.bfloat16)

    llama_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            Llama4TextDecoderLayer,
        },
    )

    model = Model(config, path=args.basepath, load_emb=True, load_head=True, target_model=target_model, type="multimodal").to(torch.bfloat16)

    model = FSDP(
        model,
        use_orig_params=True,
        auto_wrap_policy=llama_auto_wrap_policy,     
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
    )
    dist.barrier()

    dict = load_checkpoint(args.basepath)
    
    # cfg = FullStateDictConfig(rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        model.load_state_dict(dict, strict=False)
    dist.barrier()
    model = model.cuda()

    
    with rank_0_priority():
        model.scandata(args.trainpath, args.basepath, user_template, assistant_template)
    print("Finished scanning data")

    # build loss, optimizer, lr scheduler
    criterion = nn.SmoothL1Loss(reduction="none")
    num_epochs = args.num_epochs
    optimizer = AdamW(model.parameters(), lr=1e-4)
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * 0.02)
    scheduler = CosineAnnealingWarmupLR(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)

    # build dataloaders
    os.makedirs(args.savedir, exist_ok=True)

    for epoch in range(num_epochs):
        print(f"Now training epoch {epoch}")
        train_sampler.set_epoch(epoch+1)
        model.train()
        epoch_acces = [[] for _ in range(model.length)]
        epoch_plosses = [[] for _ in range(model.length)]

        for batch_idx, data in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                plosses, vlosses, acces = model(input_ids=data["input_ids"].cuda(),
                                                    attention_mask=data["attention_mask"].cuda(),
                                                    loss_mask=data["loss_mask"].cuda(),
                                                    )

            # calculate ploss
            ploss_weight = [0.8 ** i for i in range(len(plosses))]
            ploss = sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
            loss = ploss

            # run backward step
            loss.backward()
            optimizer.step()
            scheduler.step()

            if dist.get_rank() == 0:
                logdict = {"train/lr": optimizer.param_groups[0]["lr"]}
                for i in range(len(plosses)):
                    logdict[f"train/ploss_{i}"] = plosses[i].item()
                for i in range(len(acces)):
                    logdict[f"train/acc_{i}"] = acces[i]
                wandb.log(logdict)
            epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
            epoch_plosses = [epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))]


        for i in range(len(epoch_acces)):
            acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
            dist.all_reduce(acc_i)
            acc_i = acc_i / dist.get_world_size()
            acc_i = acc_i.item()

            if dist.get_rank() == 0:
                wandb.log({f"train/epochacc_{i}": acc_i})
                print(f"Train Epoch [{epoch + 1}/{num_epochs}], position {i},  Acc: {acc_i:.2f}")

        for i in range(len(epoch_plosses)):
            loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
            dist.all_reduce(loss_i)
            loss_i = loss_i / dist.get_world_size()
            loss_i = loss_i.item()

            if dist.get_rank() == 0:
                wandb.log({f"train/epochploss_{i}": loss_i})
                print(f"Train Epoch [{epoch + 1}/{num_epochs}], position {i}, pLoss: {loss_i:.2f}")

        epoch_acces = [[] for _ in range(model.length)]
        epoch_plosses = [[] for _ in range(model.length)]

        # run testing
        # TODO: make this an argument
        if epoch % 1 == 0:
            for batch_idx, data in enumerate(tqdm(test_loader)):
                # forward pass
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        plosses, vlosses, acces = model(input_ids=data["input_ids"].cuda(),
                                                            attention_mask=data["attention_mask"].cuda(),
                                                            loss_mask=data["loss_mask"].cuda(),
                                                            )
                    epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
                    epoch_plosses = [epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))]

            for i in range(len(epoch_acces)):
                acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
                dist.all_reduce(acc_i)
                acc_i = acc_i / dist.get_world_size()
                acc_i = acc_i.item()

                if dist.get_rank() == 0:
                    wandb.log({f"test/epochacc_{i}": acc_i})
                    print(f"Test Epoch [{epoch + 1}/{num_epochs}], position {i},  Acc: {acc_i:.2f}")

            for i in range(len(epoch_plosses)):
                loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
                dist.all_reduce(loss_i)
                loss_i = loss_i / dist.get_world_size()
                loss_i = loss_i.item()

                if dist.get_rank() == 0:
                    wandb.log({f"test/epochploss_{i}": loss_i})
                    print(f"Test Epoch [{epoch + 1}/{num_epochs}], position {i}, pLoss: {loss_i:.2f}")
        
        # TODO: make this an argument
        if epoch % 1 == 0:
            # Save the draft model to CHECKPOINT_DIR
            state_dict = {}
            for name, param in model.named_parameters():
                if "target" not in name:
                    state_dict[name] = param.data

            if dist.get_rank() == 0:
                torch.save(state_dict, f"{args.savedir}/model_{epoch}.pth")
            dist.barrier()

if __name__ == "__main__":
    main()
