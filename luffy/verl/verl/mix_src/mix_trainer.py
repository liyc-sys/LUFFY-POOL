# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict
from collections import defaultdict, Counter

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto, DataProtoItem
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance

import torch

from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer, 
    Role, 
    ResourcePoolManager, 
    WorkerType, 
    _timer, 
    # compute_data_metrics, 
    compute_timing_metrics, 
    dataprotoitem_to_dataproto, 
    # compute_advantage, 
    reduce_metrics
)
from verl.utils.torch_functional import masked_mean

import traceback


def ensure_tensor(x, device=None):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if device is not None and isinstance(x, torch.Tensor):
        x = x.to(device)
    return x

# directly copied from verl/trainer/ppo/ray_trainer.py
def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics

def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, grpo_use_std=True):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index,
                                                                        use_std=grpo_use_std)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo_split':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        prefix_mask = data.batch['prefix_mask']
        on_policy_mask = ~prefix_mask.any(-1)
        from .mix_core_alg import compute_grpo_outcome_advantage_split
        advantages, returns = compute_grpo_outcome_advantage_split(
            token_level_rewards=token_level_rewards,
            eos_mask=response_mask,
            index=index,
            on_policy_mask=on_policy_mask,
            use_std=grpo_use_std)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
        
    elif adv_estimator == 'reinforce':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                             eos_mask=response_mask,
                                                                             index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'reinforce_plus_plus':
        token_level_rewards = data.batch['token_level_rewards']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data



class MIXRayPPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # 修改双池子相关配置：分离容量和更新阈值
        self.pool_A = []  # 存储全部失败的prompt数据
        self.pool_B = []  # 存储其他prompt数据
        
        # 更新大小（达到这个数量就触发更新）
        self.pool_A_update_size = config.trainer.get('pool_A_update_size', 128)
        self.pool_B_update_size = config.trainer.get('pool_B_update_size', 128)
        
        # 池子容量（最大能存储的数量，是更新大小的3倍）
        self.pool_A_capacity = config.trainer.get('pool_A_capacity', self.pool_A_update_size * 3)
        self.pool_B_capacity = config.trainer.get('pool_B_capacity', self.pool_B_update_size * 3)
        
        # 池子A的替换策略配置
        self.pool_A_replacement_strategy = config.trainer.get('pool_A_replacement_strategy', 'resample')  # 'none', 'replace_half', 'resample'
        
        print(f"[双池子策略] 池子A - 容量: {self.pool_A_capacity}, 更新阈值: {self.pool_A_update_size}")
        print(f"[双池子策略] 池子B - 容量: {self.pool_B_capacity}, 更新阈值: {self.pool_B_update_size}")
        print(f"[双池子策略] 池子A替换策略: {self.pool_A_replacement_strategy}")

        # 同时更新的代码配置
        # 同时更新模式配置
        self.use_simultaneous_update = config.trainer.get('use_simultaneous_update', False)
        
        # 调试
        self.use_simultaneous_update = True

        self.simultaneous_size = config.trainer.get('simultaneous_size', 64)

        print(f"[双池子策略] 同时更新模式: {'启用' if self.use_simultaneous_update else '禁用'}")
        if self.use_simultaneous_update:
            print(f"[双池子策略] 同时更新批次大小: {self.simultaneous_size}")

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        self._create_dataloader()

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'grpo_split':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'reinforce':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'reinforce_plus_plus':
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _create_dataloader(self):
        # TODO: we have to make sure the batch size is divisible by the dp size
        from torch.utils.data import DataLoader, SequentialSampler
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        from .rl_dataset_with_target import RLHFDatasetWithTarget
        self.train_dataset = RLHFDatasetWithTarget(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True, return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error',
                                         max_target_length=self.config.actor_rollout_ref.rollout.max_prefix_len,
                                         filter_targets=self.config.data.get('filter_targets', False),
                                         sample_target_ratio=self.config.data.get('sample_target_ratio', 1.0))

        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            from verl.mix_src.rl_dataset_with_target import ResumableRandomSampler
            sampler = ResumableRandomSampler(data_source=self.train_dataset)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           drop_last=True,
                                           collate_fn=collate_fn,
                                           sampler=sampler)
        
        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=len(self.val_dataset),
                                         shuffle=True,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def process_batch_for_pools(self, batch_dict):
        """处理单个batch，生成rollout和计算reward，返回处理后的数据"""
        batch: DataProto = DataProto.from_single_dict(batch_dict)
        
        # 在generation之前保存原始prompt信息
        original_prompt_input_ids = batch.batch['input_ids'].clone()  # [batch_size, prompt_len]
        original_prompt_attention_mask = batch.batch['attention_mask'].clone()  # [batch_size, prompt_len]
        if 'position_ids' in batch.batch:
            original_prompt_position_ids = batch.batch['position_ids'].clone()
        else:
            original_prompt_position_ids = None
        
        # 保存原始的tgt_input_ids（如果存在）
        original_tgt_input_ids = None
        if 'tgt_input_ids' in batch.batch:
            original_tgt_input_ids = batch.batch['tgt_input_ids'].clone()
        
        # pop those keys for generation
        gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids', 'tgt_input_ids'])
        gen_batch.meta_info['global_steps'] = self.global_steps

        # 临时修改prefix_strategy为zeroPrefixRatio以实现纯净采样
        original_prefix_strategy = self.config.actor_rollout_ref.rollout.prefix_strategy
        try:
            # 通过meta_info传递零前缀信号
            gen_batch.meta_info['force_zero_prefix'] = True
            print(f"[双池子-采样] 通过meta_info强制使用零前缀采样")
            
            # generate a batch
            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
            
        finally:
            # 确保恢复原来的策略
            self.config.actor_rollout_ref.rollout.prefix_strategy = original_prefix_strategy

        # ===== 添加缺失的数据处理步骤 =====
        
        # This code matches a prompt ID with its N responses.
        batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                dtype=object)
        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        
        # 关键步骤：合并生成的数据
        batch = batch.union(gen_batch_output)

        # 将原始prompt信息添加到最终batch中（需要repeat以匹配rollout数量）
        n_rollouts = self.config.actor_rollout_ref.rollout.n
        batch_size = len(original_prompt_input_ids)
        
        # 重复原始prompt信息以匹配rollout数量
        batch.batch['original_prompt_input_ids'] = original_prompt_input_ids.repeat_interleave(n_rollouts, dim=0)
        batch.batch['original_prompt_attention_mask'] = original_prompt_attention_mask.repeat_interleave(n_rollouts, dim=0)
        
        if original_prompt_position_ids is not None:
            batch.batch['original_prompt_position_ids'] = original_prompt_position_ids.repeat_interleave(n_rollouts, dim=0)
        
        if original_tgt_input_ids is not None:
            batch.batch['original_tgt_input_ids'] = original_tgt_input_ids.repeat_interleave(n_rollouts, dim=0)

        # compute values
        if self.use_critic:
            values = self.critic_wg.compute_values(batch)
            batch = batch.union(values)

        # compute scores using reward model and/or reward function
        if self.use_rm:
            reward_tensor = self.rm_wg.compute_rm_score(batch)
            batch = batch.union(reward_tensor)

        # ===== 现在可以安全地访问responses了 =====
        
        # 调试代码开始
        # print(f"[调试] gen_batch_output keys: {list(gen_batch_output.batch.keys())}")
        # if 'responses' in batch.batch:  # 注意：这里改为batch.batch而不是gen_batch_output.batch
        #     responses = batch.batch['responses']
        #     print(f"[调试] 生成的responses shape: {responses.shape}")
        #     print(f"[调试] responses统计 - min: {responses.min()}, max: {responses.max()}")
        #     print(f"[调试] 第一个response: {responses[0]}")
        #     print(f"[调试] responses中非pad token数量: {(responses != self.tokenizer.pad_token_id).sum()}")
            
        #     # 检查是否所有response都是pad token
        #     if (responses == self.tokenizer.pad_token_id).all():
        #         print("[调试] 警告: 所有responses都是pad token!")
        
        # # 检查prefix_mask
        # if 'prefix_mask' in batch.batch:  # 这里也改为batch.batch
        #     prefix_mask = batch.batch['prefix_mask']
        #     print(f"[调试] prefix_mask shape: {prefix_mask.shape}")
        #     print(f"[调试] prefix_mask中True的数量: {prefix_mask.sum()}")

        # 在reward_fn调用前，详细检查responses内容
        # responses = batch.batch['responses']
        # print(f"\n[内容质量调试] ===== 检查responses实际内容 =====")
        
        # # 检查前几个responses的解码结果
        # for i in range(min(3, len(responses))):
        #     try:
        #         # 先打印prompt，再打印response
        #         print(f"[内容质量调试] Prompt: {self.tokenizer.decode(batch.batch['input_ids'][i], skip_special_tokens=True)}")
        #         decoded = self.tokenizer.decode(responses[i], skip_special_tokens=True)
        #         print(f"[内容质量调试] Response {i} 解码结果: '{decoded}...'")
        #         print(f"[内容质量调试] Response {i} 长度: {len(decoded)} 字符")
                
        #         # 检查是否包含有意义的内容
        #         if len(decoded.strip()) == 0:
        #             print(f"[内容质量调试] 警告: Response {i} 解码后为空!")
        #         elif len(decoded.strip()) < 10:
        #             print(f"[内容质量调试] 警告: Response {i} 解码后很短: '{decoded}'")
                    
        #     except Exception as e:
        #         print(f"[内容质量调试] Response {i} 解码失败: {e}")
        
        # # 检查responses的token分布
        # unique_tokens = torch.unique(responses)
        # print(f"[内容质量调试] responses中unique token数量: {len(unique_tokens)}")
        # print(f"[内容质量调试] 前20个unique tokens: {unique_tokens[:20]}")
        
        # # 检查是否大量重复token
        # if len(unique_tokens) < 100:
        #     print(f"[内容质量调试] 警告: unique token数量较少，可能生成质量差")

        # 最后调用reward_fn  
        reward_tensor = self.reward_fn(batch) # [bsz, l], only the last valid token has reward  
        batch.batch['token_level_scores'] = reward_tensor  
        
        # 设置完整显示选项  
        # torch.set_printoptions(threshold=float('inf'), linewidth=200)  
        
        print(f"[双池子] Reward tensor shape: {reward_tensor.shape}")  
        print(f"[双池子] Non-zero rewards count: {(reward_tensor != 0).sum().item()}")  
        
        # 找到并显示所有非零位置和对应的值  
        # non_zero_mask = reward_tensor != 0  
        # if non_zero_mask.any():  
        #     # 获取所有非零位置的坐标和值  
        #     non_zero_indices = torch.nonzero(reward_tensor, as_tuple=False)  
        #     non_zero_values = reward_tensor[non_zero_mask]  
            
            # print(f"[双池子] 所有非零奖励值:")  
            # for idx, (pos, value) in enumerate(zip(non_zero_indices, non_zero_values)):  
            #     batch_idx, seq_pos = pos[0].item(), pos[1].item()  
            #     print(f"  Sample {batch_idx}, Position {seq_pos}: {value.item():.6f}")  
        
            # 保存详细信息到文件  
            # with open('reward_tensor_detailed.txt', 'w') as f:  
            #     f.write(f"Reward tensor shape: {reward_tensor.shape}\n")  
            #     f.write(f"Non-zero count: {(reward_tensor != 0).sum().item()}\n\n")  
                
            #     f.write("=== 非零奖励值详情 ===\n")  
            #     for idx, (pos, value) in enumerate(zip(non_zero_indices, non_zero_values)):  
            #         batch_idx, seq_pos = pos[0].item(), pos[1].item()  
            #         f.write(f"Sample {batch_idx}, Position {seq_pos}: {value.item():.6f}\n")  
                
            #     f.write("\n=== 每行完整内容 ===\n")  
            #     for i in range(reward_tensor.shape[0]):  
            #         # 找到每行的非零位置  
            #         row_non_zero = torch.nonzero(reward_tensor[i]).squeeze()  
            #         if row_non_zero.numel() > 0:  
            #             f.write(f"Row {i} (有奖励): {reward_tensor[i]}\n")  
            #             if row_non_zero.dim() == 0:  # 只有一个非零值  
            #                 pos = row_non_zero.item()  
            #                 value = reward_tensor[i, pos].item()  
            #                 f.write(f"  -> Position {pos}: {value:.6f}\n")  
            #             else:  # 多个非零值  
            #                 for pos in row_non_zero:  
            #                     value = reward_tensor[i, pos].item()  
            #                     f.write(f"  -> Position {pos.item()}: {value:.6f}\n")  
            #         else:  
            #             f.write(f"Row {i} (无奖励): 全零\n")  
        # else:  
        #     print("[双池子] 所有奖励都是零!")  
        #     with open('reward_tensor_detailed.txt', 'w') as f:  
        #         f.write("所有奖励值都是0，可能的原因:\n")  
        #         f.write("1. 奖励函数配置问题\n")  
        #         f.write("2. 输入数据问题\n")  
        #         f.write("3. ground truth 缺失\n")  
        
        return batch, reward_tensor

    def categorize_prompts_to_pools(self, batch, reward_tensor):
        """根据reward结果，将prompt分类到不同的池子"""
        uids = batch.non_tensor_batch['uid']
        unique_uids = np.unique(uids)
        
        # 确定奖励值
        if self.config.data.reward_impl_version == 0:
            fail_value = 0
            success_value = 1
        elif self.config.data.reward_impl_version == 1:
            # fail_value还可能是-1，请对代码进行对应的修改
            fail_value = -0.5
            success_value = 1
        elif self.config.data.reward_impl_version == 2:
            fail_value = 0
            success_value = 1
        elif self.config.data.reward_impl_version == 3:
            fail_value = 0
            success_value = 1
        elif self.config.data.reward_impl_version == 4:
            fail_value = 0
            success_value = 1
        else:
            raise ValueError(f'Invalid reward implementation version: {self.config.data.reward_impl_version}')

        pool_A_candidates = []  # 全部失败的prompt
        pool_B_candidates = []  # 其他prompt
        
        for uid in unique_uids:
            uid_mask = uids == uid
            uid_mask = ensure_tensor(uid_mask, reward_tensor.device)
            uid_rewards = reward_tensor[uid_mask].sum(-1)  # Sum rewards for each sequence
            
            if self.config.data.reward_impl_version == 1:
                # 对于版本1，fail_value可能是-1或-0.5，这里硬编码
                # if (uid_rewards == -1).all() or (uid_rewards == -0.5).all() or (uid_rewards == 1).all():
                # if True:
                    # print("[注意] 为了快速填满off池子，进入了调试模式")
                
                # if random.random() < 0.5:
                #     print("[注意] 为了快速填满off池子，进入了调试模式")

                if torch.isin(uid_rewards, torch.tensor([-1.0, -0.5], device=uid_rewards.device)).all():
                    # 全部失败，加入池子A
                    pool_A_candidates.append(uid)
                else:
                    # 至少有一个不是失败，加入池子B
                    pool_B_candidates.append(uid)
            else:
                # 其他版本使用原来的逻辑
                if (uid_rewards == fail_value).all():
                    # 全部失败，加入池子A
                    pool_A_candidates.append(uid)
                else:
                    # 至少有一个不是失败，加入池子B
                    pool_B_candidates.append(uid)
        
        # 将符合条件的prompt数据添加到对应池子（检查容量限制）
        for uid in pool_A_candidates:
            if len(self.pool_A) < self.pool_A_capacity:
                uid_mask = uids == uid
                prompt_data = self.extract_prompt_data(batch, uid_mask)
                self.pool_A.append(prompt_data)
        
        for uid in pool_B_candidates:
            if len(self.pool_B) < self.pool_B_capacity:
                uid_mask = uids == uid
                prompt_data = self.extract_prompt_data(batch, uid_mask)
                self.pool_B.append(prompt_data)
        
        print(f"[双池子] 池子A: {len(self.pool_A)}/{self.pool_A_capacity} (更新阈值: {self.pool_A_update_size})")
        print(f"[双池子] 池子B: {len(self.pool_B)}/{self.pool_B_capacity} (更新阈值: {self.pool_B_update_size})")
        print(f"[双池子] 当前batch中全部失败prompt: {len(pool_A_candidates)}, 其他prompt: {len(pool_B_candidates)}")

    # def extract_prompt_data(self, batch, uid_mask):
    #     """从batch中提取单个prompt的所有数据"""
    #     prompt_data = {}
        
    #     # 提取tensor数据
    #     for key, value in batch.batch.items():
    #         if isinstance(value, torch.Tensor) and value.shape[0] == len(uid_mask):
    #             # 使用原有的ensure_tensor函数确保设备一致
    #             device_mask = ensure_tensor(uid_mask, value.device)
    #             prompt_data[key] = value[device_mask].clone()
        
    #     # 对于numpy索引，确保是numpy数组
    #     if isinstance(uid_mask, torch.Tensor):
    #         numpy_mask = uid_mask.cpu().numpy()
    #     else:
    #         numpy_mask = ensure_tensor(uid_mask).cpu().numpy() if hasattr(ensure_tensor(uid_mask), 'cpu') else np.array(uid_mask)
        
    #     # 提取non_tensor数据
    #     prompt_data['uid'] = batch.non_tensor_batch['uid'][numpy_mask]
        
    #     # 提取meta_info
    #     prompt_data['meta_info'] = batch.meta_info.copy()
        
    #     return prompt_data
    def extract_prompt_data(self, batch, uid_mask):
        """从batch中提取单个prompt的所有数据"""
        prompt_data = {}
        
        # 提取tensor数据
        for key, value in batch.batch.items():
            if isinstance(value, torch.Tensor) and value.shape[0] == len(uid_mask):
                # 使用原有的ensure_tensor函数确保设备一致
                device_mask = ensure_tensor(uid_mask, value.device)
                prompt_data[key] = value[device_mask].clone()
        
        # 对于numpy索引，确保是numpy数组
        if isinstance(uid_mask, torch.Tensor):
            numpy_mask = uid_mask.cpu().numpy()
        else:
            numpy_mask = ensure_tensor(uid_mask).cpu().numpy() if hasattr(ensure_tensor(uid_mask), 'cpu') else np.array(uid_mask)
        
        # 提取non_tensor数据
        prompt_data['uid'] = batch.non_tensor_batch['uid'][numpy_mask]
        
        # 提取meta_info并保存原始non_tensor_batch数据
        prompt_data['meta_info'] = batch.meta_info.copy()
        
        # ===== 新增：保存原始non_tensor_batch数据以供重新采样使用 =====
        if 'original_non_tensor_batch' not in prompt_data['meta_info']:
            # 保存第一个样本的原始non_tensor_batch数据
            first_idx = numpy_mask.nonzero()[0][0] if numpy_mask.any() else 0
            original_non_tensor = {}
            for key, value in batch.non_tensor_batch.items():
                if key != 'uid':  # uid是动态生成的，不保存
                    if isinstance(value, np.ndarray) and len(value) > first_idx:
                        original_non_tensor[key] = value[first_idx]
                    else:
                        original_non_tensor[key] = value
            prompt_data['meta_info']['original_non_tensor_batch'] = original_non_tensor
        
        return prompt_data

    def combine_pool_data(self, pool, n_samples=None):
        """将池子中的数据合并成一个batch"""
        if not pool:
            return None
        
        # 如果指定了样本数量，只取前n_samples个
        selected_pool = pool[:n_samples] if n_samples is not None else pool
        
        combined_dict = {}
        combined_uid = []
        
        # 合并tensor数据
        tensor_keys = set()
        for prompt_data in selected_pool:
            tensor_keys.update([k for k, v in prompt_data.items() if isinstance(v, torch.Tensor) and k != 'uid' and k != 'meta_info'])
        
        for key in tensor_keys:
            tensors = [prompt_data[key] for prompt_data in selected_pool if key in prompt_data]
            if tensors:
                # 确保所有tensor在同一设备上
                target_device = tensors[0].device
                aligned_tensors = [ensure_tensor(t, target_device) for t in tensors]
                combined_dict[key] = torch.cat(aligned_tensors, dim=0)
        
        # 合并uid
        for prompt_data in selected_pool:
            if 'uid' in prompt_data:
                combined_uid.extend(prompt_data['uid'])
        
        # 添加uid到combined_dict中，这样from_single_dict可以正确处理
        combined_dict['uid'] = np.array(combined_uid, dtype=object)
        
        # 使用from_single_dict创建DataProto，这样会自动设置batch_size等属性
        batch = DataProto.from_single_dict(combined_dict)
        
        # 设置meta_info
        batch.meta_info = selected_pool[0]['meta_info'].copy()
        
        return batch

    def replace_half_rollouts_with_external(self, batch):
        """策略1：将每个prompt的一半rollout替换成外部rollout"""
        print("[双池子] 执行策略1: 替换一半rollout为外部rollout")
        
        # 检查是否有外部trace数据
        if 'original_tgt_input_ids' not in batch.batch:
            print("[双池子] 警告: 没有外部trace数据，跳过替换")
            return batch
        
        # 获取基本信息
        uids = batch.non_tensor_batch['uid']
        unique_uids = np.unique(uids)
        n_rollouts = self.config.actor_rollout_ref.rollout.n
        half_rollouts = n_rollouts // 2
        
        # 为每个prompt随机选择一半的rollout进行替换
        for uid in unique_uids:
            uid_mask = uids == uid
            uid_indices = np.where(uid_mask)[0]
            
            if len(uid_indices) == 0:
                continue
                
            # 获取该prompt的原始外部trace
            first_idx = uid_indices[0]
            external_trace = batch.batch['original_tgt_input_ids'][first_idx]  # [tgt_len]
            
            # 随机选择一半的rollout索引进行替换
            import random
            replace_indices = random.sample(list(uid_indices), min(half_rollouts, len(uid_indices)))
            
            print(f"[双池子] 为prompt {uid} 替换 {len(replace_indices)} 个rollout")
            
            # 对选中的rollout进行实际替换
            for idx in replace_indices:
                try:
                    # 获取当前rollout的原始prompt部分
                    original_prompt_ids = batch.batch['original_prompt_input_ids'][idx]  # [prompt_len]
                    original_prompt_mask = batch.batch['original_prompt_attention_mask'][idx]  # [prompt_len]
                    
                    # 构建新的完整序列：prompt + external_trace
                    # 需要处理长度匹配问题
                    device = original_prompt_ids.device
                    external_trace = external_trace.to(device)
                    
                    # 确保external_trace不为空且有效
                    if len(external_trace) == 0 or external_trace.sum() == 0:
                        print(f"[双池子] 警告: prompt {uid} 的外部trace为空，跳过替换")
                        continue
                    
                    # 移除external_trace中的padding tokens
                    if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
                        # 移除padding
                        valid_mask = external_trace != self.tokenizer.pad_token_id
                        external_trace = external_trace[valid_mask]
                    
                    if len(external_trace) == 0:
                        print(f"[双池子] 警告: 移除padding后外部trace为空，跳过替换")
                        continue
                    
                    # 合并prompt和external trace
                    new_input_ids = torch.cat([original_prompt_ids, external_trace], dim=0)
                    new_attention_mask = torch.ones_like(new_input_ids, dtype=torch.bool)
                    
                    # 获取当前序列的最大长度，确保长度匹配
                    current_seq_len = batch.batch['input_ids'].shape[1]
                    
                    if len(new_input_ids) > current_seq_len:
                        # 如果新序列太长，截断external_trace部分
                        available_len = current_seq_len - len(original_prompt_ids)
                        if available_len > 0:
                            external_trace = external_trace[:available_len]
                            new_input_ids = torch.cat([original_prompt_ids, external_trace], dim=0)
                            new_attention_mask = torch.ones_like(new_input_ids, dtype=torch.bool)
                        else:
                            print(f"[双池子] 警告: prompt太长，无法添加外部trace")
                            continue
                    elif len(new_input_ids) < current_seq_len:
                        # 如果新序列太短，用pad token填充
                        pad_length = current_seq_len - len(new_input_ids)
                        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                        pad_ids = torch.full((pad_length,), pad_token_id, dtype=new_input_ids.dtype, device=device)
                        pad_mask = torch.zeros((pad_length,), dtype=torch.bool, device=device)
                        
                        new_input_ids = torch.cat([new_input_ids, pad_ids], dim=0)
                        new_attention_mask = torch.cat([new_attention_mask, pad_mask], dim=0)
                    
                    # 更新batch中的数据
                    batch.batch['input_ids'][idx] = new_input_ids
                    batch.batch['attention_mask'][idx] = new_attention_mask
                    
                    # 更新response部分（input_ids中prompt之后的部分就是response）
                    response_start = len(original_prompt_ids)
                    batch.batch['responses'][idx] = new_input_ids[response_start:]
                    
                    # 设置prefix_mask：external_trace部分标记为True（off-policy）
                    if 'prefix_mask' not in batch.batch:
                        # 如果没有prefix_mask，创建一个
                        response_len = batch.batch['responses'].shape[1]
                        batch.batch['prefix_mask'] = torch.zeros_like(batch.batch['responses'], dtype=torch.bool)
                    
                    # 标记external_trace部分为off-policy
                    trace_len = len(external_trace)
                    if trace_len > 0:
                        # 确保不超出response的长度
                        response_len = batch.batch['responses'].shape[1]
                        mark_len = min(trace_len, response_len)
                        batch.batch['prefix_mask'][idx, :mark_len] = True
                    
                    # 如果有position_ids，也需要更新
                    if 'position_ids' in batch.batch:
                        new_position_ids = torch.arange(len(new_input_ids), dtype=torch.long, device=device)
                        if len(new_position_ids) != batch.batch['position_ids'].shape[1]:
                            # 调整长度匹配
                            target_len = batch.batch['position_ids'].shape[1]
                            if len(new_position_ids) > target_len:
                                new_position_ids = new_position_ids[:target_len]
                            else:
                                pad_len = target_len - len(new_position_ids)
                                pad_pos = torch.full((pad_len,), new_position_ids[-1].item(), dtype=torch.long, device=device)
                                new_position_ids = torch.cat([new_position_ids, pad_pos], dim=0)
                        
                        batch.batch['position_ids'][idx] = new_position_ids
                    
                    # print(f"[双池子] 成功替换rollout {idx}: prompt_len={len(original_prompt_ids)}, trace_len={len(external_trace)}")
                    
                except Exception as e:
                    print(f"[双池子] 替换rollout {idx} 时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        print(f"[双池子] 策略1完成: 为{len(unique_uids)}个prompt中的{half_rollouts}个rollout替换了外部trace")
        return batch
    
    def resample_prompts_with_mixed_policy(self, update_data):
        """策略2：重新采样所有prompt，使用mixed-policy模式"""
        print("[双池子] 执行策略2: 重新采样所有prompt (mixed-policy模式)")
        
        # 检查输入数据是否为空
        if not update_data:
            print("[双池子] 警告: 没有可重新采样的数据")
            return []
        
        # ===== 关键修改：获取worker数量，批量处理 =====
        
        # 获取worker group的世界大小（worker数量）
        world_size = self.actor_rollout_wg.world_size if hasattr(self.actor_rollout_wg, 'world_size') else 8
        print(f"[双池子-重采样] 检测到worker数量: {world_size}")
        
        # 批量处理prompt，确保batch size能被worker数量整除
        batch_size = max(world_size, len(update_data))  # 至少等于worker数量
        # 确保batch_size是world_size的倍数
        if batch_size % world_size != 0:
            batch_size = ((batch_size // world_size) + 1) * world_size
        
        print(f"[双池子-重采样] 批量处理大小: {batch_size}, 总prompt数: {len(update_data)}")
        
        resampled_data = []
        
        # 分批处理prompt
        for batch_start in range(0, len(update_data), batch_size):
            batch_end = min(batch_start + batch_size, len(update_data))
            batch_prompts = update_data[batch_start:batch_end]
            
            # 如果当前批次大小不够，用第一个prompt填充到足够的大小
            if len(batch_prompts) < batch_size:
                padding_needed = batch_size - len(batch_prompts)
                # 重复最后一个prompt来填充
                last_prompt = batch_prompts[-1]
                batch_prompts.extend([last_prompt] * padding_needed)
                print(f"[双池子-重采样] 批次{batch_start//batch_size + 1}: 填充了{padding_needed}个重复prompt以满足worker要求")
            
            print(f"[双池子-重采样] 处理批次 {batch_start//batch_size + 1}: prompt {batch_start+1}-{batch_end}")
            
            try:
                # ===== 第1步：批量重构原始batch_dict =====
                
                batch_input_ids = []
                batch_attention_mask = []
                batch_position_ids = []
                batch_tgt_input_ids = []
                has_position_ids = False
                has_tgt_input_ids = False
                
                # ===== 新增：收集所有non_tensor_batch数据 =====
                batch_non_tensor_data = {}
                
                for i, prompt_data in enumerate(batch_prompts):
                    # 提取原始prompt信息（取第一个rollout的数据）
                    original_prompt_input_ids = prompt_data['original_prompt_input_ids'][0]  # [prompt_len]
                    original_prompt_attention_mask = prompt_data['original_prompt_attention_mask'][0]  # [prompt_len]
                    
                    batch_input_ids.append(original_prompt_input_ids)
                    batch_attention_mask.append(original_prompt_attention_mask)
                    
                    # 如果有position_ids，也要添加
                    if 'original_prompt_position_ids' in prompt_data:
                        original_prompt_position_ids = prompt_data['original_prompt_position_ids'][0]
                        batch_position_ids.append(original_prompt_position_ids)
                        has_position_ids = True
                    
                    # 关键：添加tgt_input_ids以启用mixed-policy模式
                    if 'original_tgt_input_ids' in prompt_data:
                        original_tgt_input_ids = prompt_data['original_tgt_input_ids'][0]  # [tgt_len]
                        batch_tgt_input_ids.append(original_tgt_input_ids)
                        has_tgt_input_ids = True
                    
                    # ===== 关键修复：复制原始的non_tensor_batch数据 =====
                    # 从原始prompt_data中提取第一个样本的non_tensor_batch数据
                    if i == 0:
                        # 获取第一个prompt的meta_info，其中可能包含原始的non_tensor_batch信息
                        if 'meta_info' in prompt_data:
                            sample_meta_info = prompt_data['meta_info']
                            if 'original_non_tensor_batch' in sample_meta_info:
                                # 如果保存了原始数据，使用它
                                original_non_tensor = sample_meta_info['original_non_tensor_batch']
                                for key, value in original_non_tensor.items():
                                    if key != 'uid':  # uid会重新生成
                                        batch_non_tensor_data[key] = [value] * len(batch_prompts)
                    
                    # 如果上面没有找到，尝试从当前数据推断
                    if not batch_non_tensor_data and 'uid' in prompt_data:
                        # 尝试从现有数据结构推断原始的non_tensor_batch结构
                        # 这是一个fallback方案
                        print(f"[双池子-重采样] 警告: 没有找到原始non_tensor_batch数据，使用fallback方案")
                
                # 将列表转换为tensor并stack
                reconstructed_batch_dict = {
                    'input_ids': torch.stack(batch_input_ids, dim=0),  # [batch_size, prompt_len]
                    'attention_mask': torch.stack(batch_attention_mask, dim=0),  # [batch_size, prompt_len]
                }
                
                if has_position_ids:
                    reconstructed_batch_dict['position_ids'] = torch.stack(batch_position_ids, dim=0)
                
                if has_tgt_input_ids:
                    reconstructed_batch_dict['tgt_input_ids'] = torch.stack(batch_tgt_input_ids, dim=0)
                    print(f"[双池子-重采样] 批次tgt_input_ids shape: {reconstructed_batch_dict['tgt_input_ids'].shape}")
                else:
                    print("[双池子-重采样] 警告: 批次中没有找到tgt_input_ids，可能无法进行mixed-policy采样")
                
                # ===== 第2步：创建DataProto并准备生成 =====
                
                batch = DataProto.from_single_dict(reconstructed_batch_dict)
                
                # ===== 关键修复：设置non_tensor_batch数据 =====
                if batch_non_tensor_data:
                    for key, value in batch_non_tensor_data.items():
                        batch.non_tensor_batch[key] = np.array(value, dtype=object)
                    print(f"[双池子-重采样] 成功设置non_tensor_batch keys: {list(batch_non_tensor_data.keys())}")
                else:
                    print("[双池子-重采样] 警告: 没有找到non_tensor_batch数据，可能会导致reward_fn失败")
                
                # 保存原始信息（用于后续处理）
                original_prompt_input_ids_full = batch.batch['input_ids'].clone()
                original_prompt_attention_mask_full = batch.batch['attention_mask'].clone()
                original_prompt_position_ids_full = None
                if 'position_ids' in batch.batch:
                    original_prompt_position_ids_full = batch.batch['position_ids'].clone()
                original_tgt_input_ids_full = None
                if 'tgt_input_ids' in batch.batch:
                    original_tgt_input_ids_full = batch.batch['tgt_input_ids'].clone()
                
                # 准备生成所需的batch
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids', 'tgt_input_ids'])
                gen_batch.meta_info['global_steps'] = self.global_steps
                
                # ===== 第3步：重要！确保使用原始的prefix_strategy（不是zeroPrefix） =====
                
                # 不修改prefix_strategy，让它使用原来的mixed-policy设置
                # 移除任何可能的强制零前缀标志
                if 'force_zero_prefix' in gen_batch.meta_info:
                    del gen_batch.meta_info['force_zero_prefix']
                
                print(f"[双池子-重采样] 使用原始prefix_strategy: {self.config.actor_rollout_ref.rollout.prefix_strategy}")
                
                # ===== 第4步：执行生成 =====
                
                print(f"[双池子-重采样] 开始重新生成rollout，batch_size: {len(batch_prompts)}")
                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                
                # ===== 第5步：处理生成结果，构造完整的batch =====
                
                # 为生成的数据分配uid
                batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch_prompts))], dtype=object)
                
                # 重复batch以匹配rollout数量
                n_rollouts = self.config.actor_rollout_ref.rollout.n
                batch = batch.repeat(repeat_times=n_rollouts, interleave=True)
                
                # 合并生成的输出
                batch = batch.union(gen_batch_output)
                
                # 添加原始prompt信息到最终batch中
                batch.batch['original_prompt_input_ids'] = original_prompt_input_ids_full.repeat_interleave(n_rollouts, dim=0)
                batch.batch['original_prompt_attention_mask'] = original_prompt_attention_mask_full.repeat_interleave(n_rollouts, dim=0)
                
                if original_prompt_position_ids_full is not None:
                    batch.batch['original_prompt_position_ids'] = original_prompt_position_ids_full.repeat_interleave(n_rollouts, dim=0)
                
                if original_tgt_input_ids_full is not None:
                    batch.batch['original_tgt_input_ids'] = original_tgt_input_ids_full.repeat_interleave(n_rollouts, dim=0)
                
                # ===== 第6步：计算values（如果需要） =====
                
                if self.use_critic:
                    print("[双池子-重采样] 计算values...")
                    values = self.critic_wg.compute_values(batch)
                    batch = batch.union(values)
                
                # ===== 第7步：计算reward model score（如果需要） =====
                
                if self.use_rm:
                    print("[双池子-重采样] 计算RM scores...")
                    reward_tensor = self.rm_wg.compute_rm_score(batch)
                    batch = batch.union(reward_tensor)
                
                # ===== 第8步：计算最终的reward =====
                
                print("[双池子-重采样] 计算reward...")
                print(f"[双池子-重采样] batch.non_tensor_batch keys: {list(batch.non_tensor_batch.keys())}")
                reward_tensor = self.reward_fn(batch)  # [bsz, l]
                batch.batch['token_level_scores'] = reward_tensor
                
                # ===== 第9步：提取重新采样后的prompt数据 =====
                
                # 获取这个批次中每个prompt的rollout数据
                uids = batch.non_tensor_batch['uid']
                unique_uids = np.unique(uids)
                
                # 只处理实际的prompt数量（排除填充的重复prompt）
                actual_prompt_count = batch_end - batch_start
                processed_uids = unique_uids[:actual_prompt_count]
                
                print(f"[双池子-重采样] 批次包含{len(unique_uids)}个unique_uid，实际处理{len(processed_uids)}个")
                
                for uid in processed_uids:
                    uid_mask = uids == uid
                    
                    # 提取这个prompt的所有数据
                    resampled_prompt_data = self.extract_prompt_data(batch, uid_mask)
                    resampled_data.append(resampled_prompt_data)
                    
                    # 简单检查生成质量
                    if 'token_level_scores' in resampled_prompt_data:
                        scores = resampled_prompt_data['token_level_scores'].sum(-1)  # 每个rollout的总分
                        print(f"[双池子-重采样] Prompt {uid[:8]} reward分布: min={scores.min().item():.3f}, max={scores.max().item():.3f}, mean={scores.mean().item():.3f}")
                        
                        # 检查prefix_mask情况
                        if 'prefix_mask' in resampled_prompt_data:
                            prefix_mask = resampled_prompt_data['prefix_mask']
                            prefix_ratio = prefix_mask.float().mean().item()
                            print(f"[双池子-重采样] Prefix覆盖率: {prefix_ratio:.2%} (mixed-policy模式)")
                
                print(f"[双池子-重采样] 批次 {batch_start//batch_size + 1} 处理完成，成功重新采样{len(processed_uids)}个prompt")
                        
            except Exception as e:
                print(f"[双池子-重采样] 处理批次 {batch_start//batch_size + 1} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"[双池子-重采样] 完成重新采样，成功处理了{len(resampled_data)}/{len(update_data)}个prompt")
        
        # ===== 第10步：验证重新采样结果 =====
        
        if resampled_data:
            # 统计重新采样后的整体情况
            total_rollouts = sum(len(data['uid']) for data in resampled_data)
            print(f"[双池子-重采样] 总共重新生成了{total_rollouts}个rollout")
            
            # 检查第一个prompt的数据结构
            sample_data = resampled_data[0]
            print(f"[双池子-重采样] 样本数据keys: {list(sample_data.keys())}")
            if 'responses' in sample_data:
                print(f"[双池子-重采样] Responses shape: {sample_data['responses'].shape}")
            if 'prefix_mask' in sample_data:
                print(f"[双池子-重采样] Prefix_mask shape: {sample_data['prefix_mask'].shape}")
        else:
            print("[双池子-重采样] 警告: 没有成功重新采样任何prompt")
        
        return resampled_data
    
    def combine_two_batches(self, batch_A, batch_B):
        """将两个DataProto batch合并成一个"""
        print("[双池子] 开始合并两个batch")
        
        try:
            # 获取所有tensor keys
            tensor_keys = set()
            for key, value in batch_A.batch.items():
                if isinstance(value, torch.Tensor):
                    tensor_keys.add(key)
            for key, value in batch_B.batch.items():
                if isinstance(value, torch.Tensor):
                    tensor_keys.add(key)
            
            # 合并tensor数据
            combined_dict = {}
            for key in tensor_keys:
                tensors = []
                if key in batch_A.batch and isinstance(batch_A.batch[key], torch.Tensor):
                    tensors.append(batch_A.batch[key])
                if key in batch_B.batch and isinstance(batch_B.batch[key], torch.Tensor):
                    tensors.append(batch_B.batch[key])
                
                if tensors:
                    # 确保所有tensor在同一设备上
                    target_device = tensors[0].device
                    aligned_tensors = [ensure_tensor(t, target_device) for t in tensors]
                    combined_dict[key] = torch.cat(aligned_tensors, dim=0)
            
            # 合并non_tensor数据
            combined_uid = []
            if hasattr(batch_A, 'non_tensor_batch') and 'uid' in batch_A.non_tensor_batch:
                combined_uid.extend(batch_A.non_tensor_batch['uid'])
            if hasattr(batch_B, 'non_tensor_batch') and 'uid' in batch_B.non_tensor_batch:
                combined_uid.extend(batch_B.non_tensor_batch['uid'])
            
            combined_dict['uid'] = np.array(combined_uid, dtype=object)
            
            # 创建合并后的DataProto
            combined_batch = DataProto.from_single_dict(combined_dict)
            
            # 设置meta_info（使用batch_A的meta_info）
            combined_batch.meta_info = batch_A.meta_info.copy()
            
            print(f"[双池子] 成功合并batch - A: {len(batch_A.non_tensor_batch['uid'])}, B: {len(batch_B.non_tensor_batch['uid'])}, 合并后: {len(combined_batch.non_tensor_batch['uid'])}")
            
            return combined_batch
            
        except Exception as e:
            print(f"[双池子] 合并batch时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

    def update_with_both_pools(self):
        """同时使用池子A和池子B的数据进行联合更新"""
        print(f"[双池子] 开始同时更新模式，池子A: {len(self.pool_A)}, 池子B: {len(self.pool_B)}")
        
        # 从两个池子中分别抽取数据
        pool_A_data = self.pool_A[:self.simultaneous_size]
        pool_B_data = self.pool_B[:self.simultaneous_size]
        
        print(f"[双池子] 抽取数据 - 池子A: {len(pool_A_data)}, 池子B: {len(pool_B_data)}")
        
        # 对池子A的数据执行策略处理
        if self.pool_A_replacement_strategy == 'replace_half':
            print("[双池子] 对池子A执行策略1: 替换一半rollout为外部rollout")
            batch_A = self.combine_pool_data(pool_A_data)
            if batch_A is not None:
                batch_A = self.replace_half_rollouts_with_external(batch_A)
        elif self.pool_A_replacement_strategy == 'resample':
            print("[双池子] 对池子A执行策略2: 重新采样所有prompt")
            resampled_data = self.resample_prompts_with_mixed_policy(pool_A_data)
            batch_A = self.combine_pool_data(resampled_data)
        else:
            # 默认策略：直接使用现有数据
            batch_A = self.combine_pool_data(pool_A_data)
        
        # 对池子B的数据进行处理（强制设为on-policy）
        batch_B = self.combine_pool_data(pool_B_data)
        if batch_B is not None:
            # 强制设为on-policy：将所有prefix_mask设为False
            if 'prefix_mask' in batch_B.batch:
                batch_B.batch['prefix_mask'] = torch.zeros_like(batch_B.batch['prefix_mask'], dtype=torch.bool)
            else:
                responses = batch_B.batch['responses']
                batch_B.batch['prefix_mask'] = torch.zeros_like(responses, dtype=torch.bool)
        
        # 检查两个batch是否都存在
        if batch_A is None and batch_B is None:
            print("[双池子] 警告: 两个batch都为空，跳过更新")
            return {}
        elif batch_A is None:
            print("[双池子] 警告: 池子A的batch为空，只使用池子B")
            combined_batch = batch_B
        elif batch_B is None:
            print("[双池子] 警告: 池子B的batch为空，只使用池子A")
            combined_batch = batch_A
        else:
            # 将两个batch合并
            combined_batch = self.combine_two_batches(batch_A, batch_B)
        
        if combined_batch is None:
            print("[双池子] 错误: 合并后的batch为空")
            return {}
        
        print(f"[双池子] 合并后batch大小: {len(combined_batch.non_tensor_batch['uid'])}")
        
        # 执行标准的PPO更新流程
        metrics = self.execute_ppo_update(combined_batch, pool_type="A+B")
        
        # 从两个池子中移除已使用的数据
        self.pool_A = self.pool_A[self.simultaneous_size:]
        self.pool_B = self.pool_B[self.simultaneous_size:]
        
        print(f"[双池子] 同时更新完成，剩余数据 - 池子A: {len(self.pool_A)}, 池子B: {len(self.pool_B)}")
        
        return metrics

    def update_with_pool_A(self):
        """使用池子A的数据进行mixed-policy更新"""
        print(f"[双池子] 开始使用池子A进行mixed-policy更新，当前数据量: {len(self.pool_A)}")
        
        # 取出更新所需的数据量
        update_data = self.pool_A[:self.pool_A_update_size]
        
        # 选择替换策略
        if self.pool_A_replacement_strategy == 'replace_half':
            print("[双池子] 执行策略1: 替换一半rollout为外部rollout")
            batch = self.combine_pool_data(update_data)
            if batch is not None:
                batch = self.replace_half_rollouts_with_external(batch)
        elif self.pool_A_replacement_strategy == 'resample':
            print("[双池子] 执行策略2: 重新采样所有prompt")
            resampled_data = self.resample_prompts_with_mixed_policy(update_data)
            batch = self.combine_pool_data(resampled_data)
        else:
            # 默认策略：直接使用现有数据
            batch = self.combine_pool_data(update_data)
        
        if batch is None:
            return {}
        
        # 保持原有的mixed-policy逻辑，不修改prefix_mask
        # 原代码已经在rollout阶段正确设置了prefix_mask
        
        # 执行标准的PPO更新流程
        metrics = self.execute_ppo_update(batch, pool_type="A")
        
        # 从池子A中移除已使用的数据
        self.pool_A = self.pool_A[self.pool_A_update_size:]
        print(f"[双池子] 池子A更新完成，剩余数据: {len(self.pool_A)}")
        
        return metrics

    def update_with_pool_B(self):
        """使用池子B的数据进行on-policy更新"""
        print(f"[双池子] 开始使用池子B进行on-policy更新，当前数据量: {len(self.pool_B)}")
        
        # 取出更新所需的数据量
        update_data = self.pool_B[:self.pool_B_update_size]
        
        # 合并池子B的数据
        batch = self.combine_pool_data(update_data)
        if batch is None:
            return {}
        
        # 强制设为on-policy：将所有prefix_mask设为False
        if 'prefix_mask' in batch.batch:
            batch.batch['prefix_mask'] = torch.zeros_like(batch.batch['prefix_mask'], dtype=torch.bool)
        else:
            responses = batch.batch['responses']
            batch.batch['prefix_mask'] = torch.zeros_like(responses, dtype=torch.bool)
        
        # 执行标准的PPO更新流程
        metrics = self.execute_ppo_update(batch, pool_type="B")
        
        # 从池子B中移除已使用的数据
        self.pool_B = self.pool_B[self.pool_B_update_size:]
        print(f"[双池子] 池子B更新完成，剩余数据: {len(self.pool_B)}")
        
        return metrics

    def execute_ppo_update(self, batch, pool_type="unknown"):
        """执行标准的PPO更新流程"""
        metrics = {}
        timing_raw = {}
        
        with _timer('ppo_update', timing_raw):
            # recompute old_log_probs
            with _timer('old_log_prob', timing_raw):
                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                batch = batch.union(old_log_prob)

            if self.use_reference_policy:
                # compute reference log_prob
                with _timer('ref', timing_raw):
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                    batch = batch.union(ref_log_prob)

            # compute rewards with KL penalty if needed
            if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                batch, kl_metrics = apply_kl_penalty(batch,
                                                     kl_ctrl=self.kl_ctrl,
                                                     kl_penalty=self.config.algorithm.kl_penalty)
                metrics.update(kl_metrics)
            else:
                batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

            # compute advantages
            batch = compute_advantage(batch,
                                      adv_estimator=self.config.algorithm.adv_estimator,
                                      gamma=self.config.algorithm.gamma,
                                      lam=self.config.algorithm.lam,
                                      grpo_use_std=self.config.algorithm.grpo_use_std)

            # compute alpha and beta for prefix reward weighting
            if 'prefix_mask' in batch.batch:
                prefix_mask = batch.batch['prefix_mask']
                advantages = batch.batch['advantages']
                assert prefix_mask.shape == advantages.shape
                
                alpha_weight = prefix_mask.float() * self.config.actor_rollout_ref.rollout.prefix_reward_weight_alpha
                beta_weight = (~prefix_mask).float() * self.config.actor_rollout_ref.rollout.prefix_reward_weight_beta
                prefix_weight = alpha_weight + beta_weight
                batch.batch['advantages'] = prefix_weight * advantages

            if self.config.data.get('disable_truncation_advantage', False):
                responses = batch.batch['responses']
                responses_mask = responses != self.tokenizer.pad_token_id
                response_length = responses_mask.sum(-1) # [bsz]
                max_len = self.config.data.max_response_length
                has_truncated = response_length >= max_len
                no_eos = ~((responses == self.tokenizer.eos_token_id).any(-1))
                truncated_mask = has_truncated & no_eos
                batch.batch['advantages'][truncated_mask] = 0

            n_samples = self.config.actor_rollout_ref.rollout.n
            if self.config.actor_rollout_ref.actor.get('use_sft_prefix_reward', False):
                assert self.config.actor_rollout_ref.rollout.n_prefix == -1
                reward_weight = self.config.actor_rollout_ref.actor.get('sft_prefix_reward_weight', 1.0)
                if 'prefix_mask' in batch.batch:
                    prefix_mask = batch.batch['prefix_mask']
                    batch.batch['advantages'][prefix_mask] = reward_weight / n_samples

            # balance the batch
            self._balance_batch(batch, metrics=metrics)

            # compute global_valid tokens
            batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

            # update critic
            if self.use_critic:
                with _timer('update_critic', timing_raw):
                    critic_output = self.critic_wg.update_critic(batch)
                critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                metrics.update(critic_output_metrics)

            # implement critic warmup
            if self.config.trainer.critic_warmup <= self.global_steps:
                # update actor
                with _timer('update_actor', timing_raw):
                    actor_output = self.actor_rollout_wg.update_actor(batch)
                actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                metrics.update(actor_output_metrics)

        # collect metrics
        metrics.update(compute_data_metrics_ours(batch=batch, use_critic=self.use_critic))
        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
        metrics[f'pool/type'] = pool_type
        metrics[f'pool/size'] = len(batch.batch['uid'] if 'uid' in batch.batch else batch.non_tensor_batch['uid'])
        
        return metrics

    def fit(self):
        """
        双池子的训练循环
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1

        print("[双池子策略] 开始训练，采用数据收集和批次更新的策略")

        for epoch in range(self.config.trainer.total_epochs):
            print(f"[双池子] 开始第 {epoch + 1} 个epoch")
            
            # 创建数据迭代器，支持重复使用
            dataloader_iter = iter(self.train_dataloader)
            
            while True:
                try:
                    # 尝试获取下一个batch
                    batch_dict = next(dataloader_iter)
                except StopIteration:
                    # 当前epoch的数据用完了
                    print(f"[双池子] 第 {epoch + 1} 个epoch数据用完")
                    break
                
                try:
                    # 处理batch，生成rollout和计算reward
                    batch, reward_tensor = self.process_batch_for_pools(batch_dict)
                    
                    # 根据reward结果分类到不同池子
                    self.categorize_prompts_to_pools(batch, reward_tensor)

                    # 检查更新条件
                    if self.use_simultaneous_update:
                        # 新模式：同时更新
                        if len(self.pool_A) >= self.simultaneous_size and len(self.pool_B) >= self.simultaneous_size:
                            metrics = self.update_with_both_pools()
                            logger.log(data=metrics, step=self.global_steps)
                            self.global_steps += 1
                            
                            # 检查是否需要验证
                            if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                                self.global_steps % self.config.trainer.test_freq == 0:
                                val_metrics: dict = self._validate()
                                if 'avg_score' not in val_metrics:
                                    val_metrics['avg_score'] = np.mean([val_metrics[key] for key in val_metrics if key.startswith('val/test_score/')])
                                logger.log(data=val_metrics, step=self.global_steps)
                                self.maybe_save_best_hf(val_metrics)

                            # 检查是否需要保存checkpoint
                            if self.config.trainer.save_freq > 0 and \
                                    self.global_steps % self.config.trainer.save_freq == 0:
                                self._save_checkpoint()
                            
                            # 检查是否达到最大训练步数
                            if self.global_steps >= self.total_training_steps:
                                print("[双池子] 达到最大训练步数，结束训练")
                                return
                    else:
                        # 原有模式：分别更新
                        if len(self.pool_A) >= self.pool_A_update_size:
                            metrics = self.update_with_pool_A()
                            logger.log(data=metrics, step=self.global_steps)
                            self.global_steps += 1
                            
                            # 检查是否需要验证
                            if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                                self.global_steps % self.config.trainer.test_freq == 0:
                                val_metrics: dict = self._validate()
                                if 'avg_score' not in val_metrics:
                                    val_metrics['avg_score'] = np.mean([val_metrics[key] for key in val_metrics if key.startswith('val/test_score/')])
                                logger.log(data=val_metrics, step=self.global_steps)
                                self.maybe_save_best_hf(val_metrics)

                            # 检查是否需要保存checkpoint
                            if self.config.trainer.save_freq > 0 and \
                                    self.global_steps % self.config.trainer.save_freq == 0:
                                self._save_checkpoint()
                            
                            # 检查是否达到最大训练步数
                            if self.global_steps >= self.total_training_steps:
                                print("[双池子] 达到最大训练步数，结束训练")
                                return
                        
                        elif len(self.pool_B) >= self.pool_B_update_size:
                            metrics = self.update_with_pool_B()
                            logger.log(data=metrics, step=self.global_steps)
                            self.global_steps += 1
                            
                            # 检查是否需要验证
                            if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                                self.global_steps % self.config.trainer.test_freq == 0:
                                val_metrics: dict = self._validate()
                                if 'avg_score' not in val_metrics:
                                    val_metrics['avg_score'] = np.mean([val_metrics[key] for key in val_metrics if key.startswith('val/test_score/')])
                                logger.log(data=val_metrics, step=self.global_steps)
                                self.maybe_save_best_hf(val_metrics)

                            # 检查是否需要保存checkpoint
                            if self.config.trainer.save_freq > 0 and \
                                    self.global_steps % self.config.trainer.save_freq == 0:
                                self._save_checkpoint()
                            
                            # 检查是否达到最大训练步数
                            if self.global_steps >= self.total_training_steps:
                                print("[双池子] 达到最大训练步数，结束训练")
                                return

                    # 以下为原来处理的代码，现在用同时更新模式替代
                    # 检查是否有池子达到更新阈值，优先处理池子A
                    # if len(self.pool_A) >= self.pool_A_update_size:
                    #     metrics = self.update_with_pool_A()
                    #     logger.log(data=metrics, step=self.global_steps)
                    #     self.global_steps += 1
                        
                    #     # 检查是否需要验证
                    #     if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                    #         self.global_steps % self.config.trainer.test_freq == 0:
                    #         val_metrics: dict = self._validate()
                    #         if 'avg_score' not in val_metrics:
                    #             val_metrics['avg_score'] = np.mean([val_metrics[key] for key in val_metrics if key.startswith('val/test_score/')])
                    #         logger.log(data=val_metrics, step=self.global_steps)
                    #         self.maybe_save_best_hf(val_metrics)

                    #     # 检查是否需要保存checkpoint
                    #     if self.config.trainer.save_freq > 0 and \
                    #             self.global_steps % self.config.trainer.save_freq == 0:
                    #         self._save_checkpoint()
                        
                    #     # 检查是否达到最大训练步数
                    #     if self.global_steps >= self.total_training_steps:
                    #         print("[双池子] 达到最大训练步数，结束训练")
                    #         return
                    
                    # elif len(self.pool_B) >= self.pool_B_update_size:
                    #     metrics = self.update_with_pool_B()
                    #     logger.log(data=metrics, step=self.global_steps)
                    #     self.global_steps += 1
                        
                    #     # 检查是否需要验证
                    #     if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                    #         self.global_steps % self.config.trainer.test_freq == 0:
                    #         val_metrics: dict = self._validate()
                    #         if 'avg_score' not in val_metrics:
                    #             val_metrics['avg_score'] = np.mean([val_metrics[key] for key in val_metrics if key.startswith('val/test_score/')])
                    #         logger.log(data=val_metrics, step=self.global_steps)
                    #         self.maybe_save_best_hf(val_metrics)

                    #     # 检查是否需要保存checkpoint
                    #     if self.config.trainer.save_freq > 0 and \
                    #             self.global_steps % self.config.trainer.save_freq == 0:
                    #         self._save_checkpoint()
                        
                    #     # 检查是否达到最大训练步数
                    #     if self.global_steps >= self.total_training_steps:
                    #         print("[双池子] 达到最大训练步数，结束训练")
                    #         return
                
                except Exception as e:
                    print(f"[双池子] 处理batch时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        # 训练结束后的处理
        # 训练结束后的处理，有修改
        print("[双池子] 所有epoch完成")

        if self.use_simultaneous_update:
            # 同时更新模式：如果两个池子都有剩余数据，进行最后的同时更新
            if len(self.pool_A) > 0 and len(self.pool_B) > 0:
                print(f"[双池子] 处理剩余的双池子数据: A={len(self.pool_A)}, B={len(self.pool_B)}")
                remaining_size = min(len(self.pool_A), len(self.pool_B), self.simultaneous_size)
                if remaining_size > 0:
                    metrics = self.update_with_both_pools()
                    logger.log(data=metrics, step=self.global_steps)
                    self.global_steps += 1
        else:
            # 原有模式：分别处理剩余数据
            if len(self.pool_A) > 0:
                print(f"[双池子] 处理剩余的池子A数据: {len(self.pool_A)}")
                remaining_update_size = min(len(self.pool_A), self.pool_A_update_size)
                if remaining_update_size > 0:
                    batch = self.combine_pool_data(self.pool_A, remaining_update_size)
                    if batch is not None:
                        metrics = self.execute_ppo_update(batch, pool_type="A")
                        logger.log(data=metrics, step=self.global_steps)
                        self.global_steps += 1
            
            if len(self.pool_B) > 0:
                print(f"[双池子] 处理剩余的池子B数据: {len(self.pool_B)}")
                remaining_update_size = min(len(self.pool_B), self.pool_B_update_size)
                if remaining_update_size > 0:
                    batch = self.combine_pool_data(self.pool_B, remaining_update_size)
                    if batch is not None:
                        # 强制设为on-policy
                        if 'prefix_mask' in batch.batch:
                            batch.batch['prefix_mask'] = torch.zeros_like(batch.batch['prefix_mask'], dtype=torch.bool)
                        else:
                            responses = batch.batch['responses']
                            batch.batch['prefix_mask'] = torch.zeros_like(responses, dtype=torch.bool)
                        
                        metrics = self.execute_ppo_update(batch, pool_type="B")
                        logger.log(data=metrics, step=self.global_steps)
                        self.global_steps += 1

        # perform validation after training
        if self.val_reward_fn is not None:
            val_metrics = self._validate()
            pprint(f'Final validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)

    def maybe_save_best_hf(self, val_metrics: dict):
        import json
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'best',
                                        f'actor')
        
        os.makedirs(actor_local_path, exist_ok=True)
        if os.path.exists(f'{actor_local_path}/metrics.json'):
            with open(f'{actor_local_path}/metrics.json', 'r') as f:
                metrics = json.load(f)
            best_score = metrics['best_avg_score']
        else:
            print('Find no current best saved. Best score is set to -inf')
            best_score = -float('inf')
        
        cur_score = val_metrics['avg_score']
        
        if cur_score > best_score:
            print(f'Saving best checkpoint with score {cur_score} at {actor_local_path}')
            best_score = cur_score
            self.actor_rollout_wg.save_checkpoint_hf(actor_local_path)
            with open(f'{actor_local_path}/metrics.json', 'w') as f:
                f.write(json.dumps({'best_avg_score': best_score, 'global_step': self.global_steps})+'\n')
        
def compute_data_metrics_ours(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    from verl.trainer.ppo.ray_trainer import _compute_response_info
    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    # compute on/off policy stats
    off_policy_mask = batch.batch['prefix_mask'].any(-1) # [bsz, ]
    on_policy_mask = ~off_policy_mask
    off_response_length = response_length[off_policy_mask]
    on_response_length = response_length[on_policy_mask]
    
    off_on_example_ratio = off_policy_mask.sum().item() / (on_policy_mask.sum().item() + 1e-6)

    off_sequence_score = sequence_score[off_policy_mask]
    on_sequence_score = sequence_score[on_policy_mask]

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # on/off policy response length
        'on_off_metrics/on_response_length_mean':
            torch.mean(on_response_length).detach().item() if len(on_response_length) > 0 else 0.0,
        'on_off_metrics/off_response_length_mean':
            torch.mean(off_response_length).detach().item() if len(off_response_length) > 0 else 0.0,
        'on_off_metrics/on_score':
            torch.mean(on_sequence_score).detach().item() if len(on_sequence_score) > 0 else 0.0,
        'on_off_metrics/off_score':
            torch.mean(off_sequence_score).detach().item() if len(off_sequence_score) > 0 else 0.0,
        'on_off_metrics/off_on_example_ratio':
            off_on_example_ratio,
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics