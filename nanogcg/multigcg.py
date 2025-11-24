import copy
import gc
import logging
import queue
import threading
import re

from dataclasses import dataclass
from tqdm import tqdm
from typing import List, Optional, Tuple, Union

import torch
import transformers
from torch import Tensor
from transformers import set_seed
from scipy.stats import spearmanr

from nanogcg.utils import (
    INIT_CHARS,
    configure_pad_token,
    find_executable_batch_size,
    get_nonascii_toks,
    mellowmax,
)

logger = logging.getLogger("nanogcg")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class ProbeSamplingConfig:
    draft_model: transformers.PreTrainedModel
    draft_tokenizer: transformers.PreTrainedTokenizer
    r: int = 8
    sampling_factor: int = 16


@dataclass
class GCGConfig:
    num_steps: int = 250
    optim_str_init: Union[str, List[str]] = "x x x x x x x x x x x x x x x x x x x x"
    search_width: int = 512
    batch_size: int = None
    topk: int = 256
    n_replace: int = 1
    buffer_size: int = 3
    use_mellowmax: bool = False
    mellowmax_alpha: float = 1.0
    early_stop: bool = False
    early_stop_confidence: float = None  # Confidence threshold for early stop (0.0-1.0)
    use_prefix_cache: bool = True
    allow_non_ascii: bool = False
    forbidden_ids: Tensor = None
    filter_ids: bool = True
    add_space_before_target: bool = False
    seed: int = None
    verbosity: str = "INFO"
    probe_sampling_config: Optional[ProbeSamplingConfig] = None


@dataclass
class GCGResult:
    best_loss: float
    best_strings: List[str]
    losses: List[float]
    strings: List[List[str]]


class AttackBuffer:
    def __init__(self, size: int):
        self.buffer = []  # elements are (loss: float, optim_ids: List[Tensor])
        self.size = size

    def add(self, loss: float, optim_ids: List[Tensor]) -> None:
        if self.size == 0:
            self.buffer = [(loss, optim_ids)]
            return

        if len(self.buffer) < self.size:
            self.buffer.append((loss, optim_ids))
        else:
            self.buffer[-1] = (loss, optim_ids)

        self.buffer.sort(key=lambda x: x[0])

    def get_best_ids(self) -> List[Tensor]:
        return self.buffer[0][1]

    def get_lowest_loss(self) -> float:
        return self.buffer[0][0]

    def get_highest_loss(self) -> float:
        return self.buffer[-1][0]

    def log_buffer(self, tokenizer):
        message = "buffer:"
        for loss, ids_list in self.buffer:
            optim_strs = [tokenizer.batch_decode(ids)[0] for ids in ids_list]
            optim_strs = [s.replace("\\", "\\\\").replace("\n", "\\n") for s in optim_strs]
            message += f"\nloss: {loss}" + f" | strings: {optim_strs}"
        logger.info(message)


def sample_ids_from_grad_multi(
    ids_list: List[Tensor],
    grad_list: List[Tensor],
    search_width: int,
    topk: int = 256,
    n_replace: int = 1,
    not_allowed_ids: Tensor = None,
) -> List[Tensor]:
    """Returns `search_width` combinations of token ids based on the token gradient for multiple strings.

    Args:
        ids_list : List[Tensor], shape = (n_optim_ids_1), (n_optim_ids_2), ...
            the sequences of token ids that are being optimized
        grad_list : List[Tensor], shape = (n_optim_ids_1, vocab_size), (n_optim_ids_2, vocab_size), ...
            the gradients for each optimized string
        search_width : int
            the number of candidate sequences to return
        topk : int
            the topk to be used when sampling from the gradient
        n_replace : int
            the number of token positions to update per sequence
        not_allowed_ids : Tensor, shape = (n_ids)
            the token ids that should not be used in optimization

    Returns:
        sampled_ids_list : List[Tensor], shape = (search_width, n_optim_ids_1), (search_width, n_optim_ids_2), ...
            sampled token ids for each optimized string
    """
    sampled_ids_list = []
    
    # Iterate through each optimized string and its gradient
    for i in range(len(ids_list)):
        ids = ids_list[i].squeeze(0)
        grad = grad_list[i].squeeze(0)
        n_optim_tokens = len(ids)
        original_ids = ids.repeat(search_width, 1)

        if not_allowed_ids is not None:
            grad[:, not_allowed_ids.to(grad.device)] = float("inf")

        topk_ids = (-grad).topk(topk, dim=1).indices

        sampled_ids_pos = torch.argsort(torch.rand((search_width, n_optim_tokens), device=grad.device))[..., :n_replace]
        sampled_ids_val = torch.gather(
            topk_ids[sampled_ids_pos],
            2,
            torch.randint(0, topk, (search_width, n_replace, 1), device=grad.device),
        ).squeeze(2)

        new_ids = original_ids.scatter_(1, sampled_ids_pos, sampled_ids_val)
        sampled_ids_list.append(new_ids)
        
    return sampled_ids_list


def filter_ids_multi(ids_list: List[Tensor], tokenizer: transformers.PreTrainedTokenizer):
    """Filters out sequences of token ids that change after retokenization for multiple strings.

    Args:
        ids_list : List[Tensor], shape = (search_width, n_optim_ids_1), (search_width, n_optim_ids_2), ...
            token ids for all optimized strings
        tokenizer : ~transformers.PreTrainedTokenizer
            the model's tokenizer

    Returns:
        filtered_ids_list : List[Tensor]
            filtered token ids for each optimized string
    """
    search_width = ids_list[0].shape[0]
    filtered_indices = []

    for i in range(search_width):
        is_same = True
        for ids in ids_list:
            ids_i = ids[i]
            ids_decoded = tokenizer.batch_decode(ids_i.unsqueeze(0))[0]
            ids_encoded = tokenizer(ids_decoded, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(ids.device)
            if not torch.equal(ids_i, ids_encoded):
                is_same = False
                break
        if is_same:
            filtered_indices.append(i)

    if not filtered_indices:
        raise RuntimeError(
            "No token sequences are the same after decoding and re-encoding. "
            "Consider setting `filter_ids=False` or trying a different `optim_str_init`"
        )
    
    filtered_ids_list = [ids[filtered_indices] for ids in ids_list]
    return filtered_ids_list


class GCG:
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        config: GCGConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        self.embedding_layer = model.get_input_embeddings()
        self.not_allowed_ids = None if config.allow_non_ascii else get_nonascii_toks(tokenizer, device=model.device)

        # Convert forbidden tokens to token ids
        if config.forbidden_ids is not None:
            if self.not_allowed_ids is None:
                self.not_allowed_ids = config.forbidden_ids
            else:
                self.not_allowed_ids = torch.cat([self.not_allowed_ids, config.forbidden_ids], dim=0).unique()

        self.prefix_cache = None
        self.draft_prefix_cache = None

        self.stop_flag = False

        self.draft_model = None
        self.draft_tokenizer = None
        self.draft_embedding_layer = None
        if self.config.probe_sampling_config:
            self.draft_model = self.config.probe_sampling_config.draft_model
            self.draft_tokenizer = self.config.probe_sampling_config.draft_tokenizer
            self.draft_embedding_layer = self.draft_model.get_input_embeddings()
            if self.draft_tokenizer.pad_token is None:
                configure_pad_token(self.draft_tokenizer)

        if model.dtype in (torch.float32, torch.float64):
            logger.warning(f"Model is in {model.dtype}. Use a lower precision data type, if possible, for much faster optimization.")

        if model.device == torch.device("cpu"):
            logger.warning("Model is on the CPU. Use a hardware accelerator for faster optimization.")

        if not tokenizer.chat_template:
            logger.warning("Tokenizer does not have a chat template. Assuming base model and setting chat template to empty.")
            tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"

    def run(
        self,
        messages: Union[str, List[dict]],
        target: str,
        optim_str_placeholders: Optional[List[str]] = None,
    ) -> GCGResult:
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        if config.seed is not None:
            set_seed(config.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        else:
            messages = copy.deepcopy(messages)
            
        if optim_str_placeholders is None:
            optim_str_placeholders = ["{optim_str}"]
            # Append the GCG string at the end of the prompt if location not specified
            if not any([p in d["content"] for p in optim_str_placeholders for d in messages]):
                messages[-1]["content"] = messages[-1]["content"] + "{optim_str}"

        # Combine all placeholders into a single regex for splitting
        all_placeholders = "|".join(re.escape(p) for p in optim_str_placeholders)
        template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
            template = template.replace(tokenizer.bos_token, "")

        # Split the template into fixed and optimized parts
        fixed_parts = re.split(all_placeholders, template)

        target = " " + target if config.add_space_before_target else target

        self.fixed_ids = [
            tokenizer(part, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
            for part in fixed_parts
        ]
        self.target_ids = tokenizer([target], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)

        self.fixed_embeds = [self.embedding_layer(ids) for ids in self.fixed_ids]
        self.target_embeds = self.embedding_layer(self.target_ids)
        self.num_optim_strings = len(optim_str_placeholders)

        if config.use_prefix_cache:
            with torch.no_grad():
                output = model(inputs_embeds=self.fixed_embeds[0], use_cache=True)
            self.prefix_cache = output.past_key_values

        if config.probe_sampling_config:
            assert self.draft_model and self.draft_tokenizer and self.draft_embedding_layer
            
            self.draft_fixed_ids = [
                self.draft_tokenizer(part, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
                for part in fixed_parts
            ]
            self.draft_target_ids = self.draft_tokenizer([target], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)

            self.draft_fixed_embeds = [self.draft_embedding_layer(ids) for ids in self.draft_fixed_ids]
            self.draft_target_embeds = self.draft_embedding_layer(self.draft_target_ids)

            if config.use_prefix_cache:
                with torch.no_grad():
                    output = self.draft_model(inputs_embeds=self.draft_fixed_embeds[0], use_cache=True)
                self.draft_prefix_cache = output.past_key_values

        buffer = self.init_buffer()
        optim_ids_list = buffer.get_best_ids()

        losses = []
        optim_strings_list = []

        for _ in tqdm(range(config.num_steps)):
            optim_ids_onehot_grad_list = self.compute_token_gradient(optim_ids_list)

            with torch.no_grad():
                sampled_ids_list = sample_ids_from_grad_multi(
                    optim_ids_list,
                    optim_ids_onehot_grad_list,
                    config.search_width,
                    config.topk,
                    config.n_replace,
                    not_allowed_ids=self.not_allowed_ids,
                )

                if config.filter_ids:
                    sampled_ids_list = filter_ids_multi(sampled_ids_list, tokenizer)
                
                new_search_width = sampled_ids_list[0].shape[0]

                input_embeds = self._create_input_embeds_batch(new_search_width, sampled_ids_list, self.fixed_embeds, self.target_embeds)
                
                batch_size = new_search_width if config.batch_size is None else config.batch_size
                
                if self.config.probe_sampling_config is None:
                    loss = find_executable_batch_size(self._compute_candidates_loss_original, batch_size)(input_embeds)
                    current_loss = loss.min().item()
                    best_candidate_idx = loss.argmin()
                    optim_ids_list = [ids[best_candidate_idx].unsqueeze(0) for ids in sampled_ids_list]
                else:
                    current_loss, optim_ids_list = find_executable_batch_size(self._compute_candidates_loss_probe_sampling, batch_size)(
                        input_embeds, sampled_ids_list
                    )

                losses.append(current_loss)
                if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                    buffer.add(current_loss, optim_ids_list)

            optim_ids_list = buffer.get_best_ids()
            optim_strings = [tokenizer.batch_decode(ids)[0] for ids in optim_ids_list]
            optim_strings_list.append(optim_strings)

            buffer.log_buffer(tokenizer)

            if self.stop_flag:
                logger.info("Early stopping due to finding a perfect match.")
                break

        min_loss_index = losses.index(min(losses))

        result = GCGResult(
            best_loss=losses[min_loss_index],
            best_strings=optim_strings_list[min_loss_index],
            losses=losses,
            strings=optim_strings_list,
        )
        return result

    def init_buffer(self) -> AttackBuffer:
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        logger.info(f"Initializing attack buffer of size {config.buffer_size}...")

        buffer = AttackBuffer(config.buffer_size)

        if isinstance(config.optim_str_init, str):
            init_optim_ids_list = [
                tokenizer(config.optim_str_init, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
                for _ in range(self.num_optim_strings)
            ]
        else: # assume list of strings
            init_optim_ids_list = [
                tokenizer(s, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
                for s in config.optim_str_init
            ]
            if len(init_optim_ids_list) != self.num_optim_strings:
                raise ValueError("The number of initial strings must match the number of placeholders.")

        true_buffer_size = max(1, config.buffer_size)
        
        buffer_ids_list = []
        if true_buffer_size > 1:
            init_buffer_ids = tokenizer(INIT_CHARS, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze().to(model.device)
            for init_optim_ids in init_optim_ids_list:
                init_indices = torch.randint(0, init_buffer_ids.shape[0], (true_buffer_size - 1, init_optim_ids.shape[1]))
                buffer_ids = torch.cat([init_optim_ids, init_buffer_ids[init_indices]], dim=0)
                buffer_ids_list.append(buffer_ids)
        else:
            buffer_ids_list = init_optim_ids_list

        init_buffer_embeds = self._create_input_embeds_batch(true_buffer_size, buffer_ids_list, self.fixed_embeds, self.target_embeds)
        init_buffer_losses = find_executable_batch_size(self._compute_candidates_loss_original, true_buffer_size)(init_buffer_embeds)

        for i in range(true_buffer_size):
            current_ids_list = [ids[[i]] for ids in buffer_ids_list]
            buffer.add(init_buffer_losses[i], current_ids_list)

        buffer.log_buffer(tokenizer)

        logger.info("Initialized attack buffer.")
        return buffer

    def _create_input_embeds_batch(self, batch_size, optim_ids_list, fixed_embeds, target_embeds):
        """Helper to create a single input tensor from fixed and optimized embeddings."""
        all_embeds = []
        for i in range(len(fixed_embeds)):
            all_embeds.append(fixed_embeds[i].repeat(batch_size, 1, 1))
            if i < len(optim_ids_list):
                all_embeds.append(self.embedding_layer(optim_ids_list[i]))
        all_embeds.append(target_embeds.repeat(batch_size, 1, 1))
        
        return torch.cat(all_embeds, dim=1)

    def compute_token_gradient(
        self,
        optim_ids_list: List[Tensor],
    ) -> List[Tensor]:
        model = self.model
        embedding_layer = self.embedding_layer
        
        optim_ids_onehot_list = []
        optim_embeds_list = []
        for optim_ids in optim_ids_list:
            optim_ids_onehot = torch.nn.functional.one_hot(optim_ids, num_classes=embedding_layer.num_embeddings)
            optim_ids_onehot = optim_ids_onehot.to(model.device, model.dtype)
            optim_ids_onehot.requires_grad_()
            optim_ids_onehot_list.append(optim_ids_onehot)
            optim_embeds_list.append(optim_ids_onehot @ embedding_layer.weight)

        all_embeds = []
        for i in range(len(self.fixed_embeds)):
            all_embeds.append(self.fixed_embeds[i])
            if i < len(optim_embeds_list):
                all_embeds.append(optim_embeds_list[i])
        all_embeds.append(self.target_embeds)
        
        input_embeds = torch.cat(all_embeds, dim=1)

        if self.prefix_cache:
            input_embeds_after_prefix = torch.cat(
                [part for part in all_embeds[1:]], dim=1
            )
            output = model(
                inputs_embeds=input_embeds_after_prefix,
                past_key_values=self.prefix_cache,
                use_cache=True,
            )
        else:
            output = model(inputs_embeds=input_embeds)

        logits = output.logits

        shift = input_embeds.shape[1] - self.target_ids.shape[1]
        shift_logits = logits[..., shift - 1 : -1, :].contiguous()
        shift_labels = self.target_ids

        if self.config.use_mellowmax:
            label_logits = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
            loss = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, dim=-1)
        else:
            loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        optim_ids_onehot_grad_list = torch.autograd.grad(outputs=[loss], inputs=optim_ids_onehot_list)
        return optim_ids_onehot_grad_list

    def _compute_candidates_loss_original(
        self,
        search_batch_size: int,
        input_embeds: Tensor,
    ) -> Tensor:
        all_loss = []
        prefix_cache_batch = None

        # print("search_batch_size:", search_batch_size, flush=True)
        # print("input_embeds.shape:", input_embeds.shape, flush=True)
        # search_batch_size: 1
        # input_embeds.shape: torch.Size([1, 70, 1536])

        for i in range(0, input_embeds.shape[0], search_batch_size):
            with torch.no_grad():
                input_embeds_batch = input_embeds[i:i + search_batch_size]
                current_batch_size = input_embeds_batch.shape[0]
                
                if self.prefix_cache:
                    if not prefix_cache_batch or current_batch_size != search_batch_size:
                        prefix_cache_batch = [[x.expand(current_batch_size, -1, -1, -1) for x in self.prefix_cache[i]] for i in range(len(self.prefix_cache))]
                    
                    outputs = self.model(
                        inputs_embeds=input_embeds_batch[:, self.fixed_embeds[0].shape[1]:], 
                        past_key_values=prefix_cache_batch, 
                        use_cache=True
                    )
                else:
                    outputs = self.model(inputs_embeds=input_embeds_batch)

                logits = outputs.logits
                
                # print("logits.shape:", logits.shape, flush=True)
                # print("input_embeds_batch.shape:", input_embeds_batch.shape, flush=True)
                tmp = input_embeds_batch.shape[1] - self.target_ids.shape[1]
                shift_logits = logits[..., tmp-1:-1, :].contiguous()
                shift_labels = self.target_ids.repeat(current_batch_size, 1)

                if self.config.use_mellowmax:
                    label_logits = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
                    loss = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, dim=-1)
                else:
                    # print("shift_logits.shape:", shift_logits.shape, flush=True)
                    # print("shift_labels.shape:", shift_labels.shape, flush=True)
                    # shift_logits.shape: torch.Size([1, 0, 151936])
                    # shift_labels.shape: torch.Size([1, 1])

                    loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none")

                loss = loss.view(current_batch_size, -1).mean(dim=-1)
                all_loss.append(loss)

                if self.config.early_stop:
                    greedy_match = torch.any(torch.all(torch.argmax(shift_logits, dim=-1) == shift_labels, dim=-1)).item()

                    if greedy_match:
                        # 如果贪心匹配成功，检查置信度阈值（如果设置了）
                        if self.config.early_stop_confidence is not None:
                            # 计算目标token的置信度
                            label_logits = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
                            target_probs = torch.softmax(label_logits, dim=-1)

                            # 使用平均概率作为置信度指标
                            avg_confidence = torch.mean(target_probs, dim=-1)
                            max_confidence = torch.max(avg_confidence).item()

                            # 检查是否满足置信度阈值
                            if max_confidence >= self.config.early_stop_confidence:
                                if self.config.verbosity != "WARNING":  # 避免在batch模式下输出过多信息
                                    print(f"Early stopping: greedy match achieved with confidence {max_confidence:.3f} >= {self.config.early_stop_confidence}")
                                self.stop_flag = True
                        else:
                            # 没有设置置信度阈值，直接早停
                            self.stop_flag = True

                del outputs
                gc.collect()
                torch.cuda.empty_cache()

        return torch.cat(all_loss, dim=0)

    def _create_draft_input_embeds_batch(self, batch_size, optim_ids_list, fixed_embeds, target_embeds):
        """Helper for probe sampling to create draft model input embeddings."""
        all_embeds = []
        for i in range(len(fixed_embeds)):
            all_embeds.append(fixed_embeds[i].repeat(batch_size, 1, 1))
            if i < len(optim_ids_list):
                all_embeds.append(self.draft_embedding_layer(optim_ids_list[i]))
        all_embeds.append(target_embeds.repeat(batch_size, 1, 1))
        
        return torch.cat(all_embeds, dim=1)

    def _compute_candidates_loss_probe_sampling(
        self,
        search_batch_size: int,
        input_embeds: Tensor,
        sampled_ids_list: List[Tensor],
    ) -> Tuple[float, List[Tensor]]:
        probe_sampling_config = self.config.probe_sampling_config
        assert probe_sampling_config, "Probe sampling config wasn't set up properly."
        
        B = input_embeds.shape[0]
        probe_size = B // probe_sampling_config.sampling_factor
        probe_idxs = torch.randperm(B)[:probe_size].to(input_embeds.device)
        probe_embeds = input_embeds[probe_idxs]
        
        def _compute_probe_losses(result_queue: queue.Queue, search_batch_size: int, probe_embeds: Tensor) -> None:
            probe_losses = self._compute_candidates_loss_original(search_batch_size, probe_embeds)
            result_queue.put(("probe", probe_losses))

        def _compute_draft_losses(
            result_queue: queue.Queue,
            search_batch_size: int,
            draft_sampled_ids_list: List[Tensor],
        ) -> None:
            assert self.draft_model and self.draft_embedding_layer, "Draft model and embedding layer weren't initialized properly."

            draft_losses = []
            draft_prefix_cache_batch = None
            
            for i in range(0, B, search_batch_size):
                with torch.no_grad():
                    batch_size = min(search_batch_size, B - i)
                    draft_sampled_ids_list_batch = [ids[i : i + batch_size] for ids in draft_sampled_ids_list]

                    draft_embeds = self._create_draft_input_embeds_batch(
                        batch_size, 
                        draft_sampled_ids_list_batch, 
                        self.draft_fixed_embeds, 
                        self.draft_target_embeds
                    )

                    if self.draft_prefix_cache:
                        if not draft_prefix_cache_batch or batch_size != search_batch_size:
                             draft_prefix_cache_batch = [
                                 [x.expand(batch_size, -1, -1, -1) for x in self.draft_prefix_cache[i]] for i in range(len(self.draft_prefix_cache))
                             ]
                        draft_output = self.draft_model(
                            inputs_embeds=draft_embeds[:, self.draft_fixed_embeds[0].shape[1]:],
                            past_key_values=draft_prefix_cache_batch,
                        )
                    else:
                        draft_output = self.draft_model(inputs_embeds=draft_embeds)
                    
                    draft_logits = draft_output.logits
                    tmp = draft_embeds.shape[1] - self.draft_target_ids.shape[1]
                    shift_logits = draft_logits[..., tmp - 1 : -1, :].contiguous()
                    shift_labels = self.draft_target_ids.repeat(batch_size, 1)

                    if self.config.use_mellowmax:
                        label_logits = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
                        loss = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, dim=-1)
                    else:
                        loss = (
                            torch.nn.functional.cross_entropy(
                                shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1),
                                reduction="none",
                            )
                            .view(batch_size, -1)
                            .mean(dim=-1)
                        )
                    draft_losses.append(loss)

            draft_losses = torch.cat(draft_losses)
            result_queue.put(("draft", draft_losses))

        def _convert_to_draft_tokens(token_ids_list: List[Tensor]) -> List[Tensor]:
            draft_token_ids_list = []
            assert self.draft_tokenizer, "Draft tokenizer wasn't properly initialized."

            for token_ids in token_ids_list:
                decoded_text_list = self.tokenizer.batch_decode(token_ids)
                draft_token_ids_list.append(self.draft_tokenizer(
                    decoded_text_list,
                    add_special_tokens=False,
                    padding=True,
                    return_tensors="pt",
                )["input_ids"].to(self.draft_model.device, torch.int64))
            
            return draft_token_ids_list

        result_queue = queue.Queue()
        draft_sampled_ids_list = _convert_to_draft_tokens(sampled_ids_list)

        draft_thread = threading.Thread(
            target=_compute_draft_losses,
            args=(result_queue, search_batch_size, draft_sampled_ids_list),
        )
        probe_thread = threading.Thread(
            target=_compute_probe_losses,
            args=(result_queue, search_batch_size, probe_embeds),
        )

        draft_thread.start()
        probe_thread.start()

        draft_thread.join()
        probe_thread.join()

        results = {}
        while not result_queue.empty():
            key, value = result_queue.get()
            results[key] = value

        probe_losses = results["probe"]
        draft_losses = results["draft"]

        draft_probe_losses = draft_losses[probe_idxs]
        rank_correlation = spearmanr(
            probe_losses.cpu().type(torch.float32).numpy(),
            draft_probe_losses.cpu().type(torch.float32).numpy(),
        ).correlation
        alpha = (1 + rank_correlation) / 2

        R = probe_sampling_config.r
        filtered_size = int((1 - alpha) * B / R)
        filtered_size = max(1, min(filtered_size, B))

        _, top_indices = torch.topk(draft_losses, k=filtered_size, largest=False)

        filtered_ids_list = [ids[top_indices] for ids in sampled_ids_list]
        filtered_embeds = self._create_input_embeds_batch(filtered_size, filtered_ids_list, self.fixed_embeds, self.target_embeds)
        filtered_losses = self._compute_candidates_loss_original(search_batch_size, filtered_embeds)

        best_probe_loss = probe_losses.min().item()
        best_filtered_loss = filtered_losses.min().item()

        if best_probe_loss < best_filtered_loss:
            best_idx = probe_losses.argmin()
            best_ids_list = [ids[probe_idxs][best_idx].unsqueeze(0) for ids in sampled_ids_list]
            return best_probe_loss, best_ids_list
        else:
            best_idx = filtered_losses.argmin()
            best_ids_list = [ids[top_indices][best_idx].unsqueeze(0) for ids in sampled_ids_list]
            return best_filtered_loss, best_ids_list

def run_multigcg(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    messages: Union[str, List[dict]],
    target: str,
    config: Optional[GCGConfig] = None,
    optim_str_placeholders: Optional[List[str]] = None,
) -> GCGResult:
    """Generates an optimized string using GCG with support for multiple placeholders.

    Args:
        model: The model to use for optimization.
        tokenizer: The model's tokenizer.
        messages: The conversation to use for optimization.
        target: The target generation.
        config: The GCG configuration to use.
        optim_str_placeholders: A list of placeholders, e.g., ["{optim_str_1}", "{optim_str_2}"].

    Returns:
        A GCGResult object that contains losses and the optimized strings.
    """
    if config is None:
        config = GCGConfig()

    logger.setLevel(getattr(logging, config.verbosity))

    gcg = GCG(model, tokenizer, config)
    result = gcg.run(messages, target, optim_str_placeholders=optim_str_placeholders)
    return result