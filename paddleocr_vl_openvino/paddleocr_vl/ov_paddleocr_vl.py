import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers import AutoConfig
from typing import List
import logging as log
from pathlib import Path
from transformers.generation import GenerationConfig, GenerationMixin
import numpy as np
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)

from typing import Optional, Tuple, List, Union

import openvino as ov
from openvino import Core, Type
from openvino.passes import Manager, MatcherPass, WrapType, Matcher
from openvino import opset10 as ops
from openvino.preprocess import PrePostProcessor
# import nncf

import time
import warnings
from transformers.utils.chat_template_utils import render_jinja_template

# 设为 True 可打开 batch 推理的详细日志
_BATCH_VERBOSE = False
from .image_processing_paddleocr_vl import PaddleOCRVLImageProcessor
import numpy as np

# 非文本类标签：这些 block 的输出有固定格式（HTML / LaTeX / 结构化），
# 不应施加 repetition_penalty，否则会破坏格式。
NON_TEXT_PENALTY_LABELS = {
    "table", "chart",
    "display_formula", "inline_formula", "formula_number",
    "seal", "image", "header_image", "footer_image"
}

# 文本类 block 使用的 repetition_penalty 值（>1.0 = 抑制重复）
TEXT_REPETITION_PENALTY = 1.0

# 默认聊天模板（PaddleOCR-VL 格式）
_DEFAULT_CHAT_TEMPLATE = """{%- if not add_generation_prompt is defined -%}
    {%- set add_generation_prompt = true -%}
{%- endif -%}
{%- if not cls_token is defined -%}
    {%- set cls_token = "<|begin_of_sentence|>" -%}
{%- endif -%}
{%- if not eos_token is defined -%}
    {%- set eos_token = "</s>" -%}
{%- endif -%}
{{- cls_token -}}
{%- for message in messages -%}
    {%- if message["role"] == "user" -%}
        {{- "User: " -}}
        {%- for content in message["content"] -%}
            {%- if content["type"] == "image" -%}
                {{ "<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>" }}
            {%- endif -%}
        {%- endfor -%}
        {%- for content in message["content"] -%}
            {%- if content["type"] == "text" -%}
                {{ content["text"] }}
            {%- endif -%}
        {%- endfor -%}
        {{ "\\n" -}}
    {%- elif message["role"] == "assistant" -%}
        {{- "Assistant:\\n" -}}
        {%- for content in message["content"] -%}
            {%- if content["type"] == "text" -%}
                {{ content["text"] }}
            {%- endif -%}
        {%- endfor -%}
        {{ eos_token -}}
    {%- elif message["role"] == "system" -%}
        {%- for content in message["content"] -%}
            {%- if content["type"] == "text" -%}
                {{ content["text"] + "\\n" }}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- "Assistant:\\n" -}}
{%- endif -%}
"""

class PaddleOCR_VL_OV:
    def __init__(self, pretrained_model_path=None, model=None, tokenizer=None, ov_model_path='/tmp/paddleocr_vl_ov/', device='CPU', llm_int4_compress=False, llm_int8_compress=False, vision_int8_quant=False):

        if model is None and pretrained_model_path:
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_path,
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_path,
                trust_remote_code=True
            )
        elif model and tokenizer and pretrained_model_path is None:
            self.model = model
            self.tokenizer = tokenizer

        self.int4_compress = llm_int4_compress
        self.int8_compress = llm_int8_compress
        self.int8_quant = vision_int8_quant
        self.vision_model = VisionModel(model=self.model, ov_model_path=ov_model_path, device=device, int8_quant=self.int8_quant, tokenizer=self.tokenizer)
        self.vision_mlp_model = VisionMlpModel(model=self.model, ov_model_path=ov_model_path, device=device)

        self.llm_embed_model = LlmEmbdModel(model=self.model, ov_model_path=ov_model_path, device=device)
        self.llm_stateful_model = LlmStatefulModel(model=self.model, tokenizer= self.tokenizer, ov_model_path=ov_model_path, device=device, int4_compress=self.int4_compress, int8_compress=self.int8_compress)

    def export_vision_to_ov(self):
        self.vision_model.convert_sdpa_ov()
        self.vision_mlp_model.convert_sdpa_ov()
        self.llm_embed_model.convert_sdpa_ov()
        self.llm_stateful_model.convert_sdpa_ov()

class PaddleOCRVLPreprocessor:
    """
    Preprocessor class for PaddleOCR-VL model.
    Handles message preprocessing, image processing, and tokenization.
    """

    def __init__(self, tokenizer):
        """
        Initialize the preprocessor.

        Args:
            tokenizer: Tokenizer instance for text tokenization
        """
        self.tokenizer = tokenizer

    def preprocess(
        self,
        messages: List[dict],
        chat_template: Optional[str] = None,
        add_generation_prompt: bool = True,
        image_processor_config: Optional[dict] = None,
    ) -> dict:
        """
        Preprocess messages and images for the model.

        Args:
            messages: List of conversation messages. Each message should have the format:
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": PIL.Image},
                            {"type": "text", "text": "..."}
                        ]
                    }
                ]
            chat_template: Optional Jinja2 chat template string. If None, will use the default template.
            add_generation_prompt: Whether to add generation prompt to the template.
            image_processor_config: Optional dictionary with image processor configuration.
                Default values:
                {
                    "resample": 3,
                    "rescale_factor": 0.00392156862745098,
                    "image_mean": [0.5, 0.5, 0.5],
                    "image_std": [0.5, 0.5, 0.5],
                    "min_pixels": 147384,
                    "max_pixels": 2822400,
                    "patch_size": 14,
                    "temporal_patch_size": 1,
                    "merge_size": 2
                }

        Returns:
            Dictionary containing:
                - "text_inputs": Tokenized text inputs from tokenizer
                - "images_info": Processed image information dictionary
        """
        # Use default chat template if not provided
        if chat_template is None:
            chat_template = _DEFAULT_CHAT_TEMPLATE

        # Render Jinja template to get text with placeholders
        text, generation_indices = render_jinja_template(
            conversations=[messages],
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
        )

        # Default image processor configuration
        default_image_processor_config = {
            "resample": 3,
            "rescale_factor": 0.00392156862745098,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "min_pixels": 112896,
            "max_pixels": 1003520,
            "patch_size": 14,
            "temporal_patch_size": 1,
            "merge_size": 2
        }

        # Merge user config with defaults
        if image_processor_config:
            default_image_processor_config.update(image_processor_config)

        # Create image processor
        image_processor = PaddleOCRVLImageProcessor(**default_image_processor_config)

        # Extract images from messages
        images = []
        for message in messages:
            if "content" in message:
                for content in message["content"]:
                    if content.get("type") == "image" and "image" in content:
                        images.append(content["image"])

        # Process images
        images_info = image_processor(images=images, return_tensors="pt")

        # Replace image placeholders in text
        if not isinstance(text, list):
            text = [text]

        index = 0
        for i in range(len(text)):
            while "<|IMAGE_PLACEHOLDER|>" in text[i]:
                text[i] = text[i].replace(
                    "<|IMAGE_PLACEHOLDER|>",
                    "<|placeholder|>"
                    * (
                        images_info['image_grid_thw'][index].prod()
                        // 2
                        // 2
                    ),
                    1,
                )
                index += 1
            text[i] = text[i].replace("<|placeholder|>", "<|IMAGE_PLACEHOLDER|>")

        # Tokenize text
        text_inputs = self.tokenizer(text, return_tensors="pt")

        return {
            "text_inputs": text_inputs,
            "images_info": images_info,
        }


class OVPaddleOCRVLForCausalLM(GenerationMixin):
    _is_stateful = True  # 标记为 stateful 模型，用于 transformers 的生成方法

    def __init__(
        self,
        core=None,
        ov_model_path=None,
        device='CPU',
        llm_int4_compress=False,
        llm_int8_compress=False,
        vision_int8_quant=False,
        llm_int8_quant=False,
        llm_infer_list=[],
        vision_infer=[],
    ):

        self.ov_model_path = ov_model_path
        self.core = core
        self.ov_device = device
        self.llm_int4_compress = llm_int4_compress
        self.llm_int8_compress = llm_int8_compress
        self.vision_int8_quant = vision_int8_quant
        self.llm_int8_quant = llm_int8_quant

        ov_config = {
            "DYNAMIC_QUANTIZATION_GROUP_SIZE": "64",  #32
            "PERFORMANCE_HINT": "LATENCY",
            "NUM_STREAMS": "1",
            "CACHE_DIR": "",
        }

        # 根据压缩选项加载相应的模型
        if llm_int4_compress:
            self.llm_model = Path(f"{ov_model_path}/llm_stateful_int4.xml")
        elif llm_int8_compress:
            self.llm_model = Path(f"{ov_model_path}/llm_stateful_int8.xml")
        else:
            self.llm_model = Path(f"{ov_model_path}/llm_stateful.xml")
        if llm_int8_quant:
            self.llm_compiled_model = core.compile_model(self.llm_model, device, config = ov_config)
        else:
            self.llm_compiled_model = core.compile_model(self.llm_model, device)

        self.llm_request = self.llm_compiled_model.create_infer_request()

        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(self.llm_compiled_model.inputs)}
        self.output_names = {idx: key for idx, key in enumerate(self.llm_compiled_model.outputs)}
        self.key_value_input_names = [key for key in list(self.input_names) if key not in ["beam_idx", "inputs_embeds", "attention_mask", "position_ids"]]
        self.key_value_output_names = [key for key in list(self.output_names)[1:]]
        self.stateful = len(self.key_value_input_names) == 0
        # self.compiled_model = core.compile_model(self.model, device, config = {'INFERENCE_PRECISION_HINT': 'f32'})

        self.config = AutoConfig.from_pretrained(ov_model_path, trust_remote_code=True)
        self.generation_config = GenerationConfig.from_model_config(self.config)
        self.device = torch.device("cpu")
        self.next_beam_idx = None
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.past_len = None
        self.main_input_name = "input_ids"
        self._supports_cache_class = False

        self.llm_embd = core.read_model(Path(f"{ov_model_path}/llm_embd.xml"))
        self.llm_embd_compiled_model = core.compile_model(self.llm_embd, device)
        self.llm_embd_request = self.llm_embd_compiled_model.create_infer_request()

        self.tokenizer = AutoTokenizer.from_pretrained(ov_model_path, trust_remote_code=True)

        # Initialize preprocessor
        self.preprocessor = PaddleOCRVLPreprocessor(tokenizer=self.tokenizer)

        self.vision_model_init()

        self.llm_infer_list = llm_infer_list
        self.vision_infer = vision_infer

        self.rope_deltas = None


    def vision_model_init(self):
        if self.vision_int8_quant:
            self.vision_encoder_model = Path(f"{self.ov_model_path}/vision_int8.xml")
        else:
            self.vision_encoder_model = Path(f"{self.ov_model_path}/vision.xml")
        # self.vision_encoder_compiled_model = self.core.compile_model(self.vision_encoder_model, self.ov_device, config = {'INFERENCE_PRECISION_HINT': 'f32'})
        self.vision_encoder_compiled_model = self.core.compile_model(self.vision_encoder_model, self.ov_device)

        self.vision_encoder_request = self.vision_encoder_compiled_model.create_infer_request()

        self.vision_mlp_model = self.core.read_model(Path(f"{self.ov_model_path}/vision_mlp.xml"))
        self.vision_mlp_compiled_model = self.core.compile_model(self.vision_mlp_model, self.ov_device)
        self.vision_mlp_request = self.vision_mlp_compiled_model.create_infer_request()

        # self.vision_pre_process = Preprocess()
        # self.vision_middle_process = Postprocess()

    def vision_encoder_run(self, pixel_values=None, image_grid_thw=None, cu_seqlens=None):
        inputs_dict = {}
        inputs_dict['pixel_values'] = pixel_values
        inputs_dict['image_grid_thw'] = image_grid_thw
        inputs_dict['cu_seqlens'] = cu_seqlens
        self.vision_encoder_request.start_async(inputs_dict, share_inputs=True)
        self.vision_encoder_request.wait()
        return torch.from_numpy(self.vision_encoder_request.get_tensor("vision_output").data)

    def vision_mlp_run(self, image_features=None, image_grid_thw=None):
        inputs_dict = {}
        inputs_dict['image_features'] = image_features
        inputs_dict['image_grid_thw'] = image_grid_thw
        self.vision_mlp_request.start_async(inputs_dict, share_inputs=True)
        self.vision_mlp_request.wait()
        return torch.from_numpy(self.vision_mlp_request.get_tensor("vit_mlp").data)

    def vision_model(self, pixel_values, image_grid_thw):
        encoder_start = time.perf_counter()

        if pixel_values is not None:
            pixel_values = pixel_values.unsqueeze(0)
            siglip_position_ids = list()
            image_grid_hws = list()
            sample_indices = list()
            cu_seqlens = [0]

            pro = 0
            # breakpoint()
            for idx, thw in enumerate(image_grid_thw):
                thw_tuple = tuple(thw.detach().cpu().numpy().tolist())
                numel = np.prod(thw_tuple)
                image_grid_hws.append(thw_tuple)
                image_position_ids = torch.arange(numel) % np.prod(thw_tuple[1:])
                siglip_position_ids.append(image_position_ids)
                sample_indices.append(torch.full((numel,), idx, dtype=torch.int64))
                cu_seqlens.append(cu_seqlens[-1] + numel)

            siglip_position_ids = torch.concat(siglip_position_ids, dim=0).to(
                pixel_values.device
            )
            cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32).to(
                pixel_values.device
            )
            sample_indices = torch.concat(sample_indices, dim=0).to(
                pixel_values.device
            )
            image_grid_hws = torch.tensor(image_grid_hws, dtype=torch.int64)
            # print("image_grid_hws: ", image_grid_hws)
            # print("cu_seqlens: ", cu_seqlens)

            vision_output = self.vision_encoder_run(pixel_values=pixel_values, image_grid_thw=image_grid_thw, cu_seqlens=cu_seqlens)
            encoder_end = time.perf_counter()
            mlp_start = time.perf_counter()
            vit_embeds = self.vision_mlp_run(image_features=vision_output, image_grid_thw=image_grid_thw)
            mlp_end = time.perf_counter()
            encoder_time = (encoder_end - encoder_start) * 1000
            mlp_time = (mlp_end - mlp_start) * 1000
            self.vision_infer.append(encoder_time)
            self.vision_infer.append(mlp_time)

            return vit_embeds

    def can_generate(self):
        """Returns True to validate the check that the model using `GenerationMixin.generate()` can indeed generate."""
        return True

    def _reorder_cache(self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        self.next_beam_idx = np.array(beam_idx)  # save beam_idx to be used as an input in the next iteration
        return past_key_values

    def llm_embd_run(self, input_ids):
        llm_embd_inputs = {}
        llm_embd_inputs['input_ids'] = input_ids

        self.llm_embd_request.start_async(llm_embd_inputs, share_inputs=True)
        self.llm_embd_request.wait()

        return torch.from_numpy(self.llm_embd_request.get_tensor("inputs_embeds").data)

    def __call__(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        return self.forward(
            input_ids,
            inputs_embeds,
            attention_mask,
            past_key_values,
            position_ids,
            **kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """General inference method"""
        inputs_dict = {}
        if past_key_values is not None:
            inputs_embeds = self.llm_embd_run(input_ids)
            inputs_dict['inputs_embeds'] = inputs_embeds
        else:
            self.past_len = 0
            self.llm_request.reset_state()
            inputs_dict['inputs_embeds'] = inputs_embeds

        inputs_dict["attention_mask"] = attention_mask
        inputs_dict["position_ids"] = position_ids

        batch_size = inputs_embeds.shape[0]
        if "beam_idx" in self.input_names:
            inputs_dict["beam_idx"] = self.next_beam_idx if self.next_beam_idx is not None else np.arange(batch_size, dtype=int)

        # print('attention_mask: ', inputs_dict['attention_mask'].shape)
        # print('position_ids: ', inputs_dict['position_ids'])
        # print('inputs_embeds: ', inputs_dict['inputs_embeds'])
        start = time.perf_counter()
        self.llm_request.start_async(inputs_dict, share_inputs=True)
        self.llm_request.wait()
        end = time.perf_counter()

        generation_time = (end - start) * 1000
        self.llm_infer_list.append(generation_time)

        past_key_values = ((),)
        self.past_len += inputs_dict["inputs_embeds"].shape[1]

        # print('logits: ', self.request.get_tensor("logits").data)
        return CausalLMOutputWithPast(
            loss=None,
            logits=torch.from_numpy(self.llm_request.get_tensor("logits").data),
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        position_ids=None,
        **kwargs,
    ):
        if past_key_values is not None:
            cache_length = past_length = self.past_len
            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - self.past_len) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif self.past_len < input_ids.shape[1]:
                input_ids = input_ids[:, self.past_len:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            elif self.config.image_token_index in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
            # older attention values, as their corresponding values are not part of the input.
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]) :]
        else:
            self.llm_infer_list.clear()

        if past_key_values is not None:
            position_ids = kwargs.get("position_ids", None)
            batch_size, seq_length = input_ids.shape
            delta = (
                (self.past_len + self.rope_deltas).to(input_ids.device)
                if self.past_len is not None
                else 0
            )
            # print("delta: ", delta)
            # print("self.rope_deltas: ", self.rope_deltas)
            # print("self.past_len: ", self.past_len)
            # breakpoint()
            position_ids = torch.arange(seq_length, device=input_ids.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if self.past_len is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        # breakpoint()
        return model_inputs

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (
            image_grid_thw is not None or video_grid_thw is not None
        ):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(
                    input_ids == vision_start_token_id
                ).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                    if torch.is_tensor(second_per_grid_t):
                        second_per_grid_t = second_per_grid_t.detach().item()
                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                    time_tensor = (
                        expanded_range
                        * second_per_grid_t
                        * self.config.vision_config.tokens_per_second
                    )

                    time_tensor_long = time_tensor.long()
                    t_index = time_tensor_long.flatten()

                    h_index = (
                        torch.arange(llm_grid_h)
                        .view(1, -1, 1)
                        .expand(llm_grid_t, -1, llm_grid_w)
                        .flatten()
                    )
                    w_index = (
                        torch.arange(llm_grid_w)
                        .view(1, 1, -1)
                        .expand(llm_grid_t, llm_grid_h, -1)
                        .flatten()
                    )
                    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                    )
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                    position_ids.device
                )
                mrope_position_deltas.append(
                    llm_positions.max() + 1 - len(total_input_ids[i])
                )
            mrope_position_deltas = torch.tensor(
                mrope_position_deltas, device=input_ids.device
            ).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = (
                    position_ids.unsqueeze(0)
                    .expand(3, -1, -1)
                    .to(attention_mask.device)
                )
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                    -1, keepdim=True
                )[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def chat(self, messages=None, chat_template=None, generation_config=None, image_processor_config=None, verbose=False, stopping_criteria=None):
        # Handle default generation_config
        if generation_config is None:
            generation_config = {
                "bos_token_id": self.tokenizer.bos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "max_new_tokens": 1024,
                "do_sample": False,
            }

        prepared = self.prepare_inputs(messages, chat_template=chat_template, image_processor_config=image_processor_config)
        response, _token_stats = self.generate_from_prepared(prepared, generation_config, stopping_criteria=stopping_criteria)
        return response, None

    def prepare_inputs(self, messages, chat_template=None, image_processor_config=None):
        """
        预处理阶段：图像处理 + vision 编码 + text embedding + position 计算。
        返回 generate 所需的全部 tensor，可提前批量执行。
        """
        inputs_dict = self.preprocessor.preprocess(messages=messages, chat_template=chat_template, image_processor_config=image_processor_config)
        input_ids = inputs_dict["text_inputs"]["input_ids"]
        attention_mask = inputs_dict["text_inputs"]["attention_mask"]
        pixel_values = inputs_dict["images_info"]["pixel_values"]
        image_grid_thw = inputs_dict["images_info"]["image_grid_thw"]

        inputs_embeds = self.llm_embd_run(input_ids)
        image_embeds = self.vision_model(pixel_values, image_grid_thw)

        n_image_tokens = (input_ids == 100295).sum().item()
        if isinstance(image_embeds, (list, tuple)):
            image_embeds = torch.cat(image_embeds, dim=0)
        elif isinstance(image_embeds, torch.Tensor):
            image_embeds = image_embeds.view(-1, image_embeds.shape[-1])
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        mask = input_ids == 100295
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        image_mask = mask_expanded.to(inputs_embeds.device)

        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        position_ids, rope_deltas = self.get_rope_index(
            input_ids, image_grid_thw, None, None, attention_mask,
        )

        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "rope_deltas": rope_deltas,
        }

    def batch_encode_images(self, pil_images: list, image_processor_config: Optional[dict] = None):
        """
        批量 vision encoding：一次性图像预处理 + 逐图 vision encode。

        图像预处理（resize + normalize）可以一次性完成，但 vision encoder
        由于模型导出限制，不支持 image_grid_thw batch > 1（内部广播形状不兼容），
        因此 encoding 仍需逐图进行。

        相比原始 prepare_inputs 逐块调用的优势：
        - 图像预处理只调用一次 PaddleOCRVLImageProcessor（避免多次构造）
        - 跳过文本处理和 tokenizer 开销，仅做 vision 部分

        Args:
            pil_images: PIL Image 列表
            image_processor_config: 图像处理器配置

        Returns:
            list of (image_embeds, image_grid_thw)，每个元素对应一张图片，顺序与输入一致
        """
        if not pil_images:
            return []

        # 默认图像处理器配置
        default_config = {
            "resample": 3,
            "rescale_factor": 0.00392156862745098,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "min_pixels": 112896,
            "max_pixels": 1003520,
            "patch_size": 14,
            "temporal_patch_size": 1,
            "merge_size": 2,
        }
        if image_processor_config:
            default_config.update(image_processor_config)

        image_processor = PaddleOCRVLImageProcessor(**default_config)

        # 一次性图像预处理（resize + normalize）
        images_info = image_processor(images=pil_images, return_tensors="pt")
        pixel_values = images_info["pixel_values"]   # [total_patches, 3, 14, 14]
        image_grid_thw = images_info["image_grid_thw"]  # [n_images, 3]

        merge_size = default_config.get("merge_size", 2)
        n_images = len(pil_images)

        # 计算每张图的 patch 数和 token 数
        patches_per_image = []
        tokens_per_image = []
        for thw in image_grid_thw:
            t, h, w = thw[0].item(), thw[1].item(), thw[2].item()
            patches_per_image.append(t * h * w)
            tokens_per_image.append(t * (h // merge_size) * (w // merge_size))

        # 按 pixel_values 拆分每张图的 patches
        all_patches = []
        offset = 0
        for n_patch in patches_per_image:
            all_patches.append(pixel_values[offset:offset + n_patch])
            offset += n_patch

        # 逐图 vision encoding（模型不支持多图 batch）
        # 注意：vision_model 返回的 tensor 是 OV 输出 buffer 的 view，
        # 下次调用会覆盖，必须 .clone() 保存副本。
        results = []
        for i in range(n_images):
            img_thw = image_grid_thw[i:i+1]  # [1, 3]
            img_embeds = self.vision_model(all_patches[i], img_thw)
            if isinstance(img_embeds, (list, tuple)):
                img_embeds = torch.cat(img_embeds, dim=0).clone()
            elif isinstance(img_embeds, torch.Tensor):
                img_embeds = img_embeds.view(-1, img_embeds.shape[-1]).clone()
            results.append((img_embeds, img_thw))

        return results

    def prepare_inputs_from_embeds(
        self,
        messages,
        image_embeds: torch.Tensor,
        image_grid_thw: torch.Tensor,
        chat_template=None,
        image_processor_config=None,
    ):
        """
        使用预计算的 vision embeddings 准备 LLM 输入（跳过 vision encoding）。

        Args:
            messages: 消息列表
            image_embeds: 预计算的图像 embeddings [n_tokens, hidden_dim]
            image_grid_thw: 图像网格信息 [1, 3]
            chat_template: 聊天模板
            image_processor_config: 图像处理器配置（仅用于计算 placeholder token 数量）
        """
        if chat_template is None:
            chat_template = _DEFAULT_CHAT_TEMPLATE

        # 渲染文本模板
        text, _ = render_jinja_template(
            conversations=[messages],
            chat_template=chat_template,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        if not isinstance(text, list):
            text = [text]

        # 替换图像占位符（使用 image_grid_thw 计算 token 数量）
        index = 0
        for i in range(len(text)):
            while "<|IMAGE_PLACEHOLDER|>" in text[i]:
                text[i] = text[i].replace(
                    "<|IMAGE_PLACEHOLDER|>",
                    "<|placeholder|>" * (image_grid_thw[index].prod() // 2 // 2),
                    1,
                )
                index += 1
            text[i] = text[i].replace("<|placeholder|>", "<|IMAGE_PLACEHOLDER|>")

        # Tokenize
        text_inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = text_inputs["input_ids"]
        attention_mask = text_inputs["attention_mask"]

        # LLM embedding
        inputs_embeds = self.llm_embd_run(input_ids)

        # 合并 vision embeddings
        if isinstance(image_embeds, torch.Tensor):
            image_embeds = image_embeds.view(-1, image_embeds.shape[-1])

        n_image_tokens = (input_ids == 100295).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        mask = input_ids == 100295
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        image_mask = mask_expanded.to(inputs_embeds.device)

        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        position_ids, rope_deltas = self.get_rope_index(
            input_ids, image_grid_thw, None, None, attention_mask,
        )

        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "rope_deltas": rope_deltas,
        }

    @staticmethod
    def _apply_repetition_penalty(logits, generated_ids, penalty=1.5):
        """对已生成的 token 施加 repetition penalty，降低重复概率。"""
        if penalty == 1.0 or not generated_ids:
            return logits
        prev_tokens = torch.tensor(generated_ids, dtype=torch.long, device=logits.device)
        score = torch.gather(logits, -1, prev_tokens.unsqueeze(0)).squeeze(0)
        # 正 logit 除以 penalty（降低），负 logit 乘以 penalty（更负）
        score = torch.where(score > 0, score / penalty, score * penalty)
        logits.scatter_(-1, prev_tokens.unsqueeze(0), score.unsqueeze(0))
        return logits

    def generate_from_prepared(self, prepared, generation_config, stopping_criteria=None, block_label=None):
        """
        从已准备好的 inputs 执行自回归生成（LLM decode 阶段）。

        Args:
            block_label: 可选，layout 检测的 block 类型标签。
                         若提供，则根据标签自动决定 repetition_penalty：
                         非文本类（NON_TEXT_PENALTY_LABELS）→ 1.0，文本类 → TEXT_REPETITION_PENALTY。
        """
        self.rope_deltas = prepared["rope_deltas"]

        generate_kwargs = dict(
            inputs_embeds=prepared["inputs_embeds"],
            attention_mask=prepared["attention_mask"],
            position_ids=prepared["position_ids"],
            **generation_config,
        )
        # 根据 block 类型决定 repetition_penalty
        if block_label is not None:
            generate_kwargs["repetition_penalty"] = (
                1.0 if block_label in NON_TEXT_PENALTY_LABELS else TEXT_REPETITION_PENALTY
            )
        elif "repetition_penalty" not in generate_kwargs:
            generate_kwargs["repetition_penalty"] = 1.0
        if stopping_criteria is not None:
            generate_kwargs["stopping_criteria"] = stopping_criteria
        generation_output = self.generate(**generate_kwargs)
        output_token_count = generation_output.shape[1]
        response = self.tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]

        infer_times = list(self.llm_infer_list)
        first_token_latency_ms = infer_times[0] if len(infer_times) > 0 else 0.0
        decode_times = infer_times[1:] if len(infer_times) > 1 else []
        decode_avg_ms = sum(decode_times) / len(decode_times) if decode_times else 0.0

        token_stats = {
            "output_tokens": output_token_count,
            "first_token_latency_ms": first_token_latency_ms,
            "decode_avg_ms": decode_avg_ms,
        }
        return response, token_stats

    def _get_batch_request(self):
        """获取用于 batch 推理的独立 infer request。"""
        if not hasattr(self, '_batch_request') or self._batch_request is None:
            self._batch_request = self.llm_compiled_model.create_infer_request()
        return self._batch_request

    def batch_generate(
        self,
        prepared_list: list,
        max_new_tokens: int = 4096,
        eos_token_id: int = None,
        repetition_penalty: float = 1.0,
        block_labels: list = None,
    ) -> list:
        """
        Slot-based batch inference：多个 prepared inputs 一起做 prefill + decode。

        Args:
            block_labels: 可选，每个 slot 对应的 layout block 类型标签列表。
                          若提供，则按 slot 独立决定 repetition_penalty（优先于 repetition_penalty 参数）。
        """
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id

        real_count = len(prepared_list)
        batch_size = real_count

        # 计算每个 slot 的 repetition_penalty
        if block_labels is not None:
            slot_penalties = [
                1.0 if lb in NON_TEXT_PENALTY_LABELS else TEXT_REPETITION_PENALTY
                for lb in block_labels
            ]
        else:
            slot_penalties = [repetition_penalty] * batch_size

        if _BATCH_VERBOSE:
            print(f"    [BatchGen] batch_generate called, batch_size={batch_size}, slot_penalties={slot_penalties}", flush=True)
        batch_request = self._get_batch_request()

        # Pad inputs_embeds 到同一长度（左填充）
        seq_lens = [p["inputs_embeds"].shape[1] for p in prepared_list]
        max_seq = max(seq_lens)
        hidden_dim = prepared_list[0]["inputs_embeds"].shape[2]

        if _BATCH_VERBOSE:
            print(f"    [BatchGen] seq_lens={seq_lens}, max_seq={max_seq}, hidden_dim={hidden_dim}", flush=True)

        batch_embeds = torch.zeros(batch_size, max_seq, hidden_dim)
        batch_mask = torch.zeros(batch_size, max_seq, dtype=torch.long)
        batch_pos = torch.zeros(3, batch_size, max_seq, dtype=torch.long)
        rope_deltas_list = []

        for i in range(batch_size):
            p = prepared_list[i]
            sl = p["inputs_embeds"].shape[1]
            pad_len = max_seq - sl
            batch_embeds[i, pad_len:, :] = p["inputs_embeds"][0]
            batch_mask[i, pad_len:] = p["attention_mask"][0]
            batch_pos[:, i, pad_len:] = p["position_ids"][:, 0, :]
            rope_deltas_list.append(p["rope_deltas"])

        beam_idx = np.arange(batch_size, dtype=np.int64)

        # Prefill
        batch_request.reset_state()

        if _BATCH_VERBOSE:
            print(f"    [BatchGen] Prefill: embeds={batch_embeds.shape}, mask={batch_mask.shape}, pos={batch_pos.shape}", flush=True)
        _t_prefill_start = time.perf_counter()
        batch_request.start_async({
            'inputs_embeds': batch_embeds.numpy(),
            'attention_mask': batch_mask.numpy(),
            'position_ids': batch_pos.numpy(),
            'beam_idx': beam_idx,
        }, share_inputs=True)
        batch_request.wait()
        _t_prefill = (time.perf_counter() - _t_prefill_start) * 1000
        if _BATCH_VERBOSE:
            print(f"    [BatchGen] Prefill done: {_t_prefill:.1f}ms", flush=True)

        logits = torch.from_numpy(batch_request.get_tensor("logits").data)

        # 初始化 decode 状态
        past_lens = list(seq_lens)
        generated_ids = [[] for _ in range(batch_size)]
        finished = [False] * batch_size

        for i in range(batch_size):
            row_logits = logits[i, 0, :].unsqueeze(0)
            row_logits = self._apply_repetition_penalty(row_logits, generated_ids[i], slot_penalties[i])
            next_token = row_logits.squeeze(0).argmax().item()
            generated_ids[i].append(next_token)
            if next_token == eos_token_id:
                finished[i] = True

        decode_times = []
        rope_delta_vals = [rd.item() for rd in rope_deltas_list]

        # Pre-allocate attention_mask
        max_total_len = max_seq + max_new_tokens
        batch_mask_full = torch.zeros(batch_size, max_total_len, dtype=torch.long)
        batch_mask_full[:, :max_seq] = batch_mask[:, :max_seq]
        current_mask_len = max_seq

        # Pre-cache pad_token embedding and reusable tensors
        pad_emb = self.llm_embd_run(torch.tensor([[self.pad_token_id]], dtype=torch.long))[0]
        new_pos = torch.zeros(3, batch_size, 1, dtype=torch.long)
        new_embeds = pad_emb.unsqueeze(0).expand(batch_size, -1, -1).contiguous().clone()
        new_pos_np = new_pos.numpy()
        new_embeds_np = new_embeds.numpy()

        # Decode loop
        for step in range(1, max_new_tokens):
            if all(finished):
                break

            token_ids = []
            active_indices = []
            for i in range(batch_size):
                if not finished[i]:
                    token_ids.append(generated_ids[i][-1])
                    active_indices.append(i)

            for i in range(batch_size):
                if finished[i]:
                    new_embeds[i] = pad_emb

            if active_indices:
                batch_token_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(1)
                batch_emb = self.llm_embd_run(batch_token_ids)
                for j, idx in enumerate(active_indices):
                    new_embeds[idx] = batch_emb[j]

            batch_mask_full[:, current_mask_len] = 1
            current_mask_len += 1
            batch_mask_view = batch_mask_full[:, :current_mask_len]

            for i in range(batch_size):
                new_pos[:, i, 0] = past_lens[i] + rope_delta_vals[i]

            _t_decode_start = time.perf_counter()
            batch_request.start_async({
                'inputs_embeds': new_embeds_np,
                'attention_mask': batch_mask_view.numpy(),
                'position_ids': new_pos_np,
                'beam_idx': beam_idx,
            }, share_inputs=True)
            batch_request.wait()
            _t_decode = (time.perf_counter() - _t_decode_start) * 1000
            decode_times.append(_t_decode)

            if _BATCH_VERBOSE and (step <= 3 or step % 50 == 0):
                print(f"    [BatchGen] decode step={step}, infer={_t_decode:.1f}ms, mask_len={current_mask_len}, finished={finished}", flush=True)

            logits = torch.from_numpy(batch_request.get_tensor("logits").data)

            for i in range(batch_size):
                past_lens[i] += 1
                if finished[i]:
                    continue

                row_logits = logits[i, 0, :].unsqueeze(0)
                row_logits = self._apply_repetition_penalty(row_logits, generated_ids[i], slot_penalties[i])
                next_token = row_logits.squeeze(0).argmax().item()
                generated_ids[i].append(next_token)

                if next_token == eos_token_id:
                    finished[i] = True

        # 解码结果
        total_steps = len(decode_times)
        if _BATCH_VERBOSE:
            print(f"    [BatchGen] Decode loop finished. Total steps={total_steps}, finished={finished}", flush=True)
        results = []
        for i in range(batch_size):
            output_ids = torch.tensor([generated_ids[i]], dtype=torch.long)
            response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

            n_output = len(generated_ids[i])
            n_decode = n_output - 1  # first token from prefill
            if n_decode > 0 and decode_times:
                per_seq_decode = decode_times[:n_decode]
                decode_avg = sum(per_seq_decode) / len(per_seq_decode)
                total_decode_ms = sum(per_seq_decode)
            else:
                decode_avg = 0.0
                total_decode_ms = 0.0
            if _BATCH_VERBOSE:
                print(f"    [BatchGen] slot{i}: output_tokens={n_output}, decode_steps={n_decode}, "
                      f"decode_avg={decode_avg:.1f}ms/tok, total_decode={total_decode_ms:.1f}ms", flush=True)

            token_stats = {
                "output_tokens": n_output,
                "first_token_latency_ms": _t_prefill,
                "decode_avg_ms": decode_avg,
                "total_decode_ms": total_decode_ms,
            }
            results.append((response, token_stats))

        return results
