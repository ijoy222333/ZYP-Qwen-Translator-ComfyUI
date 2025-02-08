from comfy import model_management
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import comfy.utils
import os

MODEL_PATH = "/root/ComfyUI/models/LLM/Qwen2.5-3B-Instruct"

class QwenTranslator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "dynamicPrompts": False
                }),
                "target_lang": (["English", "Chinese", "Japanese", "French", "German"], {"default": "English"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "max_tokens": ("INT", {"default": 512, "min": 32, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.01, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 50, "min": 1, "max": 100}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.01, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("translated_text",)
    FUNCTION = "translate"
    CATEGORY = "QwenTools"

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = model_management.get_torch_device()
        self.dtype = torch.float16 if model_management.should_use_fp16() else torch.float32

    def load_model(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Qwen model not found at {MODEL_PATH}")
            
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_PATH,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                device_map=self.device,
                trust_remote_code=True,
                torch_dtype=self.dtype
            ).eval()

    def translate(self, text, target_lang, seed, max_tokens, temperature, top_k, top_p):
        # 输入验证
        if not text.strip():
            raise ValueError("Input text cannot be empty")
            
        self.load_model()
        torch.manual_seed(seed)

        # 构建符合Qwen格式的prompt
        prompt = f"""<|im_start|>system
你是一个专业翻译助手，请将内容准确翻译成{target_lang}，保持原意不变，不添加额外内容。<|im_end|>
<|im_start|>user
{text}<|im_end|>
<|im_start|>assistant
"""

        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # 生成参数
        generate_kwargs = {
            "input_ids": inputs.input_ids,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id
        }

        # 执行生成
        with torch.inference_mode():
            outputs = self.model.generate(**generate_kwargs)

        # 解码输出
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        ).strip()

        return (response,)

NODE_CLASS_MAPPINGS = {"QwenTranslator": QwenTranslator}
NODE_DISPLAY_NAME_MAPPINGS = {"QwenTranslator": "🌐 ZYP Qwen Translator"}
