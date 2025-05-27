import torch
import os
import logging
import folder_paths
from transformers import AutoProcessor, CLIPVisionModel, AutoModel
from PIL import Image
import numpy as np
from .attention_processor import IPAFluxAttnProcessor2_0
from .utils import is_model_patched, FluxUpdateModules
from safetensors.torch import load_file

# Папки для моделей
MODELS_DIR = os.path.join(folder_paths.models_dir, "ipadapter-flux")
CLIP_VISION_DIR = os.path.join(folder_paths.models_dir, "clip_vision")
if "ipadapter-flux" not in folder_paths.folder_names_and_paths:
    current_paths = [MODELS_DIR]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["ipadapter-flux"]
    MODELS_DIR = current_paths[0]

folder_paths.folder_names_and_paths["ipadapter-flux"] = (current_paths, folder_paths.supported_pt_extensions)

# Mapping of file names to HF models
CLIP_MODEL_MAPPING = {
    "clip-vit-large-patch14.safetensors": "openai/clip-vit-large-patch14",
    "siglip-so400m-patch14-384.safetensors": "google/siglip-so400m-patch14-384",
    "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
}

# Scanning folder clip_vision
def get_clip_vision_models():
    clip_vision_models = []
    local_model_names = set()

    for root, dirs, files in os.walk(CLIP_VISION_DIR):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            has_config = os.path.exists(os.path.join(dir_path, "config.json"))
            has_model = os.path.exists(os.path.join(dir_path, "pytorch_model.bin")) or os.path.exists(os.path.join(dir_path, "model.safetensors"))
            if has_config and has_model:
                relative_path = os.path.relpath(dir_path, CLIP_VISION_DIR).replace(os.sep, "/")
                clip_vision_models.append(relative_path)
                model_name = relative_path.split("/")[-1]
                local_model_names.add(model_name)
                logging.info(f"Found valid CLIP Vision directory: {relative_path}")
        for file_name in files:
            if file_name.endswith(".safetensors"):
                relative_path = os.path.relpath(os.path.join(root, file_name), CLIP_VISION_DIR).replace(os.sep, "/")
                clip_vision_models.append(relative_path)
                model_name = file_name.replace(".safetensors", "")
                local_model_names.add(model_name)
                logging.info(f"Found CLIP Vision safetensors file: {relative_path}")

    hf_models = [
        "google/siglip-so400m-patch14-384",
        "openai/clip-vit-large-patch14",
        "openai/clip-vit-base-patch32"
    ]
    for hf_model in hf_models:
        model_name = hf_model.split("/")[-1]
        if model_name not in local_model_names:
            clip_vision_models.append(hf_model)
            logging.info(f"Added HF model to list: {hf_model}")
        else:
            logging.info(f"Skipped HF model {hf_model} due to local version: {model_name}")

    unique_models = sorted(list(set(clip_vision_models)))
    logging.info(f"Available CLIP Vision models: {unique_models}")
    return unique_models

class MLPProjModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, num_tokens=4):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim*2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim*2, cross_attention_dim*num_tokens),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, id_embeds):
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x

class InstantXFluxIPAdapterModel:
    def __init__(self, image_encoder_path, ip_ckpt, device, dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt

        # Loading image encoder
        try:
            clip_path = os.path.join(CLIP_VISION_DIR, image_encoder_path.replace("/", os.sep))
            if os.path.exists(clip_path):
                logging.info(f"Loading CLIP Vision from local path: {clip_path}")
                if clip_path.endswith(".safetensors"):
                    state_dict = load_file(clip_path, device="cpu")
                    # Guessing model by filename
                    model_name = os.path.basename(clip_path)
                    hf_model = CLIP_MODEL_MAPPING.get(model_name, "openai/clip-vit-large-patch14")
                    self.image_encoder = CLIPVisionModel.from_pretrained(hf_model, state_dict=None).to(self.device, dtype=self.dtype)
                    self.image_encoder.load_state_dict(state_dict, strict=False)
                    self.clip_image_processor = AutoProcessor.from_pretrained(hf_model)
                else:
                    self.image_encoder = AutoModel.from_pretrained(clip_path).to(self.device, dtype=self.dtype)
                    self.clip_image_processor = AutoProcessor.from_pretrained(clip_path)
            else:
                logging.info(f"Loading CLIP Vision from Hugging Face: {image_encoder_path}")
                self.image_encoder = AutoModel.from_pretrained(image_encoder_path).to(self.device, dtype=self.dtype)
                self.clip_image_processor = AutoProcessor.from_pretrained(image_encoder_path)
        except Exception as e:
            logging.error(f"Failed to load clip image processor: {e}")
            raise

        # Loading IP-Adapter
        path = os.path.join(MODELS_DIR, self.ip_ckpt)
        logging.info(f"Loading IP-Adapter: {path}")
        if path.endswith('.safetensors'):
            state_dict = load_file(path, device="cpu")
            logging.info("Loaded using safetensors")
        else:
            state_dict = torch.load(path, map_location="cpu")
            logging.info("Loaded using torch.load")

        self.joint_attention_dim = 4096
        self.hidden_size = 3072

        all_keys = list(state_dict.keys())
        logging.info(f"State dict keys (first 10): {all_keys[:10]}")

        # Default num_tokens
        self.num_tokens = 4
        # Check for XLabs model explicitly
        if 'flux' in ip_ckpt:
            self.num_tokens = 128
            logging.info(f"Set num_tokens={self.num_tokens} for XLabs model {ip_ckpt}")
        else:
            # Auto-detect num_tokens from to_k_ip.weight
            for k in all_keys:
                if k.endswith("ip_adapter_double_stream_k_proj.weight") or k.endswith("to_k_ip.weight"):
                    weight_shape = state_dict[k].shape
                    if weight_shape[1] % self.joint_attention_dim == 0:
                        detected_num_tokens = weight_shape[1] // self.joint_attention_dim
                        self.num_tokens = detected_num_tokens
                        logging.info(f"Detected num_tokens={self.num_tokens} from {k} shape={weight_shape}")
                        break
        logging.info(f"Final num_tokens={self.num_tokens}")

        # Checking image_proj
        image_proj_dict = None
        has_image_proj = any(k.startswith("image_proj.") for k in all_keys) or "image_proj" in state_dict
        if has_image_proj:
            if "image_proj" in state_dict:
                image_proj_dict = state_dict["image_proj"]
            else:
                image_proj_dict = {k.replace("image_proj.", ""): v for k in all_keys if k.startswith("image_proj.")}
            logging.info(f"Found image_proj keys: {list(image_proj_dict.keys())[:5]}")
            self.image_proj_model = MLPProjModel(
                cross_attention_dim=self.joint_attention_dim,
                id_embeddings_dim=1152,
                num_tokens=self.num_tokens,
            ).to(self.device, dtype=self.dtype)
            try:
                self.image_proj_model.load_state_dict(image_proj_dict, strict=True)
            except Exception as e:
                logging.error(f"Failed to load image_proj: {e}")
                raise
        else:
            logging.warning("No image_proj keys found. Using raw clip_image_embeds.")
            self.image_proj_model = None

        # Parsing ip_adapter
        ip_adapter_dict = {}
        if "ip_adapter" in state_dict:
            ip_adapter_dict = state_dict["ip_adapter"]
            logging.info("Using InstantX ip_adapter format")
        else:
            for k in all_keys:
                if k.startswith("double_blocks.") or k.startswith("single_blocks."):
                    new_key = k.replace("double_blocks.", "").replace("single_blocks.", "")
                    new_key = new_key.replace("processor.ip_adapter_double_stream_", "").replace("processor.ip_adapter_single_stream_", "")
                    ip_adapter_dict[new_key] = state_dict[k]
            logging.info("Using XLabs ip_adapter format")

        if not ip_adapter_dict:
            logging.error("No ip_adapter keys found")
            raise KeyError(f"No ip_adapter keys found in {path}")

        # Initialize ipadapter
        self.ip_attn_procs = self.init_ip_adapter()
        ip_layers = torch.nn.ModuleList(self.ip_attn_procs.values())
        logging.info(f"Loading ip_adapter with keys: {list(ip_adapter_dict.keys())[:5]}")
        try:
            ip_layers.load_state_dict(ip_adapter_dict, strict=False)
        except Exception as e:
            logging.error(f"Failed to load ip_adapter: {e}")
            raise

        del state_dict

    def init_ip_adapter(self, weight=1.0, timestep_percent_range=(0.0, 1.0)):
        ip_attn_procs = {}
        dsb_count = 19
        for i in range(dsb_count):
            name = f"double_blocks.{i}"
            ip_attn_procs[name] = IPAFluxAttnProcessor2_0(
                hidden_size=self.hidden_size,
                cross_attention_dim=self.joint_attention_dim,
                num_tokens=self.num_tokens,
                scale=weight,
                timestep_range=timestep_percent_range
            ).to(self.device, dtype=self.dtype)
        ssb_count = 38
        for i in range(ssb_count):
            name = f"single_blocks.{i}"
            ip_attn_procs[name] = IPAFluxAttnProcessor2_0(
                hidden_size=self.hidden_size,
                cross_attention_dim=self.joint_attention_dim,
                num_tokens=self.num_tokens,
                scale=weight,
                timestep_range=timestep_percent_range
            ).to(self.device, dtype=self.dtype)
        return ip_attn_procs

    def update_ip_adapter(self, flux_model, weight, timestep_percent_range=(0.0, 1.0)):
        s = flux_model.model_sampling
        percent_to_timestep_function = lambda a: s.percent_to_sigma(a)
        timestep_range = (percent_to_timestep_function(timestep_percent_range[0]), percent_to_timestep_function(timestep_percent_range[1]))
        for ip_layer in self.ip_attn_procs.values():
            ip_layer.scale = weight
            ip_layer.timestep_range = timestep_range

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=self.dtype)).pooler_output
            clip_image_embeds = clip_image_embeds.to(dtype=self.dtype)
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=self.dtype)

        if self.image_proj_model is None:
            logging.info("Using raw clip_image_embeds")
            target_size = self.joint_attention_dim
            if clip_image_embeds.shape[1] != target_size:
                logging.warning(f"Projecting clip_image_embeds from {clip_image_embeds.shape[1]} to joint_attention_dim={target_size}")
                proj = torch.nn.Linear(clip_image_embeds.shape[1], target_size).to(self.device, dtype=self.dtype)
                torch.nn.init.xavier_uniform_(proj.weight)
                proj.bias.data.fill_(0)
                clip_image_embeds = proj(clip_image_embeds)
            clip_image_embeds = clip_image_embeds.unsqueeze(1).repeat(1, self.num_tokens, 1)
            return clip_image_embeds
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        return image_prompt_embeds

class IPAdapterFluxLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ipadapter": (folder_paths.get_filename_list("ipadapter-flux"),),
                "clip_vision": (get_clip_vision_models(),),
                "provider": (["cuda", "cpu", "mps"],),
                "dtype": (["auto", "float16", "bfloat16"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("IP_ADAPTER_FLUX_INSTANTX",)
    RETURN_NAMES = ("ipadapterFlux",)
    FUNCTION = "load_model"
    CATEGORY = "InstantXNodes"

    def load_model(self, ipadapter, clip_vision, provider, dtype="auto"):
        logging.info(f"Loading InstantX IPAdapter Flux model with clip_vision={clip_vision}, dtype={dtype}")
        if dtype == "auto":
            dtype = "bfloat16" if provider == "cuda" else "float16"
        model = InstantXFluxIPAdapterModel(
            image_encoder_path=clip_vision,
            ip_ckpt=ipadapter,
            device=provider,
            dtype=getattr(torch, dtype)
        )
        return (model,)

class ApplyIPAdapterFlux:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "ipadapter_flux": ("IP_ADAPTER_FLUX_INSTANTX",),
                "image": ("IMAGE",),
                "weight": ("FLOAT", {"default": 0.5, "min": -1.0, "max": 5.0, "step": 0.05}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_ipadapter_flux"
    CATEGORY = "InstantXNodes"

    def apply_ipadapter_flux(self, model, ipadapter_flux, image, weight, start_percent, end_percent):
        pil_image = Image.fromarray((image.numpy()[0] * 255.0).astype(np.uint8))
        ipadapter_flux.update_ip_adapter(model.model, weight, (start_percent, end_percent))
        image_prompt_embeds = ipadapter_flux.get_image_embeds(pil_image=pil_image, clip_image_embeds=None)
        is_patched = is_model_patched(model.model)
        bi = model.clone()
        FluxUpdateModules(bi, ipadapter_flux.ip_attn_procs, image_prompt_embeds, is_patched)
        return (bi,)

NODE_CLASS_MAPPINGS = {
    "IPAdapterFluxLoader": IPAdapterFluxLoader,
    "ApplyIPAdapterFlux": ApplyIPAdapterFlux,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IPAdapterFluxLoader": "Load IPAdapter Flux Model",
    "ApplyIPAdapterFlux": "Apply IPAdapter Flux Model",
}