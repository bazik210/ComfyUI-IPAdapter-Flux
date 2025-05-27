import torch
import os
import logging
import folder_paths
import json
from transformers import AutoProcessor, SiglipVisionModel, AutoModel
from PIL import Image
import numpy as np
from .attention_processor_advanced import IPAFluxAttnProcessor2_0Advanced
from .utils import is_model_patched, FluxUpdateModules
from safetensors.torch import load_file

# Define model directories
MODELS_DIR = os.path.join(folder_paths.models_dir, "ipadapter-flux")
CLIP_VISION_DIR = os.path.join(folder_paths.models_dir, "clip_vision")
if "ipadapter-flux" not in folder_paths.folder_names_and_paths:
    current_paths = [MODELS_DIR]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["ipadapter-flux"]
    MODELS_DIR = current_paths[0]

folder_paths.folder_names_and_paths["ipadapter-flux"] = (current_paths, folder_paths.supported_pt_extensions)

# Mapping of CLIP Vision model files to Hugging Face models
CLIP_MODEL_MAPPING = {
    "clip-vit-large-patch14.safetensors": "openai/clip-vit-large-patch14",
    "siglip-so400m-patch14-384.safetensors": "google/siglip-so400m-patch14-384",
    "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
}

# Scan clip_vision directory for models
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

    return sorted(list(set(clip_vision_models)))

class MLPProjModelAdvanced(torch.nn.Module):
    def __init__(self, cross_attention_dim=4096, id_embeddings_dim=1152, num_tokens=4):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim * 2, cross_attention_dim * num_tokens),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, id_embeds):
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x

class InstantXFluxIPAdapterModelAdvanced:
    def __init__(self, image_encoder_path, ip_ckpt, device, dtype=torch.float16):
        self.device = device
        self.dtype = dtype
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt

        # Load CLIP Vision model
        try:
            clip_path = os.path.join(CLIP_VISION_DIR, image_encoder_path.replace("/", os.sep))
            if os.path.exists(clip_path):
                logging.info(f"Loading CLIP Vision from local path: {clip_path}")
                if clip_path.endswith(".safetensors"):
                    state_dict = load_file(clip_path, device="cpu")
                    pos_embedding_key = "vision_model.embeddings.position_embedding.weight"
                    if pos_embedding_key in state_dict and state_dict[pos_embedding_key].shape[0] == 729:
                        logging.warning("Patching position_embedding from 729 to 730")
                        padding = torch.zeros(1, state_dict[pos_embedding_key].shape[1], device="cpu")
                        state_dict[pos_embedding_key] = torch.cat([state_dict[pos_embedding_key], padding], dim=0)
                    model_name = os.path.basename(clip_path)
                    hf_model = CLIP_MODEL_MAPPING.get(model_name, image_encoder_path)
                    self.image_encoder = SiglipVisionModel.from_pretrained(
                        hf_model, state_dict=None, ignore_mismatched_sizes=True
                    ).to(self.device, dtype=self.dtype)
                    self.image_encoder.load_state_dict(state_dict, strict=False)
                    self.clip_image_processor = AutoProcessor.from_pretrained(hf_model)
                else:
                    self.image_encoder = AutoModel.from_pretrained(clip_path).to(self.device, dtype=self.dtype)
                    self.clip_image_processor = AutoProcessor.from_pretrained(clip_path)
            else:
                logging.info(f"Loading CLIP Vision from Hugging Face: {image_encoder_path}")
                self.image_encoder = SiglipVisionModel.from_pretrained(image_encoder_path).to(self.device, dtype=self.dtype)
                self.clip_image_processor = AutoProcessor.from_pretrained(image_encoder_path)
        except Exception as e:
            logging.error(f"Failed to load CLIP Vision: {e}")
            raise

        # Load IP-Adapter state_dict
        path = os.path.join(MODELS_DIR, ip_ckpt)
        if path.endswith('.safetensors'):
            state_dict = load_file(path, device="cpu")
            logging.info(f"Loaded {ip_ckpt} using safetensors")
        else:
            state_dict = torch.load(path, map_location="cpu")
            logging.info(f"Loaded {ip_ckpt} using torch.load")

        self.joint_attention_dim = 4096
        self.hidden_size = 3072

        # Log state_dict keys
        all_keys = list(state_dict.keys())
        logging.info(f"State dict keys (first 10): {all_keys[:10]}")

        # Determine num_tokens
        self.num_tokens = 4
        if "ip_adapter" in state_dict:
            ip_adapter_dict = state_dict.get("ip_adapter", {})
            for k in ip_adapter_dict.keys():
                if k.endswith("to_k_ip.weight"):
                    self.num_tokens = 128
                    logging.info(f"Detected InstantX via ip_adapter.{k}, num_tokens={self.num_tokens}")
                    break
        elif 'flux-ip-adapter' in ip_ckpt.lower():
            self.num_tokens = 128
            logging.info(f"Detected XLabs model {ip_ckpt}, num_tokens={self.num_tokens}")
        if self.num_tokens == 4:
            image_proj_dict = state_dict.get("image_proj", {})
            if "proj.2.weight" in image_proj_dict:
                weight_shape = image_proj_dict["proj.2.weight"].shape
                if weight_shape[0] % self.joint_attention_dim == 0:
                    self.num_tokens = weight_shape[0] // self.joint_attention_dim
                    logging.info(f"Detected num_tokens={self.num_tokens} from image_proj.proj.2.weight shape={weight_shape}")
        logging.info(f"Final num_tokens={self.num_tokens}")

        # Initialize image_proj model
        image_proj_dict = state_dict.get("image_proj", {})
        self.image_proj_model = MLPProjModelAdvanced(
            cross_attention_dim=self.joint_attention_dim, # 4096
            id_embeddings_dim=1152,
            num_tokens=self.num_tokens,
        ).to(self.device, dtype=self.dtype)
        try:
            self.image_proj_model.load_state_dict(image_proj_dict, strict=True)
            logging.info("Loaded image_proj successfully")
        except Exception as e:
            logging.error(f"Failed to load image_proj: {e}")
            raise

        # Initialize ip_adapter
        ip_adapter_dict = state_dict.get("ip_adapter", {})
        if not ip_adapter_dict:
            for k in all_keys:
                if k.startswith("double_blocks.") or k.startswith("single_blocks."):
                    new_key = k.replace("double_blocks.", "").replace("single_blocks.", "")
                    new_key = new_key.replace("processor.ip_adapter_double_stream_", "").replace("processor.ip_adapter_single_stream_", "")
                    ip_adapter_dict[new_key] = state_dict[k]
            logging.info("Using XLabs ip_adapter format")

        if not ip_adapter_dict:
            logging.error("No ip_adapter keys found")
            raise KeyError(f"No ip_adapter keys in {path}")

        self.ip_attn_procs = self.init_ip_adapter_advanced()
        ip_layers = torch.nn.ModuleList(self.ip_attn_procs.values())
        try:
            ip_layers.load_state_dict(ip_adapter_dict, strict=True)
            logging.info("Loaded ip_adapter successfully")
        except Exception as e:
            logging.error(f"Failed to load ip_adapter: {e}")
            raise

        del state_dict

    def init_ip_adapter_advanced(self, weight_params=(1.0, 1.0, 10), timestep_percent_range=(0.0, 1.0)):
        # Initialize attention processors for double and single blocks
        weight_start, weight_end, steps = weight_params
        ip_attn_procs = {}
        dsb_count = 19 #len(flux_model.diffusion_model.double_blocks)
        for i in range(dsb_count):
            name = f"double_blocks.{i}"
            ip_attn_procs[name] = IPAFluxAttnProcessor2_0Advanced(
                hidden_size=self.hidden_size,
                cross_attention_dim=self.joint_attention_dim,
                num_tokens=self.num_tokens,
                scale_start=weight_start,
                scale_end=weight_end,
                total_steps=steps,
                timestep_range=timestep_percent_range
            ).to(self.device, dtype=self.dtype)
        ssb_count = 38 #len(flux_model.diffusion_model.single_blocks)
        for i in range(ssb_count):
            name = f"single_blocks.{i}"
            ip_attn_procs[name] = IPAFluxAttnProcessor2_0Advanced(
                hidden_size=self.hidden_size,
                cross_attention_dim=self.joint_attention_dim,
                num_tokens=self.num_tokens,
                scale_start=weight_start,
                scale_end=weight_end,
                total_steps=steps,
                timestep_range=timestep_percent_range
            ).to(self.device, dtype=self.dtype)
        return ip_attn_procs

    def update_ip_adapter_advanced(self, flux_model, weight_params, timestep_percent_range=(0.0, 1.0)):
        # Update attention processors with new weights and timestep range
        weight_start, weight_end, steps = weight_params
        s = flux_model.model_sampling
        percent_to_timestep_function = lambda a: s.percent_to_sigma(a)
        timestep_range = (percent_to_timestep_function(timestep_percent_range[0]), percent_to_timestep_function(timestep_percent_range[1]))
        for ip_layer in self.ip_attn_procs.values():
            ip_layer.scale_start = weight_start
            ip_layer.scale_end = weight_end
            ip_layer.total_steps = steps
            ip_layer.timestep_range = timestep_range
        logging.info(f"Updated ip_adapter with weight_start={weight_start}, weight_end={weight_end}, steps={steps}")

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        # Generate image embeddings from PIL image or provided embeddings
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=self.dtype)).pooler_output
            clip_image_embeds = clip_image_embeds.to(dtype=self.dtype)
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=self.dtype)
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        return image_prompt_embeds

class IPAdapterFluxLoaderAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ipadapter": (folder_paths.get_filename_list("ipadapter-flux"),),
                "clip_vision": (get_clip_vision_models(),),
                "provider": (["cuda", "cpu", "mps"],),
                "dtype": (["float16", "bfloat16"], {"default": "float16"}),
            }
        }

    RETURN_TYPES = ("IP_ADAPTER_FLUX_INSTANTX",)
    RETURN_NAMES = ("ipadapterFlux",)
    FUNCTION = "load_model_advanced"
    CATEGORY = "InstantXNodes"

    def load_model_advanced(self, ipadapter, clip_vision, provider, dtype="float16"):
        # Load the advanced IP-Adapter model
        logging.info(f"Loading InstantX IPAdapter Flux Advanced model: {ipadapter}, clip_vision={clip_vision}, dtype={dtype}")
        model = InstantXFluxIPAdapterModelAdvanced(
            image_encoder_path=clip_vision,
            ip_ckpt=ipadapter,
            device=provider,
            dtype=getattr(torch, dtype)
        )
        return (model,)

class ApplyIPAdapterFluxAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "ipadapter_flux": ("IP_ADAPTER_FLUX_INSTANTX",),
                "image": ("IMAGE",),
                "weight_start": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 5.0, "step": 0.05}),
                "weight_end": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 5.0, "step": 0.05}),
                "steps": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_ipadapter_flux_advanced"
    CATEGORY = "InstantXNodes"

    def apply_ipadapter_flux_advanced(self, model, ipadapter_flux, image, weight_start, weight_end, steps, start_percent, end_percent):
        # Clean up old processors if they exist
        if hasattr(model.model, '_ip_attn_procs'):
            for proc in model.model._ip_attn_procs.values():
                proc.clear_memory()  # Add a new method for cleanup
            del model.model._ip_attn_procs

        # Convert image tensor to PIL
        pil_image = image.numpy()[0] * 255.0
        pil_image = Image.fromarray(pil_image.astype(np.uint8))

        # Reset attention processor instances
        IPAFluxAttnProcessor2_0Advanced.reset_all_instances()

        # Update IP-Adapter with new parameters
        ipadapter_flux.update_ip_adapter_advanced(model.model, (weight_start, weight_end, steps), (start_percent, end_percent))

        # Get image embeddings
        image_prompt_embeds = ipadapter_flux.get_image_embeds(pil_image=pil_image, clip_image_embeds=None)
        
        # Apply IP-Adapter to the model
        is_patched = is_model_patched(model.model)
        bi = model.clone()
        FluxUpdateModules(bi, ipadapter_flux.ip_attn_procs, image_prompt_embeds, is_patched)

        # Store reference to processors
        bi.model._ip_attn_procs = ipadapter_flux.ip_attn_procs
        logging.info(f"Applied IPAdapter with weight_start={weight_start}, weight_end={weight_end}, steps={steps}")
        return (bi,)

NODE_CLASS_MAPPINGS = {
    "IPAdapterFluxLoaderAdvanced": IPAdapterFluxLoaderAdvanced,
    "ApplyIPAdapterFluxAdvanced": ApplyIPAdapterFluxAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IPAdapterFluxLoaderAdvanced": "Load IP-Adapter Advanced",
    "ApplyIPAdapterFluxAdvanced": "Apply IP-Adapter Advanced",
}