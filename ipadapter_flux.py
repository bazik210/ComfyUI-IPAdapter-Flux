import torch
import os
import logging
import folder_paths
import json
from transformers import AutoProcessor, CLIPVisionModel, SiglipVisionModel
from PIL import Image
import numpy as np
from .attention_processor import IPAFluxAttnProcessor2_0
from .utils import is_model_patched, FluxUpdateModules, _compute_sha256
from safetensors.torch import load_file

KNOWN_XLAB_HASHES = {
    "750f912149b84bbb0c2a6ce90ffa7e78afd1795821407718724ebcd36372dc2d",
    "8f2bfddaffc4fe2a6667bef24c8ce6075e81d01d0f6b0f9adbe46ad686057ee2"
}

# Define model directories
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

class InstantXFluxIPAdapterModel:
    def __init__(self, image_encoder_path, ip_ckpt, device, dtype=torch.bfloat16, shuffle_weights=False):
        self.device = device
        self.dtype = dtype
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.fallback_proj = None

        # Load Vision model
        try:
            clip_path = os.path.join(CLIP_VISION_DIR, image_encoder_path.replace("/", os.sep))
            if os.path.exists(clip_path):
                logging.info(f"Loading CLIP Vision from local path: {clip_path}")
                if clip_path.endswith(".safetensors"):
                    state_dict = load_file(clip_path, device="cpu")
                    # Determine hf_model dynamically
                    model_name = os.path.basename(clip_path)
                    hf_model = CLIP_MODEL_MAPPING.get(model_name)
                    if not hf_model:
                        # Check config.json in the same directory
                        config_path = os.path.join(os.path.dirname(clip_path), "config.json")
                        if os.path.exists(config_path):
                            with open(config_path, "r") as f:
                                config = json.load(f)
                                # Try multiple keys for model identification
                                hf_model = config.get("_name_or_path") or config.get("model_type") or config.get("architecture")
                                if hf_model:
                                    logging.info(f"Extracted hf_model from config.json: {hf_model}")
                                else:
                                    logging.warning(f"No valid model identifier in {config_path}. Checking local fallback model.")
                                    # Check for local clip-vit-large-patch14.safetensors
                                    fallback_model = "clip-vit-large-patch14.safetensors"
                                    fallback_path = os.path.join(CLIP_VISION_DIR, fallback_model)
                                    if os.path.exists(fallback_path) and fallback_model in CLIP_MODEL_MAPPING:
                                        hf_model = CLIP_MODEL_MAPPING[fallback_model]
                                        clip_path = fallback_path
                                        logging.info(f"Using local model: {clip_path} with hf_model: {hf_model}")
                                    else:
                                        hf_model = "openai/clip-vit-large-patch14"
                                        logging.info(f"Falling back to Hugging Face model: {hf_model}")
                        else:
                            # No config.json, check local fallback model
                            logging.warning(f"No config.json for {model_name}. Checking local fallback model.")
                            fallback_model = "clip-vit-large-patch14.safetensors"
                            fallback_path = os.path.join(CLIP_VISION_DIR, fallback_model)
                            if os.path.exists(fallback_path) and fallback_model in CLIP_MODEL_MAPPING:
                                hf_model = CLIP_MODEL_MAPPING[fallback_model]
                                clip_path = fallback_path
                                logging.info(f"Using local model: {clip_path} with hf_model: {hf_model}")
                            else:
                                if "/" in image_encoder_path and not os.path.exists(image_encoder_path):
                                    hf_model = image_encoder_path
                                    logging.info(f"Using image_encoder_path as hf_model: {hf_model}")
                                else:
                                    hf_model = "openai/clip-vit-large-patch14"
                                    logging.warning(f"No local fallback model found. Falling back to Hugging Face model: {hf_model}")

                    # Selecting model class
                    if "siglip" in model_name.lower() or "siglip" in hf_model.lower():
                        logging.info(f"Loading SiglipVisionModel for: {hf_model}")
                        self.image_encoder = SiglipVisionModel.from_pretrained(
                            hf_model, state_dict=None, ignore_mismatched_sizes=True
                        ).to(self.device, dtype=self.dtype)
                    else:
                        logging.info(f"Loading CLIPVisionModel for: {hf_model}")
                        self.image_encoder = CLIPVisionModel.from_pretrained(
                            hf_model, state_dict=None, ignore_mismatched_sizes=True
                        ).to(self.device, dtype=self.dtype)

                    self.image_encoder.load_state_dict(state_dict, strict=False)
                    self.clip_image_processor = AutoProcessor.from_pretrained(hf_model)
                else:
                    self.image_encoder = CLIPVisionModel.from_pretrained(clip_path).to(self.device, dtype=self.dtype)
                    self.clip_image_processor = AutoProcessor.from_pretrained(clip_path)
            else:
                logging.info(f"Loading CLIP Vision from Hugging Face: {image_encoder_path}")
                if "siglip" in image_encoder_path.lower():
                    self.image_encoder = SiglipVisionModel.from_pretrained(image_encoder_path).to(self.device, dtype=self.dtype)
                else:
                    self.image_encoder = CLIPVisionModel.from_pretrained(image_encoder_path).to(self.device, dtype=self.dtype)
                self.clip_image_processor = AutoProcessor.from_pretrained(image_encoder_path)
        except Exception as e:
            logging.error(f"Failed to load CLIP Vision: {e}")
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

        sha256_hash = _compute_sha256(path)
        is_xlab = sha256_hash in KNOWN_XLAB_HASHES
        logging.info(f"SHA256: {sha256_hash}, is_xlab: {is_xlab}")

        # Determine num_tokens
        self.num_tokens = 4
        logging.info(f"Set default num_tokens={self.num_tokens}")

        # Check adapter keys
        ip_adapter_keys = state_dict.get('ip_adapter', {}).keys()
        image_proj_keys = state_dict.get('image_proj', {}).keys()
        logging.info(f"ip_adapter keys: {list(ip_adapter_keys)[:10]}, image_proj keys: {list(image_proj_keys)[:10]}")
                
        # Detect adapter type
        is_instantx = any("to_k_ip" in k.lower() for k in ip_adapter_keys)
        to_k_ip_shape = next((state_dict['ip_adapter'][k].shape[-1] for k in ip_adapter_keys if "to_k_ip" in k.lower()), 0) if is_instantx else 0

        # Adapter configurations
        ADAPTER_CONFIGS = {
            'flux.1': {'num_tokens': 128, 'joint_attention_dim': 4096, 'hidden_size': 3072, 'blocks': [('double_blocks', 19), ('single_blocks', 38)]},
            'sd3.5': {'num_tokens': 64, 'joint_attention_dim': 2560, 'hidden_size': 1536, 'blocks': [('block', 12)]},
            'sdxl': {'num_tokens': 16, 'joint_attention_dim': 2048, 'hidden_size': 1280, 'blocks': [('block', 12)]},
            'sd1.5': {'num_tokens': 4, 'joint_attention_dim': 768, 'hidden_size': None, 'blocks': [('block', 16)]},  # Dynamic hidden_size
            'sd1.5_classic': {'num_tokens': 4, 'joint_attention_dim': 768, 'hidden_size': 640, 'blocks': [('block', 512)]},
            'instantx': {'num_tokens': 128, 'joint_attention_dim': 4096, 'hidden_size': 3072, 'blocks': [('double_blocks', 19), ('single_blocks', 38)]}
        }

        # Detect adapter type
        is_instantx = any("to_k_ip" in k.lower() for k in ip_adapter_keys)
        to_k_ip_shape = next((state_dict['ip_adapter'][k].shape[1] for k in ip_adapter_keys if "to_k_ip" in k.lower()), 0) if is_instantx else 0

        is_instantx_flux = False
        if is_xlab or any(k.startswith(("double_blocks", "single_blocks")) for k in state_dict) or "flux" in self.ip_ckpt.lower():
            self.config = ADAPTER_CONFIGS['flux.1']
            logging.info("Detected Flux.1: num_tokens=128")
        elif is_instantx and to_k_ip_shape == 2560:
            self.config = ADAPTER_CONFIGS['sd3.5']
            logging.info("Detected SD3.5: num_tokens=64")
        elif is_instantx and to_k_ip_shape == 4096 and 'proj.2.weight' in image_proj_keys:
            self.config = {**ADAPTER_CONFIGS['instantx'], 'type': 'instantx'}
            logging.info("Detected InstantX: num_tokens=128")
            is_instantx_flux = True
        elif is_instantx and to_k_ip_shape == 768:
            self.config = ADAPTER_CONFIGS['sd1.5']
            logging.info("Detected SD1.5 (h94/IP-Adapter): num_tokens=4")
        elif "sdxl" in self.ip_ckpt.lower() or any(k.startswith("unet.down_blocks") for k in state_dict):
            self.config = ADAPTER_CONFIGS['sdxl']
            logging.info("Detected SDXL: num_tokens=16")
        else:
            self.config = ADAPTER_CONFIGS['sd1.5_classic']
            logging.info("Detected SD1.5 (classic): num_tokens=4")
            
        self.num_tokens = self.config['num_tokens']
        self.joint_attention_dim = self.config['joint_attention_dim']
        self.hidden_size = self.config.get('hidden_size')
        self.blocks = self.config['blocks']

        # Log state_dict keys
        all_keys = list(state_dict.keys())
        logging.info(f"State dict keys (first 10): {all_keys[:10]}")

        # Find image_proj keys
        image_proj_dict = state_dict.get("image_proj", {})
        if not image_proj_dict:
            # Try ip_adapter for proj keys (XLabs/InstantX)
            image_proj_dict = {k.replace("ip_adapter.", ""): v for k, v in state_dict.items() if k.startswith("ip_adapter.") and "proj" in k.lower()}
            if image_proj_dict:
                logging.info(f"Found ip_adapter proj keys: {list(image_proj_dict.keys())[:5]}")
            else:
                logging.warning("No image_proj, InstantX, or proj keys found")

        # Auto-detect to make sure about the num_tokens   
        if "norm.weight" in image_proj_dict and ("proj.weight" in image_proj_dict or "proj.2.weight" in image_proj_dict):
            norm_size = image_proj_dict["norm.weight"].shape[0]
            if "proj.weight" in image_proj_dict:  # Single-layer projector
                weight_shape = image_proj_dict["proj.weight"].shape
                if weight_shape[0] % norm_size == 0:
                    self.num_tokens = weight_shape[0] // norm_size
                    logging.info(f"Detected num_tokens={self.num_tokens} from proj.weight shape={weight_shape}, norm_size={norm_size}")
            elif "proj.2.weight" in image_proj_dict:  # Multi-layer projector
                weight_shape = image_proj_dict["proj.2.weight"].shape
                if weight_shape[0] % norm_size == 0:
                    self.num_tokens = weight_shape[0] // norm_size
                    logging.info(f"Detected num_tokens={self.num_tokens} from proj.2.weight shape={weight_shape}, norm_size={norm_size}")

        logging.info(f"Final num_tokens after auto-detect check={self.num_tokens}")
		
        # Handle XLabs projector
        if is_xlab:
            xlabs_proj_dict = {}
            for k in all_keys:
                if "proj" in k.lower() and any(p in k.lower() for p in ["weight", "bias"]):
                    if "double_blocks" in k:
                        new_key = k.replace("double_blocks.0.processor.ip_adapter_double_stream_", "")
                        xlabs_proj_dict[new_key] = state_dict[k]
                    else:
                        xlabs_proj_dict[k] = state_dict[k]
            if xlabs_proj_dict:
                logging.info(f"Found XLabs proj keys: {list(xlabs_proj_dict.keys())[:5]}")
                self.image_proj_model = MLPProjModel(
                    cross_attention_dim=self.joint_attention_dim,  # 4096
                    id_embeddings_dim=self.image_encoder.config.hidden_size,  # 1024 for clip-vit-large-patch14
                    num_tokens=self.num_tokens,  # 128
                ).to(self.device, dtype=self.dtype)
                try:
                    proj = {}
                    for key, value in state_dict.items():
                        if key.startswith("ip_adapter_proj_model"):
                            proj[key[len("ip_adapter_proj_model."):]] = value
                    if not proj:
                        logging.error("No ip_adapter_proj_model keys found in state_dict")
                        raise KeyError("No ip_adapter_proj_model keys found")
                    self.image_proj_model.load_state_dict(proj, strict=False)
                    logging.info("Loaded XLabs image_proj successfully")
                except Exception as e:
                    logging.error(f"Failed to load XLabs parameters: {e}")
                    raise
        else:
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
                    id_embeddings_dim=self.image_encoder.config.hidden_size, #1152
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=self.dtype)
                try:
                    self.image_proj_model.load_state_dict(image_proj_dict, strict=False)
                    logging.info("Loaded image_proj successfully")
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
        self.ip_attn_procs = self.init_ip_adapter(state_dict)
        # Create ModuleList for IP-Adapter processors
        if shuffle_weights:
            if is_instantx_flux:
                # For InstantX Flux, use values directly without sorting
                ip_layers = torch.nn.ModuleList(self.ip_attn_procs.values())
            else:
                # Shuffle keys randomly for non-InstantX Flux adapters
                import random
                keys = list(self.ip_attn_procs.keys())
                random.shuffle(keys)
                ip_layers = torch.nn.ModuleList([self.ip_attn_procs[key] for key in keys])
            logging.info("Using shuffled weights for IP-Adapter")
        else:
            # Check if we're dealing with SD1.5 (h94/IP-Adapter) or another adapter
            if self.hidden_size is None:  # SD1.5 (h94/IP-Adapter)
                # Sort keys numerically for SD1.5 (e.g., block_1, block_3, ..., block_31)
                sorted_keys = sorted(self.ip_attn_procs.keys(), key=lambda x: int(x.split('_')[1]))
            else:
                # Use standard sorting for other adapters (Flux.1, InstantX, SD3.5, SDXL)
                sorted_keys = sorted(self.ip_attn_procs.keys())
            ip_layers = torch.nn.ModuleList([self.ip_attn_procs[key] for key in sorted_keys])
            logging.info("Using sorted weights for IP-Adapter")
            
        ip_adapter_dict = state_dict.get('ip_adapter', state_dict)
        logging.info(f"Loading ip_adapter with keys: {list(ip_adapter_dict.keys())[:5]}")
        
        # Adapt state_dict to match our block structure
        new_state_dict = {}
        for key in ip_adapter_dict:
            if not (key.endswith("to_k_ip.weight") or key.endswith("to_v_ip.weight")):
                # Skip keys that are not related to to_k_ip.weight or to_v_ip.weight
                logging.debug(f"Skipping non-attention key: {key}")
                continue

            # Determine key format and extract block index
            parts = key.split('.')
            block_idx = None
            block_name = None

            if parts[0].isdigit():  # SD1.5 format: "1.to_k_ip.weight"
                block_idx = int(parts[0])
                block_name = f"block_{block_idx}"
            elif parts[0] in ["double_blocks", "single_blocks"] and len(parts) > 1 and parts[1].isdigit():  # Flux.1/InstantX format
                block_idx = int(parts[1])
                block_name = f"{parts[0]}.{block_idx}"
            elif parts[0] == "block" and len(parts) > 1 and parts[1].isdigit():  # SD3.5/SDXL format
                block_idx = int(parts[1])
                block_name = f"block_{block_idx}"
            else:
                # Unknown format, skip adaptation
                logging.warning(f"Unknown key format for {key}. Skipping.")
                continue

            # Form new key for ModuleList
            new_key = f"{block_idx}.to_k_ip.weight" if "to_k_ip" in key else f"{block_idx}.to_v_ip.weight"

            # Check if block exists in ip_attn_procs
            if block_name not in self.ip_attn_procs:
                logging.warning(f"Block {block_name} not found in ip_attn_procs. Skipping {key}.")
                continue

            # Check if hidden_size matches
            expected_hidden_size = self.ip_attn_procs[block_name].hidden_size
            actual_hidden_size = ip_adapter_dict[key].shape[0]
            if actual_hidden_size != expected_hidden_size:
                logging.warning(f"Size mismatch for {key}: expected hidden_size={expected_hidden_size}, got {actual_hidden_size}. Skipping.")
                continue

            new_state_dict[new_key] = ip_adapter_dict[key]

    def init_ip_adapter(self, state_dict, weight=1.0, timestep_percent_range=(0.0, 1.0)):
        # Dictionary to store IP-Adapter attention processors
        ip_attn_procs = {}
        
        if self.hidden_size is None:  # SD1.5 (h94/IP-Adapter)
            # Extract hidden_size dynamically from state_dict for each block
            ip_adapter_dict = state_dict.get('ip_adapter', state_dict)
            hidden_sizes = {}
            for key in ip_adapter_dict:
                if key.endswith("to_k_ip.weight"):
                    block_idx = int(key.split('.')[0])
                    hidden_size = ip_adapter_dict[key].shape[0]
                    hidden_sizes[block_idx] = hidden_size
                    logging.info(f"Detected block_{block_idx} with hidden_size={hidden_size}")
            
            # Create processors for each block with the detected hidden_size
            for block_idx, hidden_size in hidden_sizes.items():
                name = f"block_{block_idx}"
                ip_attn_procs[name] = IPAFluxAttnProcessor2_0(
                    hidden_size=hidden_size,
                    cross_attention_dim=self.joint_attention_dim,
                    num_tokens=self.num_tokens,
                    scale=weight,
                    timestep_range=timestep_percent_range
                ).to(self.device, dtype=self.dtype)
                ip_attn_procs[name].weight = weight
                logging.info(f"SD1.5: Created processor for {name} with hidden_size={hidden_size}, weight={weight}")
        else:  # Flux.1, SD3.5, SDXL, InstantX
            # Create processors for other adapters with fixed hidden_size
            for block_prefix, block_count in self.blocks:
                for i in range(block_count):
                    name = f"{block_prefix}.{i}"
                    ip_attn_procs[name] = IPAFluxAttnProcessor2_0(
                        hidden_size=self.hidden_size,
                        cross_attention_dim=self.joint_attention_dim,
                        num_tokens=self.num_tokens,
                        scale=weight,
                        timestep_range=timestep_percent_range
                    ).to(self.device, dtype=self.dtype)
                    logging.info(f"Created processor for {name} with hidden_size={self.hidden_size}")
        
        return ip_attn_procs

    def update_ip_adapter(self, flux_model, weight, timestep_percent_range=(0.0, 1.0)):
        s = flux_model.model_sampling
        percent_to_timestep_function = lambda a: s.percent_to_sigma(a)
        timestep_range = (percent_to_timestep_function(timestep_percent_range[0]), percent_to_timestep_function(timestep_percent_range[1]))
        for name, ip_layer in self.ip_attn_procs.items():
            if self.hidden_size is None:  # SD1.5
                ip_layer.weight = weight
                logging.info(f"SD1.5: Updated IP-Adapter layer {name}: weight={ip_layer.weight}, timestep_range={timestep_range}")
            else:  # Flux
                ip_layer.scale = weight
                logging.info(f"Flux: Updated IP-Adapter layer {name}: scale={ip_layer.scale}, timestep_range={timestep_range}")
            ip_layer.timestep_range = timestep_range

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

        if self.image_proj_model is None:
            logging.info("Using raw clip_image_embeds")
            target_size = self.joint_attention_dim
            if clip_image_embeds.shape[1] != target_size:
                if self.fallback_proj is None:
                    logging.warning(f"Creating fallback projection from {clip_image_embeds.shape[1]} to joint_attention_dim={target_size}")
                    self.fallback_proj = torch.nn.Linear(clip_image_embeds.shape[1], target_size).to(self.device, dtype=self.dtype)
                    torch.nn.init.xavier_uniform_(self.fallback_proj.weight)
                    self.fallback_proj.bias.data.fill_(0)
                clip_image_embeds = self.fallback_proj(clip_image_embeds)
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
                "shuffle_weights": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("IP_ADAPTER_FLUX_INSTANTX",)
    RETURN_NAMES = ("ipadapterFlux",)
    FUNCTION = "load_model"
    CATEGORY = "InstantXNodes"

    def load_model(self, ipadapter, clip_vision, provider, dtype="auto", shuffle_weights="False"):
        # Load IP-Adapter model
        logging.info(f"Loading InstantX IPAdapter Flux model with clip_vision={clip_vision}, dtype={dtype}")
        if dtype == "auto":
            dtype = "float16"
            if provider == "cuda" and torch.cuda.is_bf16_supported():
                dtype = "bfloat16"
        model = InstantXFluxIPAdapterModel(
            image_encoder_path=clip_vision,
            ip_ckpt=ipadapter,
            device=provider,
            dtype=getattr(torch, dtype),
            shuffle_weights=shuffle_weights
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
                "weight": ("FLOAT", {"default": 0.75, "min": -1.0, "max": 5.0, "step": 0.05}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_ipadapter_flux"
    CATEGORY = "InstantXNodes"

    def apply_ipadapter_flux(self, model, ipadapter_flux, image, weight, start_percent, end_percent):
        """
        Apply IP-Adapter to the model for Flux.1/InstantX or SD1.5 architectures.

        Args:
            model: Diffusion model to patch.
            ipadapter_flux: IP-Adapter instance.
            image: Input image tensor.
            weight: Adapter weight.
            start_percent: Start timestep percentage (0 to 1).
            end_percent: End timestep percentage (0 to 1).
        Returns:
            Tuple containing the patched model clone.
        """
        # Convert image to PIL format
        pil_image = Image.fromarray((image.numpy()[0] * 255.0).astype(np.uint8))
        logging.info(f"Converted image to PIL format with size: {pil_image.size}")

        # Update IP-Adapter parameters
        ipadapter_flux.update_ip_adapter(model.model, weight, (start_percent, end_percent))
        logging.info(f"Updated IP-Adapter with weight={weight}, timestep range=({start_percent}, {end_percent})")

        # Get image embeddings
        image_prompt_embeds = ipadapter_flux.get_image_embeds(pil_image=pil_image, clip_image_embeds=None)
        logging.info(f"Generated image embeddings with shape: {image_prompt_embeds.shape}")

        # Apply IP-Adapter to the model
        is_patched = is_model_patched(model.model)
        logging.info(f"Model is already patched: {is_patched}")
        bi = model.clone()
        logging.info(f"Created model clone for patching. Type of bi: {type(bi)}")

        # Compute sigma range for SD1.5
        sigma_start, sigma_end = float("inf"), 0.0  # Default: apply always if percent_to_sigma unavailable
        if hasattr(model.get_model_object("model_sampling"), "percent_to_sigma"):
            try:
                sigma_start = model.get_model_object("model_sampling").percent_to_sigma(start_percent)
                sigma_end = model.get_model_object("model_sampling").percent_to_sigma(end_percent)
                logging.info(f"Computed sigma range for SD1.5: [{sigma_end}, {sigma_start}]")
            except Exception as e:
                logging.warning(f"Failed to compute sigma range: {e}. Using full range.")
        else:
            logging.warning("model_sampling.percent_to_sigma not found. Applying IP-Adapter for full range.")

        FluxUpdateModules(bi, ipadapter_flux.ip_attn_procs, image_prompt_embeds, is_patched, sigma_start=sigma_start, sigma_end=sigma_end)
        logging.info("Applied IP-Adapter to the model clone")
        return (bi,)


NODE_CLASS_MAPPINGS = {
    "IPAdapterFluxLoader": IPAdapterFluxLoader,
    "ApplyIPAdapterFlux": ApplyIPAdapterFlux,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IPAdapterFluxLoader": "Load IPAdapter Flux Model",
    "ApplyIPAdapterFlux": "Apply IPAdapter Flux Model",
}