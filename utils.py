import torch
from torch import Tensor
from .flux.layers import DoubleStreamBlockIPA, SingleStreamBlockIPA, CrossAttentionIPA
from comfy.ldm.flux.layers import timestep_embedding as timestep_embedding_flux
from comfy.ldm.modules.diffusionmodules.util import timestep_embedding as timestep_embedding_sd15
from types import MethodType
import hashlib
import logging

def FluxUpdateModules(bi, ip_attn_procs, image_emb=None, is_patched=False, sigma_start=0.0, sigma_end=0.0):
    """
    Update the diffusion model with IP-Adapter processors and image embeddings.
    Supports Flux.1/InstantX (double_blocks, single_blocks) and SD1.5 (input_blocks/middle_block/output_blocks).

    Args:
        bi: Model clone to be patched.
        ip_attn_procs: Dictionary of IP-Adapter processors.
        image_emb: Image embeddings for IP-Adapter.
        is_patched: Whether the model is already patched.
        sigma_start: Start sigma for SD1.5 IP-Adapter range.
        sigma_end: End sigma for SD1.5 IP-Adapter range.
    """
    flux_model = bi.model
    diffusion_model = getattr(flux_model, "diffusion_model", flux_model)

    is_sd15 = False
    if hasattr(diffusion_model, "down_blocks") and hasattr(diffusion_model, "mid_block") and hasattr(diffusion_model, "up_blocks"):
        logging.info("Detected standard SD1.5 (diffusers UNet) architecture")
        is_sd15 = True
        down_blocks = diffusion_model.down_blocks
        mid_block = diffusion_model.mid_block
        up_blocks = diffusion_model.up_blocks
        block_prefix = "down_blocks"
        mid_prefix = "mid_block"
        up_prefix = "up_blocks"
        bi.add_object_patch("diffusion_model.forward_orig", MethodType(forward_orig_ipa, diffusion_model))
    elif hasattr(diffusion_model, "input_blocks") and hasattr(diffusion_model, "middle_block") and hasattr(diffusion_model, "output_blocks"):
        logging.info("Detected ComfyUI SD1.5 (UNetModel) architecture")
        is_sd15 = True
        down_blocks = diffusion_model.input_blocks
        mid_block = diffusion_model.middle_block
        up_blocks = diffusion_model.output_blocks
        block_prefix = "input_blocks"
        mid_prefix = "middle_block"
        up_prefix = "output_blocks"
        bi.add_object_patch("diffusion_model.forward", MethodType(forward_orig_ipa, diffusion_model))
    elif any(k.startswith("block_") for k in ip_attn_procs):
        logging.warning("Assuming SD1.5 architecture based on ip_attn_procs keys")
        is_sd15 = True
        down_blocks = getattr(diffusion_model, "input_blocks", None)
        mid_block = getattr(diffusion_model, "middle_block", None)
        up_blocks = getattr(diffusion_model, "output_blocks", None)
        block_prefix = "input_blocks"
        mid_prefix = "middle_block"
        up_prefix = "output_blocks"
        if not (down_blocks and mid_block and up_blocks):
            raise ValueError("Could not identify SD1.5 blocks")
        bi.add_object_patch("diffusion_model.forward", MethodType(forward_orig_ipa, diffusion_model))
    else:
        logging.info("Detected Flux.1/InstantX architecture")
        double_blocks = diffusion_model.double_blocks
        single_blocks = diffusion_model.single_blocks
        block_prefix = "double_blocks"
        single_prefix = "single_blocks"

    if is_sd15:
        block_idx = 0
        # Log all blocks with attn2 for debugging
        logging.debug("Scanning blocks for attn2 modules...")
        for block_type, blocks, prefix in [
            ("input", down_blocks, block_prefix),
            ("middle", [mid_block] if mid_block else [], mid_prefix),
            ("output", up_blocks, up_prefix)
        ]:
            for i, block in enumerate(blocks):
                if isinstance(block, (list, tuple)):
                    block_modules = block
                else:
                    block_modules = [block]
                for module in block_modules:
                    for name, sub_module in module.named_modules():
                        if "transformer_blocks" in name and hasattr(sub_module, "attn2") and sub_module.attn2 is not None:
                            logging.info(f"Found attn2 in {prefix}.{i}.{name}, block_idx={block_idx}")
                            block_idx += 1
        block_idx = 0

        # Patch CrossAttention (attn2) in BasicTransformerBlock
        for block_type, blocks, prefix in [
            ("input", down_blocks, block_prefix),
            ("middle", [mid_block] if mid_block else [], mid_prefix),
            ("output", up_blocks, up_prefix)
        ]:
            for i, block in enumerate(blocks):
                if isinstance(block, (list, tuple)):
                    block_modules = block
                else:
                    block_modules = [block]
                for module in block_modules:
                    for name, sub_module in module.named_modules():
                        if "transformer_blocks" in name and hasattr(sub_module, "attn2") and sub_module.attn2 is not None:
                            patch_name = f"block_{block_idx}"
                            if patch_name not in ip_attn_procs:
                                logging.warning(f"No IP-Adapter processor for {patch_name}. Skipping.")
                                block_idx += 1
                                continue
                            # Build path to attn2
                            path = f"diffusion_model.{prefix}.{i}{'.' + name if name else ''}.attn2"
                            try:
                                maybe_patched_layer = bi.get_model_object(path)
                            except AttributeError as e:
                                logging.error(f"Failed to access {path}: {e}")
                                block_idx += 1
                                continue
                            procs = [ip_attn_procs[patch_name]]
                            embs = [image_emb]
                            new_layer = CrossAttentionIPA(
                                attn_module=maybe_patched_layer,
                                ip_adapter=procs,
                                image_emb=embs,
                                sigma_start=sigma_start,
                                sigma_end=sigma_end
                            )
                            bi.add_object_patch(path, new_layer)
                            logging.info(f"Patched CrossAttention module: {path} with {patch_name}")
                            block_idx += 1

    elif hasattr(diffusion_model, "double_blocks") or hasattr(diffusion_model, "single_blocks"):
        logging.info("Detected Flux.1/InstantX model architecture")
        bi.add_object_patch("diffusion_model.forward_orig", MethodType(forward_orig_ipa, diffusion_model))
        for i, original in enumerate(diffusion_model.double_blocks):
            patch_name = f"double_blocks.{i}"
            if patch_name not in ip_attn_procs:
                logging.warning(f"No IP-Adapter processor for {patch_name}. Skipping.")
                continue
            maybe_patched_layer = bi.get_model_object(f"diffusion_model.{patch_name}")
            procs = [ip_attn_procs[patch_name]]
            embs = [image_emb]
            if isinstance(maybe_patched_layer, DoubleStreamBlockIPA):
                procs = maybe_patched_layer.ip_adapter + procs
                embs = maybe_patched_layer.image_emb + embs
            new_layer = DoubleStreamBlockIPA(original, procs, embs)
            bi.add_object_patch(f"diffusion_model.{patch_name}", new_layer)
            logging.info(f"Patched DoubleStreamBlock: {patch_name}")

        for i, original in enumerate(diffusion_model.single_blocks):
            patch_name = f"single_blocks.{i}"
            if patch_name not in ip_attn_procs:
                logging.warning(f"No IP-Adapter processor for {patch_name}. Skipping.")
                continue
            maybe_patched_layer = bi.get_model_object(f"diffusion_model.{patch_name}")
            procs = [ip_attn_procs[patch_name]]
            embs = [image_emb]
            if isinstance(maybe_patched_layer, SingleStreamBlockIPA):
                procs = maybe_patched_layer.ip_adapter + procs
                embs = maybe_patched_layer.image_emb + embs
            new_layer = SingleStreamBlockIPA(original, procs, embs)
            bi.add_object_patch(f"diffusion_model.{patch_name}", new_layer)
            logging.info(f"Patched SingleStreamBlock: {patch_name}")

    else:
        raise ValueError(
            "Unsupported model architecture: Model must have either double_blocks/single_blocks (Flux.1/InstantX) "
            "or down_blocks/mid_block/up_blocks or input_blocks/middle_block/output_blocks (SD1.5)."
        )
        
def is_model_patched(model):
    """
    Check if the model has been patched with IP-Adapter layers.
    Supports both Flux.1/InstantX (DoubleStreamBlockIPA, SingleStreamBlockIPA) and SD1.5 (CrossAttentionIPA).
    
    Args:
        model: The model to check.
    
    Returns:
        bool: True if the model contains IP-Adapter patched layers, False otherwise.
    """
    def test(mod):
        # Check for Flux.1/InstantX IP-Adapter layers
        if isinstance(mod, (DoubleStreamBlockIPA, SingleStreamBlockIPA)):
            return True
        # Check for SD1.5 IP-Adapter layers
        if isinstance(mod, CrossAttentionIPA):
            return True
        # Recursively check children
        for child in mod.children():
            if test(child):
                return True
        return False
    
    result = test(model)
    logging.debug(f"Model patched status: {result}")
    return result

def apply_control(h, control, block_type):
    # Original apply_control implementation (copy if different in your codebase)
    if control is not None and block_type in control:
        for item in control[block_type]:
            if item is not None:
                h = h + item
    return h

def forward_orig_ipa(
    self,
    img: torch.Tensor,
    img_ids: torch.Tensor = None,  # Optional for Flux.1/InstantX
    txt: torch.Tensor = None,      # For Flux.1/InstantX
    txt_ids: torch.Tensor = None,  # For Flux.1/InstantX
    timesteps: torch.Tensor = None,
    y: torch.Tensor = None,
    guidance: torch.Tensor | None = None,
    control=None,
    transformer_options={},
    attn_mask: torch.Tensor = None,
    context: torch.Tensor = None,  # For SD1.5 compatibility
) -> torch.Tensor:
    """
    Forward pass for IP-Adapter with ComfyUI UNetModel (SD1.5) or Flux.1/InstantX.
    Handles 'context' for SD1.5 and 'txt'/'txt_ids' for Flux.1.
    Uses sigmas for SD1.5 and timesteps for Flux.1 for IP-Adapter range control.
    """
    patches_replace = transformer_options.get("patches_replace", {})

    if hasattr(self, "double_blocks") and hasattr(self, "single_blocks"):
        # Flux.1/InstantX forward pass (unchanged)
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")
        if timesteps is None:
            raise ValueError("timesteps must be provided for Flux.1 architecture")

        img = self.img_in(img)
        vec = self.time_in(timestep_embedding_flux(timesteps, 256).to(img.dtype))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding_flux(guidance, 256).to(img.dtype))

        vec = vec + self.vector_in(y[:, :self.params.vec_in_dim])
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        blocks_replace = patches_replace.get("dit", {})
        for i, block in enumerate(self.double_blocks):
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    if isinstance(block, DoubleStreamBlockIPA):
                        out["img"], out["txt"] = block(img=args["img"], txt=args["txt"], vec=args["vec"], pe=args["pe"], t=args["timesteps"], attn_mask=args.get("attn_mask"))
                    else:
                        out["img"], out["txt"] = block(img=args["img"], txt=args["txt"], vec=args["vec"], pe=args["pe"], attn_mask=args.get("attn_mask"))
                    return out
                out = blocks_replace[("double_block", i)]({"img": img, "txt": txt, "vec": vec, "pe": pe, "timesteps": timesteps, "attn_mask": attn_mask}, {"original_block": block_wrap, "transformer_options": transformer_options})
                txt = out["txt"]
                img = out["img"]
            else:
                if isinstance(block, DoubleStreamBlockIPA):
                    img, txt = block(img=img, txt=txt, vec=vec, pe=pe, t=timesteps, attn_mask=attn_mask)
                else:
                    img, txt = block(img=img, txt=txt, vec=vec, pe=pe, attn_mask=attn_mask)

            if control is not None:
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img += add

        img = torch.cat((txt, img), 1)

        for i, block in enumerate(self.single_blocks):
            if ("single_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    if isinstance(block, SingleStreamBlockIPA):
                        out["img"] = block(args["img"], vec=args["vec"], pe=args["pe"], t=args["timesteps"], attn_mask=args.get("attn_mask"))
                    else:
                        out["img"] = block(args["img"], vec=args["vec"], pe=args["pe"], attn_mask=args.get("attn_mask"))
                    return out
                out = blocks_replace[("single_block", i)]({"img": img, "vec": vec, "pe": pe, "timesteps": timesteps, "attn_mask": attn_mask}, {"original_block": block_wrap, "transformer_options": transformer_options})
                img = out["img"]
            else:
                if isinstance(block, SingleStreamBlockIPA):
                    img = block(img, vec=vec, pe=pe, t=timesteps, attn_mask=attn_mask)
                else:
                    img = block(img, vec=vec, pe=pe, attn_mask=attn_mask)

            if control is not None:
                control_o = control.get("output")
                if i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        img[:, txt.shape[1]:, ...] += add

        img = img[:, txt.shape[1]:, ...]
        img = self.final_layer(img, vec)
        return img

    elif hasattr(self, "input_blocks") and hasattr(self, "middle_block") and hasattr(self, "output_blocks"):
        # ComfyUI SD1.5 (UNetModel) forward pass
        if context is None:
            raise ValueError("context must be provided for SD1.5 architecture")
        # Use sigmas instead of timesteps for SD1.5
        sigmas = transformer_options.get("sigmas", None)
        if sigmas is None:
            raise ValueError("sigmas must be provided for SD1.5 architecture")
        logging.info(f"SD1.5 forward: sigma shape={sigmas.shape}, img shape={img.shape}, context shape={context.shape if context is not None else None}")

        #Adding sigmas to extra_options
        transformer_options = transformer_options.copy()
        transformer_options["extra_options"] = transformer_options.get("extra_options", {})
        transformer_options["extra_options"]["sigmas"] = sigmas

        transformer_options["original_shape"] = list(img.shape)
        transformer_options["transformer_index"] = 0
        transformer_patches = transformer_options.get("patches", {})

        num_video_frames = transformer_options.get("num_video_frames", None)
        image_only_indicator = transformer_options.get("image_only_indicator", None)
        time_context = transformer_options.get("time_context", None)

        assert (y is not None) == (self.num_classes is not None), "y must be provided if model is class-conditional"
        hs = []
        # Ensure sigmas has shape [batch_size]
        if sigmas.shape != torch.Size([img.shape[0]]):
            sigmas = sigmas.expand(img.shape[0]).to(device=img.device, dtype=img.dtype)
        logging.info(f"Using sigmas with shape: {sigmas.shape}")
        t_emb = timestep_embedding_sd15(sigmas, self.model_channels, repeat_only=False).to(img.dtype)
        emb = self.time_embed(t_emb)

        if "emb_patch" in transformer_patches:
            for p in transformer_patches["emb_patch"]:
                emb = p(emb, self.model_channels, transformer_options)

        if self.num_classes is not None:
            assert y.shape[0] == img.shape[0]
            emb = emb + self.label_emb(y)

        h = img
        for id, module in enumerate(self.input_blocks):
            transformer_options["block"] = ("input", id)
            h = module(h, emb, context, transformer_options, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)
            h = apply_control(h, control, 'input')
            if "input_block_patch" in transformer_patches:
                for p in transformer_patches["input_block_patch"]:
                    h = p(h, transformer_options)
            hs.append(h)
            if "input_block_patch_after_skip" in transformer_patches:
                for p in transformer_patches["input_block_patch_after_skip"]:
                    h = p(h, transformer_options)

        transformer_options["block"] = ("middle", 0)
        if self.middle_block is not None:
            h = self.middle_block(h, emb, context, transformer_options, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)
        h = apply_control(h, control, 'middle')

        for id, module in enumerate(self.output_blocks):
            transformer_options["block"] = ("output", id)
            hsp = hs.pop()
            hsp = apply_control(hsp, control, 'output')
            if "output_block_patch" in transformer_patches:
                for p in transformer_patches["output_block_patch"]:
                    h, hsp = p(h, hsp, transformer_options)
            h = torch.cat([h, hsp], dim=1)
            del hsp
            output_shape = hs[-1].shape if hs else None
            h = module(h, emb, context, transformer_options, output_shape, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)

        img = h.type(img.dtype)
        if self.predict_codebook_ids:
            img = self.id_predictor(img)
        else:
            img = self.out(img)
        return img

    else:
        raise ValueError(
            "Unsupported model architecture: Model must have either double_blocks/single_blocks (Flux.1/InstantX) "
            "or input_blocks/middle_block/output_blocks (SD1.5)."
        )

def _compute_sha256(file_path):
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except Exception as e:
        logging.warning(f"Failed to compute SHA256 for {file_path}: {e}")
        return None