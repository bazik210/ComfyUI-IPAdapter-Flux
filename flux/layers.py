import torch
from torch import Tensor, nn

from .math import attention
from comfy.ldm.flux.layers import DoubleStreamBlock, SingleStreamBlock
from ..attention_processor import IPAFluxAttnProcessor2_0
from comfy.ldm.modules.attention import optimized_attention
from typing import List, Optional
import comfy.model_management
import logging

class DoubleStreamBlockIPA(nn.Module):
    def __init__(self, original_block: DoubleStreamBlock, ip_adapter: list[IPAFluxAttnProcessor2_0], image_emb):
        super().__init__()

        mlp_hidden_dim  = original_block.img_mlp[0].out_features
        mlp_ratio = mlp_hidden_dim / original_block.hidden_size
        mlp_hidden_dim = int(original_block.hidden_size * mlp_ratio)
        self.num_heads = original_block.num_heads
        self.hidden_size = original_block.hidden_size
        self.img_mod = original_block.img_mod
        self.img_norm1 = original_block.img_norm1
        self.img_attn = original_block.img_attn

        self.img_norm2 = original_block.img_norm2
        self.img_mlp = original_block.img_mlp

        self.txt_mod = original_block.txt_mod
        self.txt_norm1 = original_block.txt_norm1
        self.txt_attn = original_block.txt_attn

        self.txt_norm2 = original_block.txt_norm2
        self.txt_mlp = original_block.txt_mlp
        self.flipped_img_txt = original_block.flipped_img_txt

        self.ip_adapter = ip_adapter
        self.image_emb = image_emb
        self.device = comfy.model_management.get_torch_device()

    def add_adapter(self, ip_adapter: IPAFluxAttnProcessor2_0, image_emb):
        self.ip_adapter.append(ip_adapter)
        self.image_emb.append(image_emb)
    
    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, t: Tensor, attn_mask=None):
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = img_qkv.view(img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = txt_qkv.view(txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        if self.flipped_img_txt:
             # run actual attention
            attn = attention(torch.cat((img_q, txt_q), dim=2),
                             torch.cat((img_k, txt_k), dim=2),
                             torch.cat((img_v, txt_v), dim=2),
                             pe=pe, mask=attn_mask)

            img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1]:]           
        else:
            # run actual attention
            attn = attention(torch.cat((txt_q, img_q), dim=2),
                            torch.cat((txt_k, img_k), dim=2),
                            torch.cat((txt_v, img_v), dim=2),
                            pe=pe, mask=attn_mask)
            
            txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        for adapter, image in zip(self.ip_adapter, self.image_emb):
            # this does a separate attention for each adapter
            ip_hidden_states = adapter(self.num_heads, img_q, image, t)
            if ip_hidden_states is not None:
                ip_hidden_states = ip_hidden_states.to(self.device)
                img_attn = img_attn + ip_hidden_states

        # calculate the img bloks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt += txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt += txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)

        if txt.dtype == torch.float16:
            txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)

        return img, txt

class SingleStreamBlockIPA(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(self, original_block: SingleStreamBlock, ip_adapter: list[IPAFluxAttnProcessor2_0], image_emb):
        super().__init__()
        self.hidden_dim = original_block.hidden_size
        self.num_heads = original_block.num_heads
        self.scale = original_block.scale

        self.mlp_hidden_dim = original_block.mlp_hidden_dim
        # qkv and mlp_in
        self.linear1 = original_block.linear1
        # proj and mlp_out
        self.linear2 = original_block.linear2

        self.norm = original_block.norm

        self.hidden_size = original_block.hidden_size
        self.pre_norm = original_block.pre_norm

        self.mlp_act = original_block.mlp_act
        self.modulation = original_block.modulation

        self.ip_adapter = ip_adapter
        self.image_emb = image_emb
        self.device = comfy.model_management.get_torch_device()

    def add_adapter(self, ip_adapter: IPAFluxAttnProcessor2_0, image_emb):
        self.ip_adapter.append(ip_adapter)
        self.image_emb.append(image_emb)

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor, t:Tensor, attn_mask=None) -> Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k = self.norm(q, k, v)

        # compute attention
        attn = attention(q, k, v, pe=pe, mask=attn_mask)

        for adapter, image in zip(self.ip_adapter, self.image_emb):
            # this does a separate attention for each adapter
            # maybe we want a single joint attention call for all adapters?
            ip_hidden_states = adapter(self.num_heads, q, image, t)
            if ip_hidden_states is not None:
                ip_hidden_states = ip_hidden_states.to(self.device)
                attn = attn + ip_hidden_states

        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        x += mod.gate * output
        if x.dtype == torch.float16:
            x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
        return x

class CrossAttentionIPA(nn.Module):
    """IP-Adapter attention processor for SD1.5, compatible with ComfyUI CrossAttention."""
    
    def __init__(
        self,
        attn_module: nn.Module,
        ip_adapter: List[nn.Module],
        image_emb: List[Optional[torch.Tensor]],
        sigma_start: float = 999999999.9,
        sigma_end: float = 0.0
    ):
        super().__init__()
        # Store original attention module
        self.attn_module = attn_module
        # IP-Adapter processors (to_k_ip, to_v_ip, weight)
        self.ip_adapter = nn.ModuleList(ip_adapter)
        # Image embeddings
        self.image_emb = image_emb
        # Sigma range for SD1.5
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        # Log initialization
        logging.info(
            f"Initialized CrossAttentionIPA: sigma_start={sigma_start}, sigma_end={sigma_end}, "
            f"ip_adapter_count={len(ip_adapter)}, image_emb_shape={[e.shape if e is not None else None for e in image_emb]}"
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        transformer_options: Optional[dict] = None
    ) -> torch.Tensor:
        """Forward pass with IP-Adapter integration."""
        transformer_options = transformer_options or {}
        extra_options = transformer_options.get("extra_options", {})
        sigma = extra_options.get("sigmas", None)

        # Extract sigma value
        if sigma is not None and sigma.numel() > 0:
            sigma = sigma[0].item() if sigma.dim() > 0 else sigma.item()
            logging.debug(
                f"CrossAttentionIPA called: sigma={sigma}, sigma_range=[{self.sigma_end}, {self.sigma_start}], "
                f"hidden_states_shape={x.shape}, encoder_hidden_states_shape={context.shape if context is not None else None}"
            )
        else:
            sigma = None
            logging.debug(
                f"CrossAttentionIPA called: sigma=None, hidden_states_shape={x.shape}, "
                f"encoder_hidden_states_shape={context.shape if context is not None else None}"
            )

        # Compute standard attention
        q = self.attn_module.to_q(x)
        context = context if context is not None else x
        k = self.attn_module.to_k(context)
        v = self.attn_module.to_v(context if value is None else value)

        # Apply IP-Adapter if sigma is in range
        if (
            sigma is not None
            and self.sigma_end <= sigma <= self.sigma_start
            and len(self.ip_adapter) > 0
            and any(emb is not None for emb in self.image_emb)
        ):
            logging.debug(f"Applying IP-Adapter at sigma {sigma}")
            for ip_proc, img_emb in zip(self.ip_adapter, self.image_emb):
                if img_emb is None:
                    continue
                # Get IP key and value
                to_k_ip = getattr(ip_proc, "to_k_ip", None)
                to_v_ip = getattr(ip_proc, "to_v_ip", None)
                if to_k_ip is None or to_v_ip is None:
                    continue

                ip_key = to_k_ip(img_emb)
                ip_value = to_v_ip(img_emb)

                # Log IP-Adapter details
                weight = getattr(ip_proc, "weight", 1.0)
                logging.debug(
                    f"IP-Adapter weight={weight}, ip_key shape={ip_key.shape}, "
                    f"min={ip_key.min().item()}, max={ip_key.max().item()}"
                )

                # Concatenate IP key and value
                k = torch.cat([k, ip_key], dim=1)
                v = torch.cat([v, ip_value], dim=1)

        # Compute attention
        out = optimized_attention(
            q, k, v, self.attn_module.heads,
            mask=mask, attn_precision=self.attn_module.attn_precision
        )
        out = self.attn_module.to_out(out)

        return out