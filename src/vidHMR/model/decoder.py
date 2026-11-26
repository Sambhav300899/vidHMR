"""The person decoder (tasks 2 and 3 of docs/training_plan.md): the frozen DiT features of one
window and a prompt mask -> per-frame SMPL outputs for the prompted person.

A reduced copy of ACE-Ego-Hand's decoder (MemorySegmentBetasTransformer with the alternating
encoder, external/ACE-Ego-Hand/ace_ego_hand/archs/projector/), with a mask encoder copied from
SAM 2's memory encoder (sam2/modeling/memory_encoder.py; SAM 3 has the same).

Shapes for a batch of B items (L = 21 latents, H x W tokens, C DiT channels, D = 384):

  features (B, L, H, W, C)       -> encoder conditioning (B, L, H*W, D)
  prompt mask (B, 32H, 32W) 0/1  -> mask tokens (B, H*W, D)
  query tokens (B, L, 29, D)     1 person slot token + 24 joint tokens + 4 register tokens per
                                 latent (ACE's alt_query_mode "joint")
  4 alternating layers           -> person slot token (B, L, D), joint tokens (B, L, 24, D)
  joint readout                  each joint token attends to its latent's encoder conditioning:
                                 the attention map is the joint's heatmap -> soft-argmax 2D
                                 (B, L, 24, 2), the attended feature -> root-relative 3D
                                 (B, L, 24, 3)
  linear interpolation in time   -> 81 frames = 4(L - 1) + 1, latent k -> frame 4k
  heads per frame                -> 6D rotations (B, 81, 24, 6), pelvis u, v, log depth (B, 81, 3)
                                    (from the person slot token, as ACE's default "pooled" readout)
  head per window                -> betas (B, 10)
"""

import math

import torch
import torch.nn.functional as F  # noqa: N812 (the usual PyTorch name)
from torch import nn

HIDDEN = 384  # D, the decoder width (ACE config)
HEADS = 8
N_LAYERS = 4
N_REGISTERS = 4
N_JOINTS = 24  # the SMPL joints
RAY_FREQUENCIES = 8
START_DEPTH = 3.4  # meters, the median train pelvis depth (ACE starts hands at 0.5 m)


class LayerNorm2d(nn.Module):
    """SAM's LayerNorm over the channels of each pixel of a (B, C, H, W) map."""

    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + 1e-6)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class ConvNeXtBlock(nn.Module):
    """SAM 2's fuser block (CXBlock): depthwise 7 x 7 conv, LayerNorm, MLP (4x), layer scale
    that starts at 1e-6, residual."""

    def __init__(self):
        super().__init__()
        self.depthwise = nn.Conv2d(HIDDEN, HIDDEN, kernel_size=7, padding=3, groups=HIDDEN)
        self.norm = LayerNorm2d(HIDDEN)
        self.mlp = nn.Sequential(
            nn.Linear(HIDDEN, 4 * HIDDEN), nn.GELU(), nn.Linear(4 * HIDDEN, HIDDEN)
        )
        self.scale = nn.Parameter(1e-6 * torch.ones(HIDDEN))

    def forward(self, x):
        y = self.norm(self.depthwise(x)).permute(0, 2, 3, 1)  # (B, H, W, C)
        y = (self.scale * self.mlp(y)).permute(0, 3, 1, 2)
        return x + y


class MaskEncoder(nn.Module):
    """SAM 2's memory encoder: the prompt mask and the encoder conditioning of the prompt latent
    -> one token per feature token (the mask tokens)."""

    def __init__(self):
        super().__init__()
        layers, channels = [], 1
        for _ in range(5):  # 5 stride-2 convs: one feature token = 32 x 32 mask pixels
            layers += [
                nn.Conv2d(channels, 4 * channels, kernel_size=3, stride=2, padding=1),
                LayerNorm2d(4 * channels),
                nn.GELU(),
            ]
            channels *= 4
        self.downsample = nn.Sequential(*layers, nn.Conv2d(channels, HIDDEN, kernel_size=1))
        self.conditioning_proj = nn.Conv2d(HIDDEN, HIDDEN, kernel_size=1)
        self.fuser = nn.Sequential(ConvNeXtBlock(), ConvNeXtBlock())

    def forward(self, mask, conditioning):
        """mask (B, 32H, 32W) 0/1, conditioning (B, D, H, W) -> (B, D, H, W)."""
        x = self.conditioning_proj(conditioning) + self.downsample(mask[:, None].float())
        return self.fuser(x)


def ray_angles(K, image_size, height: int, width: int):
    """(B, H*W, 2): azimuth and elevation (radians) of the camera ray through the centre of each
    token, for images of image_size (B, 2) pixels (width, height) with intrinsics K (B, 3, 3)."""
    row, column = torch.meshgrid(
        torch.arange(height, device=K.device), torch.arange(width, device=K.device), indexing="ij"
    )
    u = (column.flatten() + 0.5) * image_size[:, :1] / width  # (B, H*W) pixels
    v = (row.flatten() + 0.5) * image_size[:, 1:] / height
    azimuth = torch.atan((u - K[:, 0, 2:3]) / K[:, 0, 0:1])
    elevation = torch.atan((v - K[:, 1, 2:3]) / K[:, 1, 1:2])
    return torch.stack([azimuth, elevation], -1)


class RayEncoding(nn.Module):
    """ACE's ray position encoding: sin and cos of the ray angles at 8 frequencies, then an MLP
    whose last layer starts at zero (so the encoding starts as nothing)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("frequencies", 2.0 ** torch.arange(RAY_FREQUENCIES).float())
        self.mlp = nn.Sequential(
            nn.Linear(4 * RAY_FREQUENCIES, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, HIDDEN)
        )
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, K, image_size, height: int, width: int):
        """-> (B, H*W, D)"""
        angles = ray_angles(K.float(), image_size.float(), height, width)[..., None]
        azimuth, elevation = (angles * self.frequencies).unbind(-2)  # (B, H*W, n) each
        sincos = [azimuth.sin(), azimuth.cos(), elevation.sin(), elevation.cos()]
        return self.mlp(torch.cat(sincos, -1))


def rope_angles(head_dim: int, n_positions: int, device):
    """ACE's temporal RoPE: (n_positions, head_dim) angles, base 10000, interleaved pairs."""
    position = torch.arange(n_positions, device=device, dtype=torch.float32)
    frequencies = 1.0 / 10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    return torch.repeat_interleave(torch.outer(position, frequencies), 2, dim=-1)


def apply_rope(x, angles):
    """Rotate each pair of channels of x (B, heads, N, head_dim) by angles (N, head_dim)."""
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    rotated = torch.stack([-pairs[..., 1], pairs[..., 0]], -1).reshape(x.shape)
    return x * angles.cos() + rotated * angles.sin()


class AlternatingLayer(nn.Module):
    """ACE's alternating round. 1) Read: each latent's query tokens cross-attend to that latent's
    encoder conditioning and the mask tokens (no feed-forward). 2) Exchange: all query tokens of
    the item attend to each other, with RoPE on the latent index, then a feed-forward block."""

    def __init__(self):
        super().__init__()
        self.cross_attention_norm_queries = nn.LayerNorm(HIDDEN)
        self.cross_attention_norm_tokens = nn.LayerNorm(HIDDEN)
        self.cross_attention = nn.MultiheadAttention(HIDDEN, HEADS, batch_first=True)
        self.self_attention_norm = nn.LayerNorm(HIDDEN)
        self.qkv = nn.Linear(HIDDEN, 3 * HIDDEN)
        self.self_attention_out = nn.Linear(HIDDEN, HIDDEN)
        self.ffn_norm = nn.LayerNorm(HIDDEN)
        self.ffn = nn.Sequential(
            nn.Linear(HIDDEN, 4 * HIDDEN), nn.GELU(), nn.Linear(4 * HIDDEN, HIDDEN)
        )

    def forward(self, queries, tokens, rope_angles):
        """queries (B, L, Q, D); tokens (B*L, N, D), read only; rope_angles (L, D/HEADS).
        Returns the queries (B, L, Q, D)."""
        B, L, Q, D = queries.shape
        x = queries.reshape(B * L, Q, D)
        keys = self.cross_attention_norm_tokens(tokens)
        queries_normed = self.cross_attention_norm_queries(x)
        x = x + self.cross_attention(queries_normed, keys, keys, need_weights=False)[0]

        x = x.reshape(B, L * Q, D)
        q, k, v = self.qkv(self.self_attention_norm(x)).reshape(B, L * Q, 3, HEADS, -1).unbind(2)
        angles = rope_angles.repeat_interleave(Q, dim=0)  # a latent's Q tokens share its position
        q = apply_rope(q.transpose(1, 2), angles)
        k = apply_rope(k.transpose(1, 2), angles)
        attended = F.scaled_dot_product_attention(q, k, v.transpose(1, 2))
        x = x + self.self_attention_out(attended.transpose(1, 2).reshape(B, L * Q, D))
        x = x + self.ffn(self.ffn_norm(x))
        return x.reshape(B, L, Q, D)


def head(out: int) -> nn.Sequential:
    """ACE's output head: Linear D -> D/2, GELU, Linear D/2 -> out."""
    return nn.Sequential(nn.Linear(HIDDEN, HIDDEN // 2), nn.GELU(), nn.Linear(HIDDEN // 2, out))


class PersonDecoder(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(channels, HIDDEN), nn.LayerNorm(HIDDEN))
        # ACE's learned 16 x 16 position grid, resized to H x W (so landscape is a stretch).
        self.position_grid = nn.Parameter(torch.randn(1, HIDDEN, 16, 16) * 0.02)
        self.rays = RayEncoding()
        self.mask_encoder = MaskEncoder()
        self.memory = nn.Parameter(torch.randn(1, 1, HIDDEN) * 0.02)  # marks the mask tokens
        self.person_slot = nn.Parameter(torch.randn(1, 1, HIDDEN) * 0.02)
        self.joint_tokens = nn.Parameter(torch.randn(1, N_JOINTS, HIDDEN) * 0.02)
        self.registers = nn.Parameter(torch.randn(1, N_REGISTERS, HIDDEN) * 0.02)
        self.layers = nn.ModuleList(AlternatingLayer() for _ in range(N_LAYERS))
        self.final_norm = nn.LayerNorm(HIDDEN)
        self.frame_fuse = nn.Sequential(nn.LayerNorm(HIDDEN), nn.Linear(HIDDEN, HIDDEN), nn.GELU())

        # ACE's joint readout (memory_encoder.py, query_mode "joint").
        self.readout_norm_queries = nn.LayerNorm(HIDDEN)
        self.readout_norm_tokens = nn.LayerNorm(HIDDEN)
        self.readout = nn.MultiheadAttention(HIDDEN, HEADS, batch_first=True)
        self.joints_relative = nn.Sequential(nn.LayerNorm(HIDDEN), head(3))

        self.global_rotation = head(6)
        self.joint_rotations = head(23 * 6)
        self.pelvis = head(3)  # pelvis u, v in -1..1 and log depth (ACE's camera head)
        self.betas = nn.Sequential(nn.LayerNorm(HIDDEN), head(10))

        # As ACE: the pelvis starts at the image centre and START_DEPTH, the betas at 0.
        nn.init.zeros_(self.pelvis[-1].weight)
        self.pelvis[-1].bias.data = torch.tensor([0.0, 0.0, math.log(START_DEPTH)])
        nn.init.zeros_(self.betas[-1][-1].weight)
        nn.init.zeros_(self.betas[-1][-1].bias)

    def forward(self, features, prompt_mask, prompt_latent, K, image_size):
        """features (B, L, H, W, C), prompt_mask (B, 32H, 32W), prompt_latent (B,), K (B, 3, 3),
        image_size (B, 2) -> rotations_6d (B, F, 24, 6), pelvis_uv_log_depth (B, F, 3),
        betas (B, 10), joints_2d (B, F, 24, 2; 0..1 of the image width and height) and
        joints_relative (B, F, 24, 3; meters, minus joint 0), with F = 4(L - 1) + 1 frames."""
        B, L, H, W, C = features.shape
        D = HIDDEN

        # Encoder conditioning: the DiT tokens in D channels with the position encodings.
        conditioning = self.input_proj(features.reshape(B, L, H * W, C).float())
        position = F.interpolate(
            self.position_grid, size=(H, W), mode="bilinear", align_corners=True
        )
        conditioning = (
            conditioning
            + position.flatten(2).transpose(1, 2)[:, None]
            + self.rays(K, image_size, H, W)[:, None]
        )

        # Mask tokens from the prompt mask and the prompt latent's encoder conditioning.
        prompt_conditioning = (
            conditioning[torch.arange(B), prompt_latent].transpose(1, 2).reshape(B, D, H, W)
        )
        mask_tokens = (
            self.mask_encoder(prompt_mask, prompt_conditioning).flatten(2).transpose(1, 2)
        )
        mask_tokens = mask_tokens + self.memory
        tokens = torch.cat([conditioning, mask_tokens[:, None].expand(B, L, H * W, D)], 2)
        tokens = tokens.reshape(B * L, 2 * H * W, D)

        queries = torch.cat([self.person_slot, self.joint_tokens, self.registers], 1)
        queries = queries[:, None].expand(B, L, -1, -1)
        angles = rope_angles(D // HEADS, L, features.device)
        for layer in self.layers:
            queries = layer(queries, tokens, angles)
        queries = self.final_norm(queries)
        person = queries[:, :, 0]  # (B, L, D)

        # Joint readout: the attention over the H x W grid (mean of the heads) is the heatmap.
        joints = self.readout_norm_queries(queries[:, :, 1 : 1 + N_JOINTS]).flatten(0, 1)
        grid = self.readout_norm_tokens(conditioning.flatten(0, 1))  # (B*L, H*W, D)
        read, heatmap = self.readout(joints, grid, grid, need_weights=True)

        # Frames and heads in float32: bf16 outputs would round the log depth (~9 mm of pelvis).
        with torch.autocast(features.device.type, enabled=False):
            heatmap = heatmap.float().reshape(B, L, N_JOINTS, H, W)
            u = (heatmap.sum(-2) * (torch.arange(W, device=heatmap.device) + 0.5) / W).sum(-1)
            v = (heatmap.sum(-1) * (torch.arange(H, device=heatmap.device) + 0.5) / H).sum(-1)
            joints_relative = self.joints_relative(read.float()).reshape(B, L, N_JOINTS, 3)
            joints_relative = joints_relative - joints_relative[:, :, :1]

            frames = to_frames(person.float())
            frames = frames + self.frame_fuse(frames)
            rotations = torch.cat([self.global_rotation(frames), self.joint_rotations(frames)], -1)
            return (
                rotations.reshape(B, -1, 24, 6),
                self.pelvis(frames),
                self.betas(frames.mean(1)),
                to_frames(torch.stack([u, v], -1)),
                to_frames(joints_relative),
            )


def to_frames(x):
    """(B, L, ...) per latent -> (B, 4(L - 1) + 1, ...) per frame, linear in time: latent k is
    frame 4k (ACE interpolates its latent outputs the same way)."""
    B, L = x.shape[:2]
    flat = x.reshape(B, L, -1).transpose(1, 2)
    frames = F.interpolate(flat, size=4 * (L - 1) + 1, mode="linear", align_corners=True)
    return frames.transpose(1, 2).reshape(B, -1, *x.shape[2:])
