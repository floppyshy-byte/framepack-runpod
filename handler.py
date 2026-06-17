"""
RunPod Serverless Handler for FramePack F1 (Image-to-Video).

Accepts a base64-encoded image + prompt, generates a video, and returns
the result as a base64-encoded MP4.

Input schema:
{
    "input": {
        "image": "<base64 encoded image>",
        "prompt": "a person waving hello",
        "negative_prompt": "",           // optional
        "seed": 31337,                   // optional
        "total_second_length": 5,        // optional, 1-120
        "steps": 25,                     // optional, 1-100
        "gs": 10.0,                      // optional, distilled CFG scale
        "gpu_memory_preservation": 6,    // optional, GB to reserve
        "use_teacache": true,            // optional
        "mp4_crf": 16                    // optional, 0=lossless
    }
}

Output schema:
{
    "video": "<base64 encoded mp4>",
    "seed": 31337,
    "duration_seconds": 5.0,
    "total_frames": 141
}
"""

import os
import sys
import base64
import tempfile
import traceback
from runpod.serverless.utils import rp_upload

import numpy as np
import torch
import einops
from PIL import Image
from io import BytesIO

# ---------------------------------------------------------------------------
# Model directory resolution (same logic as demo_gradio_f1.py)
# ---------------------------------------------------------------------------

def resolve_model_dir():
    """Resolve model directory from env var or RunPod HF cache."""
    explicit = os.environ.get('MODEL_DIR')
    if explicit:
        return os.path.abspath(explicit)

    hf_cache = os.environ.get('HF_HOME', '/runpod-volume/huggingface-cache')
    cache_repo_dir = os.path.join(hf_cache, 'hub', 'models--floppyshy--framepack-runpod-cache')
    snapshots_dir = os.path.join(cache_repo_dir, 'snapshots')

    if os.path.isdir(snapshots_dir):
        hashes = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
        if hashes:
            resolved = os.path.join(snapshots_dir, hashes[0])
            print(f'Found RunPod model cache: {resolved}')
            return resolved

    return os.path.abspath('./models')


# ---------------------------------------------------------------------------
# Load models at import time (cold start). RunPod keeps the container warm
# between invocations so these stay in memory.
# ---------------------------------------------------------------------------

print('Loading models...')
model_dir = resolve_model_dir()
print(f'Model directory: {model_dir}')

from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from transformers import SiglipImageProcessor, SiglipVisionModel

from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import (
    cpu, gpu, get_cuda_free_memory_gb,
    move_model_to_device_with_memory_preservation,
    offload_model_from_device_for_memory_preservation,
    fake_diffusers_current_device, DynamicSwapInstaller,
    unload_complete_models, load_model_as_complete,
)
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60

print(f'Free VRAM {free_mem_gb} GB')
print(f'High-VRAM Mode: {high_vram}')

text_encoder = LlamaModel.from_pretrained(os.path.join(model_dir, 'HunyuanVideo', 'text_encoder'), torch_dtype=torch.float16, local_files_only=True).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained(os.path.join(model_dir, 'HunyuanVideo', 'text_encoder_2'), torch_dtype=torch.float16, local_files_only=True).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained(os.path.join(model_dir, 'HunyuanVideo', 'tokenizer'), local_files_only=True)
tokenizer_2 = CLIPTokenizer.from_pretrained(os.path.join(model_dir, 'HunyuanVideo', 'tokenizer_2'), local_files_only=True)
vae = AutoencoderKLHunyuanVideo.from_pretrained(os.path.join(model_dir, 'HunyuanVideo', 'vae'), torch_dtype=torch.float16, local_files_only=True).cpu()

feature_extractor = SiglipImageProcessor.from_pretrained(os.path.join(model_dir, 'flux_redux_bfl', 'feature_extractor'), local_files_only=True)
image_encoder = SiglipVisionModel.from_pretrained(os.path.join(model_dir, 'flux_redux_bfl', 'image_encoder'), torch_dtype=torch.float16, local_files_only=True).cpu()

transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(os.path.join(model_dir, 'FramePack_F1_I2V_HY'), torch_dtype=torch.bfloat16, local_files_only=True).cpu()

vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()
transformer.eval()

if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()

transformer.high_quality_fp32_output_for_inference = True
transformer.to(dtype=torch.bfloat16)
vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)
transformer.requires_grad_(False)

if not high_vram:
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    text_encoder.to(gpu)
    text_encoder_2.to(gpu)
    image_encoder.to(gpu)
    vae.to(gpu)
    transformer.to(gpu)

print('Models loaded successfully.')


# ---------------------------------------------------------------------------
# Generation logic (adapted from demo_gradio_f1.py worker)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_video(
    input_image_np: np.ndarray,
    prompt: str,
    negative_prompt: str = "",
    seed: int = 31337,
    total_second_length: float = 5.0,
    latent_window_size: int = 9,
    steps: int = 25,
    cfg: float = 1.0,
    gs: float = 10.0,
    rs: float = 0.0,
    gpu_memory_preservation: float = 6.0,
    use_teacache: bool = False,
    mp4_crf: int = 16,
) -> str:
    """
    Generate a video from an input image and prompt.
    Returns the path to the output MP4 file.
    """
    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    job_id = generate_timestamp()
    outputs_folder = tempfile.mkdtemp(prefix='framepack_')

    try:
        # Clean GPU
        if not high_vram:
            unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)

        # Text encoding
        print('Text encoding...')
        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)
            load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(negative_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # Process input image
        print('Image processing...')
        H, W, C = input_image_np.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image_np, target_width=width, target_height=height)

        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # VAE encoding
        print('VAE encoding...')
        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        start_latent = vae_encode(input_image_pt, vae)

        # CLIP Vision
        print('CLIP Vision encoding...')
        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # Dtype casting
        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # Sampling
        print('Start sampling...')
        rnd = torch.Generator("cpu").manual_seed(seed)

        history_latents = torch.zeros(size=(1, 16, 16 + 2 + 1, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None

        history_latents = torch.cat([history_latents, start_latent.to(history_latents)], dim=2)
        total_generated_latent_frames = 1

        output_filename = None

        for section_index in range(total_latent_sections):
            print(f'section_index = {section_index}, total_latent_sections = {total_latent_sections}')

            if not high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            def callback(d):
                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                print(f'  Sampling {current_step}/{steps} ({percentage}%)')
                return

            indices = torch.arange(0, sum([1, 16, 2, 1, latent_window_size])).unsqueeze(0)
            clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = indices.split([1, 16, 2, 1, latent_window_size], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)

            clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[:, :, -sum([16, 2, 1]):, :, :].split([16, 2, 1], dim=2)
            clean_latents = torch.cat([start_latent.to(history_latents), clean_latents_1x], dim=2)

            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=latent_window_size * 4 - 3,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=gpu,
                dtype=torch.bfloat16,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([history_latents, generated_latents.to(history_latents)], dim=2)

            if not high_vram:
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)

            real_history_latents = history_latents[:, :, -total_generated_latent_frames:, :, :]

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                section_latent_frames = latent_window_size * 2
                overlapped_frames = latent_window_size * 4 - 3

                current_pixels = vae_decode(real_history_latents[:, :, -section_latent_frames:], vae).cpu()
                history_pixels = soft_append_bcthw(history_pixels, current_pixels, overlapped_frames)

            if not high_vram:
                unload_complete_models()

            output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')
            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)
            print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')

    except Exception:
        traceback.print_exc()
        if not high_vram:
            unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
        raise

    total_frames = int(max(0, total_generated_latent_frames * 4 - 3))
    print(f'Generation complete: {total_frames} frames, {total_frames / 30:.2f}s')
    return output_filename, total_frames


# ---------------------------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------------------------

import runpod
import secrets
import json

# AES-256-GCM encryption — enabled when COMFY_ENCRYPTION_KEY is set (64 hex chars)
_ENCRYPTION_KEY: bytes | None = None
_RAW_KEY = os.getenv("COMFY_ENCRYPTION_KEY", "")
if _RAW_KEY:
    _key_bytes = bytes.fromhex(_RAW_KEY)
    if len(_key_bytes) != 32:
        raise RuntimeError(
            "COMFY_ENCRYPTION_KEY must be 64 hex characters (32 bytes)"
        )
    _ENCRYPTION_KEY = _key_bytes


def _aes_decrypt(encoded: str) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    data = base64.b64decode(encoded)
    nonce, ciphertext = data[:12], data[12:]
    return AESGCM(_ENCRYPTION_KEY).decrypt(nonce, ciphertext, None)


def _aes_encrypt_bytes(plaintext: bytes) -> bytes:
    """Encrypt plaintext bytes and return raw nonce + ciphertext + tag."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    nonce = secrets.token_bytes(12)
    ciphertext = AESGCM(_ENCRYPTION_KEY).encrypt(nonce, plaintext, None)
    return nonce + ciphertext


def _s3_configured() -> bool:
    """Return True if the base image's S3 uploader is configured."""
    return bool(os.environ.get("BUCKET_ENDPOINT_URL"))


def _upload_to_s3(data: bytes, filename: str, job_id: str) -> str:
    """Upload bytes via the base image's S3 helper and return the URL."""
    suffix = os.path.splitext(filename)[1] or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return rp_upload.upload_image(job_id, tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def decode_image(image_b64: str) -> np.ndarray:
    """Decode a base64-encoded image string to a numpy array (H, W, C) uint8 RGB."""
    image_bytes = base64.b64decode(image_b64)
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    return np.array(image)


def handler(job):
    """RunPod serverless handler entry point."""
    job_input = job['input']

    # Encryption mode: triggered by "encryption": true or "encrypted" field
    encryption_enabled = (
        job_input.get("encryption") is True or "encrypted" in job_input
    )
    was_fully_encrypted = "encrypted" in job_input

    if encryption_enabled and not _ENCRYPTION_KEY:
        return {"error": "Encryption is enabled but COMFY_ENCRYPTION_KEY is not set"}

    # Decrypt full payload if it arrived in the "encrypted" field
    if was_fully_encrypted:
        try:
            job_input = json.loads(_aes_decrypt(job_input["encrypted"]))
        except Exception as exc:
            return {"error": f"Failed to decrypt job input: {exc}"}

    # --- Validate required fields ---
    image_b64 = job_input.get('image')
    if not image_b64:
        return {'error': 'Missing required field: image (base64-encoded)'}

    prompt = job_input.get('prompt', '')

    # --- Parse optional parameters with defaults ---
    negative_prompt = job_input.get('negative_prompt', job_input.get('n_prompt', ''))
    seed = int(job_input.get('seed', 31337))
    total_second_length = float(job_input.get('total_second_length', 5.0))
    steps = int(job_input.get('steps', 25))
    gs = float(job_input.get('gs', 10.0))
    gpu_memory_preservation = float(job_input.get('gpu_memory_preservation', 6.0))
    use_teacache = bool(job_input.get('use_teacache', False))
    mp4_crf = int(job_input.get('mp4_crf', 16))

    latent_window_size = int(job_input.get('latent_window_size', 9))
    cfg = float(job_input.get('cfg', 1.0))
    rs = float(job_input.get('rs', 0.0))

    # --- Clamp values ---
    total_second_length = max(1.0, min(120.0, total_second_length))
    steps = max(1, min(100, steps))
    gs = max(1.0, min(32.0, gs))
    gpu_memory_preservation = max(6.0, min(128.0, gpu_memory_preservation))
    mp4_crf = max(0, min(100, mp4_crf))
    latent_window_size = max(1, min(100, latent_window_size))
    cfg = max(0.1, min(20.0, cfg))
    rs = max(0.0, min(10.0, rs))

    try:
        # Decode input image
        input_image_np = decode_image(image_b64)
        print(f'Input image shape: {input_image_np.shape}')

        # Generate video
        output_path, total_frames = generate_video(
            input_image_np=input_image_np,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            total_second_length=total_second_length,
            latent_window_size=latent_window_size,
            steps=steps,
            cfg=cfg,
            gs=gs,
            rs=rs,
            gpu_memory_preservation=gpu_memory_preservation,
            use_teacache=use_teacache,
            mp4_crf=mp4_crf,
        )

        # Read output video
        with open(output_path, 'rb') as f:
            video_bytes = f.read()

        if encryption_enabled:
            # Encrypt output using _aes_encrypt_bytes
            video_bytes = _aes_encrypt_bytes(video_bytes)

        if _s3_configured():
            try:
                job_id = job.get('id') or 'unknown'
                video_val = _upload_to_s3(video_bytes, 'output.mp4', job_id)
            except Exception as e:
                return {'error': f'Failed to upload output to S3: {e}'}
        else:
            video_val = base64.b64encode(video_bytes).decode('utf-8')

        # Clean up temp file
        try:
            os.remove(output_path)
        except OSError:
            pass

        res_dict = {
            'video': video_val,
            'seed': seed,
            'duration_seconds': round(total_frames / 30.0, 2),
            'total_frames': total_frames,
        }
        if encryption_enabled:
            res_dict['encrypted'] = True

        return res_dict

    except Exception as e:
        traceback.print_exc()
        return {'error': str(e)}


runpod.serverless.start({'handler': handler})
