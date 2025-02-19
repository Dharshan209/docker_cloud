import os
import random
import torch
import numpy as np
from PIL import Image
import subprocess
import gradio as gr

# Import nodes and mappings
import nodes
from nodes import NODE_CLASS_MAPPINGS
from totoro_extras import nodes_custom_sampler, nodes_flux
from totoro import model_management

# Load nodes for the generation pipeline
CheckpointLoaderSimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
LoraLoader = NODE_CLASS_MAPPINGS["LoraLoader"]()
FluxGuidance = nodes_flux.NODE_CLASS_MAPPINGS["FluxGuidance"]()
RandomNoise = nodes_custom_sampler.NODE_CLASS_MAPPINGS["RandomNoise"]()
BasicGuider = nodes_custom_sampler.NODE_CLASS_MAPPINGS["BasicGuider"]()
KSamplerSelect = nodes_custom_sampler.NODE_CLASS_MAPPINGS["KSamplerSelect"]()
BasicScheduler = nodes_custom_sampler.NODE_CLASS_MAPPINGS["BasicScheduler"]()
SamplerCustomAdvanced = nodes_custom_sampler.NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()

# Initialize models
with torch.inference_mode():
    original_unet, original_clip, vae = CheckpointLoaderSimple.load_checkpoint("models/checkpoints/flux1-dev-fp8-all-in-one.safetensors")

# Initialize variables for tracking merged models
current_unet = original_unet
current_clip = original_clip
current_lora_filename = None
current_model_strength = 1.0
current_clip_strength = 1.0

def closestNumber(n, m):
    q = int(n / m)
    n1 = m * q
    if (n * m) > 0:
        n2 = m * (q + 1)
    else:
        n2 = m * (q - 1)
    if abs(n - n1) < abs(n - n2):
        return n1
    return n2

def download_lora(lora_url, hf_token):
    if not lora_url:
        return None
    filename = lora_url.split('/')[-1]
    try:
        subprocess.run([
            'aria2c', '--console-log-level=error',
            '-c', '-x', '16', '-s', '16', '-k', '1M',
            f'--header=Authorization: Bearer {hf_token}',
            lora_url,
            '-d', 'models/loras',
            '-o', filename
        ], check=True)
        return filename
    except Exception as e:
        print(f"Error downloading LoRA: {e}")
        return None

@torch.inference_mode()
def generate(num_scenes, *args):
    global current_unet, current_clip, current_lora_filename, current_model_strength, current_clip_strength

    # Parse arguments
    scene_prompts = list(args[:8])
    other_args = args[8:]
    width = other_args[0]
    height = other_args[1]
    seed = other_args[2]
    steps = other_args[3]
    sampler_name = other_args[4]
    scheduler = other_args[5]
    guidance = other_args[6]
    lora_url = other_args[7]
    hf_token = other_args[8]
    lora_strength_model = other_args[9]
    lora_strength_clip = other_args[10]

    num_scenes = int(num_scenes)
    scene_prompts = [p.strip() for p in scene_prompts[:num_scenes] if p.strip()]
    if not scene_prompts:
        return []

    # Handle LoRA
    lora_filename = download_lora(lora_url, hf_token) if lora_url else None
    if lora_filename:
        if (lora_filename != current_lora_filename or
            lora_strength_model != current_model_strength or
            lora_strength_clip != current_clip_strength):
            current_unet, current_clip = LoraLoader.load_lora(
                original_unet,
                original_clip,
                lora_filename,
                lora_strength_model,
                lora_strength_clip
            )
            current_lora_filename = lora_filename
            current_model_strength = lora_strength_model
            current_clip_strength = lora_strength_clip
    else:
        current_unet = original_unet
        current_clip = original_clip
        current_lora_filename = None

    # Create output directory if it doesn't exist
    os.makedirs("/workspace/outputs", exist_ok=True)

    # Seed handling
    if seed == 0:
        base_seed = random.randint(0, 18446744073709551615)
    else:
        base_seed = seed

    output_paths = []
    for i, positive_prompt in enumerate(scene_prompts):
        scene_seed = base_seed + i
        print(f"Generating scene {i+1} with seed: {scene_seed}")

        # Generate each scene
        cond, pooled = current_clip.encode_from_tokens(
            current_clip.tokenize(positive_prompt),
            return_pooled=True
        )
        cond = [[cond, {"pooled_output": pooled}]]
        cond = FluxGuidance.append(cond, guidance)[0]

        noise = RandomNoise.get_noise(scene_seed)[0]
        guider = BasicGuider.get_guider(current_unet, cond)[0]
        sampler = KSamplerSelect.get_sampler(sampler_name)[0]
        sigmas = BasicScheduler.get_sigmas(current_unet, scheduler, steps, 1.0)[0]
        latent_image = EmptyLatentImage.generate(closestNumber(width, 16), closestNumber(height, 16))[0]
        sample, sample_denoised = SamplerCustomAdvanced.sample(noise, guider, sampler, sigmas, latent_image)
        decoded = VAEDecode.decode(vae, sample)[0].detach()

        output_path = f"/workspace/outputs/flux_scene_{i+1}.png"
        Image.fromarray(np.array(decoded * 255, dtype=np.uint8)[0]).save(output_path)
        output_paths.append(output_path)

    return output_paths

# Set up the Gradio UI with RunPod-specific configurations
with gr.Blocks(analytics_enabled=False, css=".gradio-container {max-width: 1200px !important}") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            num_scenes = gr.Number(value=1, label="Number of Scenes", precision=0, minimum=1, maximum=8, step=1)
            width = gr.Slider(256, 2048, 1024, step=16, label="Width")
            height = gr.Slider(256, 2048, 1024, step=16, label="Height")
            seed = gr.Slider(0, 18446744073709551615, 0, step=1, label="Base Seed (0=random)")
            steps = gr.Slider(4, 100, 20, step=1, label="Steps")
            guidance = gr.Slider(0, 20, 3.5, step=0.1, label="Guidance")

            with gr.Accordion("LoRA Settings", open=False):
                lora_url = gr.Textbox(label="LoRA Model URL")
                hf_token = gr.Textbox(label="Hugging Face Token", type="password")
                lora_strength_model = gr.Slider(0, 1, 1.0, step=0.1, label="LoRA Model Strength")
                lora_strength_clip = gr.Slider(0, 1, 1.0, step=0.1, label="LoRA Clip Strength")

            sampler_name = gr.Dropdown(["euler", "heun", "dpm_2", "lms", "dpmpp_2m", "ddim"], value="euler", label="Sampler")
            scheduler = gr.Dropdown(["normal", "simple", "ddim_uniform"], value="simple", label="Scheduler")
            generate_btn = gr.Button("Generate All Scenes", variant="primary")

        with gr.Column(scale=2):
            scene_prompts = []
            with gr.Column() as scenes_container:
                for i in range(8):
                    scene_prompts.append(
                        gr.Textbox(visible=False,
                                 label=f"Scene {i+1} Prompt",
                                 placeholder=f"Description for scene {i+1}...",
                                 lines=2)
                    )
            gallery = gr.Gallery(label="Generated Scenes", columns=3, height=800)

    def update_scene_visibility(num_scenes):
        num = int(num_scenes)
        return [gr.Textbox.update(visible=i < num) for i, _ in enumerate(scene_prompts)]

    num_scenes.change(fn=update_scene_visibility, inputs=num_scenes, outputs=scene_prompts)

    inputs = [num_scenes] + scene_prompts + [
        width, height, seed, steps, sampler_name, scheduler, guidance,
        lora_url, hf_token, lora_strength_model, lora_strength_clip
    ]
    generate_btn.click(fn=generate, inputs=inputs, outputs=gallery)

# Launch with RunPod-specific settings
demo.queue(concurrency_count=1).launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,
    debug=False
)