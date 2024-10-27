import torch
import imageio
import os
import gradio as gr
import time
import psutil
import subprocess
import warnings
import random
import gc
import devicetorch

from datetime import datetime
from pathlib import Path
from subprocess import getoutput
from huggingface_hub import snapshot_download

from diffusers.schedulers import EulerAncestralDiscreteScheduler
from transformers import T5EncoderModel, T5Tokenizer
from allegro.pipelines.pipeline_allegro import AllegroPipeline
from allegro.models.vae.vae_allegro import AllegroAutoencoderKL3D
from allegro.models.transformers.transformer_3d_allegro import AllegroTransformer3DModel

device = devicetorch.get(torch)
    
save_path = "output_videos"  # Can be changed to a preferred directory: "C:\path\to\save_folder"
FPS = 15
VIDEO_QUALITY = 8  # imageio quality setting (0-10, higher is better)

weights_dir = './allegro_weights'
os.makedirs(weights_dir, exist_ok=True)

# prompt templates
POSITIVE_TEMPLATE = """(masterpiece), (best quality), (ultra-detailed), (unwatermarked),

emotional, harmonious, vignette, 4k epic detailed, shot on kodak, 35mm photo, sharp focus, high budget, cinemascope, moody, epic, gorgeous"""

NEGATIVE_TEMPLATE = """lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry.
"""

# check if weights are already downloaded
def check_weights_exist():
    required_paths = [
        './allegro_weights/scheduler',
        './allegro_weights/text_encoder',
        './allegro_weights/tokenizer',
        './allegro_weights/transformer',
        './allegro_weights/vae'
    ]
    return all(os.path.exists(path) for path in required_paths)

# Modified download logic
weights_dir = './allegro_weights'
os.makedirs(weights_dir, exist_ok=True)

if not check_weights_exist():
    print("Downloading model weights...")
    snapshot_download(
        repo_id='rhymes-ai/Allegro',
        allow_patterns=[
            'scheduler/**',
            'text_encoder/**',
            'tokenizer/**',
            'transformer/**',
            'vae/**',
        ],
        local_dir=weights_dir,
    )
else:
    print("Model weights already present, skipping download.")
    
    
def single_inference(user_prompt, negative_prompt, save_path, guidance_scale, num_sampling_steps, seed, enable_cpu_offload):
    dtype = torch.float16 
    try:
        # Load models
        vae = AllegroAutoencoderKL3D.from_pretrained(
            "./allegro_weights/vae/", 
            torch_dtype=torch.float32
        ).to(device)
        vae.eval()

        text_encoder = T5EncoderModel.from_pretrained(
            "./allegro_weights/text_encoder/", 
            torch_dtype=dtype
        ).to(device)
        text_encoder.eval()

        tokenizer = T5Tokenizer.from_pretrained("./allegro_weights/tokenizer/")

        scheduler = EulerAncestralDiscreteScheduler()

        transformer = AllegroTransformer3DModel.from_pretrained(
            "./allegro_weights/transformer/", 
            torch_dtype=dtype
        ).to(device)
        transformer.eval()

        allegro_pipeline = AllegroPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            transformer=transformer
        ).to(device)

        if enable_cpu_offload:
            allegro_pipeline.enable_sequential_cpu_offload()

        # Clear any existing cache before generation
        torch.cuda.empty_cache()

        out_video = allegro_pipeline(
            user_prompt, 
            negative_prompt=negative_prompt, 
            num_frames=88,
            height=720,
            width=1280,
            num_inference_steps=num_sampling_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=512,
            generator=torch.Generator(device=device).manual_seed(seed)
        ).video[0]

        # Save video before cleanup
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        imageio.mimwrite(save_path, out_video, fps=FPS, quality=VIDEO_QUALITY)

        return save_path

    finally:
        # Cleanup section - runs even if there's an error
        try:
            # Delete model objects
            del vae
            del text_encoder
            del transformer
            del allegro_pipeline
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            print(f"Cleanup warning (non-critical): {str(e)}")


def run_inference(user_prompt, negative_prompt, guidance_scale, num_sampling_steps, seed, enable_cpu_offload, progress=gr.Progress(track_tqdm=True)):
    output_path = generate_output_path(user_prompt)
    
    try:
        # Create output directory
        os.makedirs(save_path, exist_ok=True)
        
        result_path = single_inference(
            user_prompt=user_prompt,
            negative_prompt=negative_prompt,
            save_path=output_path, 
            guidance_scale=guidance_scale,
            num_sampling_steps=num_sampling_steps,
            seed=seed,
            enable_cpu_offload=enable_cpu_offload
        )
        
        # Save prompt info alongside the video
        if result_path:
            save_prompt_info(
                result_path,
                user_prompt,
                negative_prompt,
                guidance_scale,
                num_sampling_steps,
                seed
            )
            
        return result_path
    
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        return None
     
     
def randomize_seed():
    return random.randint(0, 10000)

    
def get_system_info():
    """Get detailed system status including peak VRAM and shared memory"""
    try:
        # Basic GPU name
        gpu_info = f"🎮 GPU: {torch.cuda.get_device_name(0)}\n"
        
        # Get GPU metrics from nvidia-smi
        try:
            result = subprocess.check_output([
                'nvidia-smi', '--query-gpu=memory.used,memory.total,memory.reserved,temperature.gpu,utilization.gpu',
                '--format=csv,nounits,noheader'
            ], encoding='utf-8', timeout=1.0)
            memory_used, memory_total, memory_reserved, temp, util = map(int, result.strip().split(','))
            
            # Convert memory to GB for display
            gpu_memory = memory_used / 1024  # Convert MB to GB
            total_memory = memory_total / 1024
            shared_memory = memory_reserved / 1024
            
            gpu_info += f"📊 GPU Memory: {gpu_memory:.1f}GB / {total_memory:.1f}GB\n"
            gpu_info += f"💫 Shared Memory: {shared_memory:.1f}GB\n"
            gpu_info += f"🌡️ GPU Temp: {temp}°C\n"
            gpu_info += f"⚡ GPU Load: {util}%\n"
            
        except:
            gpu_info += "Unable to get detailed GPU metrics\n"
            
        # Quick CPU and RAM checks
        cpu_info = f"💻 CPU Usage: {psutil.cpu_percent()}%\n"
        ram_info = f"🎯 RAM Usage: {psutil.virtual_memory().percent}%"
        
        return f"{gpu_info}{cpu_info}{ram_info}"
        
    except Exception as e:
        return f"Error collecting system info: {str(e)}"


def generate_output_path(user_prompt):
    timestamp = datetime.now().strftime("%y%m%d_%H%M")  
    return f"{save_path}/alle_{timestamp}.mp4"  


def save_prompt_info(video_path, user_prompt, negative_prompt, guidance_scale, steps, seed):
    info_path = video_path.replace('.mp4', '_info.txt')
    with open(info_path, 'w') as f:
        f.write(f"Prompt: {user_prompt}\n")
        f.write(f"Negative Prompt: {negative_prompt}\n")
        f.write(f"Guidance Scale: {guidance_scale}\n")
        f.write(f"Steps: {steps}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Generated: {datetime.now().strftime('%y%m%d_%H%M')}\n")


def open_output_folder():
    folder_path = os.path.abspath(save_path) 
    
    # Create folder if it doesn't exist
    try:
        os.makedirs(folder_path, exist_ok=True)
    except Exception as e:
        return f"Error creating folder: {str(e)}"
        
    # Open folder
    try:
        if os.name == 'nt':  # Windows
            os.startfile(folder_path)
        elif os.name == 'posix':  # macOS and Linux
            subprocess.run(['xdg-open' if os.name == 'posix' else 'open', folder_path])
        return f"Opening folder: {folder_path}"
    except Exception as e:
        return f"Error opening folder: {str(e)}"

        
def update_info_display(display_type):
    if display_type == "welcome":
        return get_welcome_message()
    else:
        return get_system_info()
        
    
def get_welcome_message():
    return """Welcome to Allegro Text-to-Video!
    
🎬 What to expect:
• Generation takes about 1 hour per video (on a 3090)
• Output will be 720p at 15fps
• Each video is ~88 frames long

⚙️ Important Settings:
• "Enable CPU Offload" is ON by default - recommended for most users
• Only disable CPU Offload if you have a Workstation GPU with 30GB+ VRAM 
• Generation will fail if you run out of VRAM!

🎯 For best results:
• Keep the default quality tags in the prompt
• Add your creative prompt between the tag sections

📝 Examples:
• "A monkey playing the drums in a jazz club"
• "A spaceship landing on an alien planet"
• "A timelapse of a flower blooming in a garden"

Ready to generate? Enter your prompt above and click 'Generate Video'"""

    
#UI title bar  
title = """<style>.allegro-banner{background:linear-gradient(to bottom,#162828,#101c1c);color:#fff;padding:0.5rem;border-radius:0.5rem;border:1px solid rgba(255,255,255,0.1);box-shadow:0 4px 6px rgba(0,0,0,0.1);margin-bottom:0.5rem;text-align:center}.allegro-banner h1{font-size:1.75rem;margin:0 0 0.25rem 0;font-weight:300;color:#ff6b35 !important}.allegro-banner p{color:#b0c4c4;font-size:1rem;margin:0 0 0.75rem 0}.allegro-banner .footer{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;font-size:0.875rem;color:#a0a0a0}.allegro-banner .powered-by{display:flex;align-items:center;gap:0.25rem}.allegro-banner .credits{display:flex;flex-direction:column;align-items:center;gap:0.25rem}.allegro-banner a{color:#4a9eff;text-decoration:none;transition:color 0.2s ease}.allegro-banner a:hover{color:#6db3ff}@media (max-width:768px){.allegro-banner .footer{flex-direction:column;gap:0.5rem;align-items:center}}</style><div class="allegro-banner"><h1>Allegro Text-to-Video</h1><p>Transform your prompts to video.</p><div class="footer"><div class="powered-by"><span>⚡ Powered by</span><a href="https://pinokio.computer/" target="_blank">Pinokio</a></div><div class="credits"><div>OG Project: <a href="https://github.com/rhymes-ai/Allegro" target="_blank">Rhymes</a></div><div>Thanks: <a href="https://huggingface.co/spaces/fffiloni/allegro-text2video" target="_blank">fffiloni</a></div></div></div></div>"""


# Create Gradio interface
with gr.Blocks() as demo:
    gr.HTML(title)
    with gr.Row():
        video_output = gr.Video(label="Generated Video")
    with gr.Row():
        submit_btn = gr.Button("Generate Video", variant="primary")
    with gr.Row():        
        with gr.Column():
            user_prompt = gr.Textbox(
                value=POSITIVE_TEMPLATE,
                label="Add your video prompt: e.g. 'A monkey playing the drums'",
                lines=2
            )
            
            with gr.Row():
                guidance_scale = gr.Slider(minimum=0, maximum=20, step=0.1, 
                                       label="Guidance Scale", value=7.5)
                num_sampling_steps = gr.Slider(minimum=10, maximum=100, step=1, 
                                           label="Number of Sampling Steps", value=20)    
                
            with gr.Row():
                seed = gr.Slider(minimum=0, maximum=10000, step=1, label="Seed", value=42, scale=3)
                random_seed = gr.Button("🎲", scale=1)
            with gr.Row():
                enable_cpu_offload = gr.Checkbox(label="Enable CPU Offload", value=True, scale=1)
                    
        with gr.Column():    
            negative_prompt = gr.Textbox(
                value=NEGATIVE_TEMPLATE,
                label="Negative Prompt",
                placeholder="Enter negative prompt",
                lines=2
            )
               
            # Info display section with radio toggle
            with gr.Row():
                info_type = gr.Radio(
                    choices=["welcome", "system"],
                    value="welcome",
                    label="Information Display",
                    interactive=True
                )
                open_folder_btn = gr.Button("📁 Open Output Folder")
            
            with gr.Row():
                status_info = gr.Textbox(
                    label="Status",
                    lines=5,
                    interactive=False,
                    value=get_welcome_message()
                )

    # Event handlers
    
    random_seed.click(fn=randomize_seed, outputs=seed)
    
    # Timer that updates system info if system view is selected
    timer = gr.Timer(value=2)
    timer.tick(
        fn=lambda display_type: get_system_info() if display_type == "system" else status_info.value,
        inputs=[info_type],
        outputs=status_info
    )
    
    info_type.change(
        fn=update_info_display,
        inputs=[info_type],
        outputs=[status_info]
    )
    
    open_folder_btn.click(
        fn=open_output_folder,
        inputs=None,
        outputs=status_info
    )
    
    submit_btn.click(
        fn=run_inference,
        inputs=[user_prompt, negative_prompt, guidance_scale, 
                num_sampling_steps, seed, enable_cpu_offload],
        outputs=video_output,
        show_progress=True
    )

# Launch the interface
demo.launch(share=False)
