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

from datetime import datetime
from pathlib import Path
from subprocess import getoutput
from huggingface_hub import snapshot_download

from diffusers.schedulers import EulerAncestralDiscreteScheduler
from transformers import T5EncoderModel, T5Tokenizer
from allegro.pipelines.pipeline_allegro import AllegroPipeline
from allegro.models.vae.vae_allegro import AllegroAutoencoderKL3D
from allegro.models.transformers.transformer_3d_allegro import AllegroTransformer3DModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
save_path = "./output_videos"  # Can be changed to a preferred directory: "C:\path\to\save_folder"
FPS = 15
VIDEO_QUALITY = 8  # imageio quality setting (0-10, higher is better)

weights_dir = './allegro_weights'
os.makedirs(weights_dir, exist_ok=True)

# prompt templates
POSITIVE_TEMPLATE = """(masterpiece), (best quality), (ultra-detailed), (unwatermarked),


emotional, harmonious, vignette, 4k epic detailed, shot on kodak, 35mm photo, sharp focus, high budget, cinemascope, moody, epic, gorgeous"""

NEGATIVE_TEMPLATE = """lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry.
"""

# Function to check if weights are already downloaded
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


def get_system_info():
    """Get detailed system status with both nvidia-smi and PyTorch metrics"""
    try:
        import subprocess
        gpu_info = f"🎮 GPU: {torch.cuda.get_device_name(0)}\n"
        
        # GPU memory from nvidia-smi
        result = subprocess.check_output([
            'nvidia-smi', '--query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
        memory_used, memory_total, temp, util = map(int, result.strip().split(','))
        gpu_memory = memory_used / 1024  # Convert MB to GB
        total_memory = memory_total / 1024
        
        # Peak memory from PyTorch
        peak_memory = torch.cuda.max_memory_allocated(0)/1024**3
        
        gpu_info += f"📊 GPU Memory: {gpu_memory:.1f}GB / {total_memory:.1f}GB\n"
        gpu_info += f"📈 Peak Usage: {peak_memory:.1f}GB\n"
        gpu_info += f"🌡️ GPU Temp: {temp}°C\n"
        gpu_info += f"⚡ GPU Load: {util}%\n"
        
    except Exception as e:
        gpu_info = f"🎮 GPU: {torch.cuda.get_device_name(0)}\n"
        gpu_info += f"📊 VRAM Reserved: {torch.cuda.memory_reserved(0)/1024**3:.2f}GB\n"
        gpu_info += f"📈 Peak Usage: {torch.cuda.max_memory_allocated(0)/1024**3:.2f}GB\n"
    
    cpu_info = f"💻 CPU Usage: {psutil.cpu_percent(interval=1.0)}%\n"
    ram_info = f"🎯 RAM Usage: {psutil.virtual_memory().percent}%"
    
    return f"{gpu_info}{cpu_info}{ram_info}"


def update_system_info():
    return f"""🎮 System Status:

{get_system_info()}"""


def get_completion_message(output_path):
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    return f"""✨ Generation Complete!
💾 Saved as: alle_{timestamp}.mp4

{get_system_info()}"""


def generate_output_path(user_prompt):
    timestamp = datetime.now().strftime("%y%m%d_%H%M")  
    return f"./save_path/alle_{timestamp}.mp4"


def save_prompt_info(video_path, user_prompt, negative_prompt, guidance_scale, steps, seed):
    info_path = video_path.replace('.mp4', '_info.txt')
    with open(info_path, 'w') as f:
        f.write(f"Prompt: {user_prompt}\n")
        f.write(f"Negative Prompt: {negative_prompt}\n")
        f.write(f"Guidance Scale: {guidance_scale}\n")
        f.write(f"Steps: {steps}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Generated: {datetime.now().strftime('%y%m%d_%H%M')}\n")


def run_inference(user_prompt, negative_prompt, guidance_scale, num_sampling_steps, seed, enable_cpu_offload, progress=gr.Progress(track_tqdm=True)):
    output_path = generate_output_path(user_prompt)
    
    try:
        result_path = single_inference(
            user_prompt=user_prompt,
            negative_prompt=negative_prompt,
            save_path=output_path,
            guidance_scale=guidance_scale,
            num_sampling_steps=num_sampling_steps,
            seed=seed,
            enable_cpu_offload=enable_cpu_offload
        )
        return result_path
    
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        return None
     

def randomize_seed():
    return random.randint(0, 10000)
    
    
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
title = """
<style>
.title-container{text-align:center;margin:auto;padding:8px 12px;background:linear-gradient(to bottom,#162828,#101c1c);color:#fff;border-radius:8px;font-family:Arial,sans-serif;border:2px solid #0a1212;box-shadow:0 2px 4px rgba(0,0,0,0.1);position:relative}.title-container h1{font-size:2em;margin:0 0 5px;font-weight:300;color:#ff6b35}.title-container p{color:#b0c4c4;font-size:0.9em;margin:0 0 5px}.title-container a{color:#ff6b35;text-decoration:none;transition:color 0.3s ease}.title-container a:hover{color:#ff8c5a}.links-left,.links-right{position:absolute;bottom:5px;font-size:0.8em;color:#a0a0a0}.links-left{left:10px}.links-right{right:10px}.emoji-icon{vertical-align:middle;margin-right:3px;font-size:1em}
</style>
<div class="title-container">
<h1>Allegro Text-to-Video</h1>
<p>Transform your prompts into video - Takes about an hour for completion.</p>
<div class="links-left"><span class="emoji-icon">⚡</span>Powered by <a href="https://pinokio.computer/" target="_blank">Pinokio</a></div>
<div class="links-right">OG Project by <a href="https://github.com/rhymes-ai/Allegro" target="_blank">Rhymes</a> | gradio_app code borrowed from <a href="https://huggingface.co/spaces/fffiloni/allegro-text2video" target="_blank">fffiloni</a></div>
</div>
"""

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
                random_seed = gr.Button("🎲", size="sm", scale=1)
            with gr.Row():
                enable_cpu_offload = gr.Checkbox(label="Enable CPU Offload", value=True, scale=1)
                    
                    
            
        with gr.Column():    
            negative_prompt = gr.Textbox(
                value=NEGATIVE_TEMPLATE,
                label="Negative Prompt",
                placeholder="Enter negative prompt",
                lines=2
            )
               
            with gr.Row():
                status_info = gr.Textbox(
                    label="Status",
                    lines=5,
                    interactive=False,
                    value=get_welcome_message()
                )


    random_seed.click(fn=randomize_seed, outputs=seed)
    timer = gr.Timer(1, active=False)  # 1 second interval, starts inactive
    
    # Timer updates system info
    timer.tick(
        fn=update_system_info,
        inputs=None,
        outputs=status_info
    )
    
    # Complete flow: start timer -> generate -> stop timer and show completion
    submit_btn.click(
        fn=lambda: gr.Timer(active=True),  # Start timer
        inputs=None,
        outputs=timer
    ).then(  # Generate video
        fn=run_inference,
        inputs=[user_prompt, negative_prompt, guidance_scale, 
                num_sampling_steps, seed, enable_cpu_offload],
        outputs=video_output,
        show_progress=True
    ).then(  # Stop timer and show completion
        fn=lambda: (gr.Timer(active=False), get_completion_message()),
        outputs=[timer, status_info]
    )

   
# Launch the interface
demo.launch(share=False)
