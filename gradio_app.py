# Standard library imports
import os
import gc
import re  # added for filename cleaning
import math # for video filter effects
import time
import random
import warnings
import subprocess
import psutil  # for system stats - gpu/cpu etc
import threading # for model download monitoring
from datetime import datetime
from pathlib import Path
from subprocess import getoutput

# Third-party imports
import numpy as np
import torch
import gradio as gr
import imageio
import devicetorch

from rife_adapter import EnhancedRIFEModel
from huggingface_hub import snapshot_download
from huggingface_hub import hf_hub_download
from torchvision.transforms.functional import to_tensor, to_pil_image

# ML/AI framework imports
from diffusers.schedulers import EulerAncestralDiscreteScheduler
# from diffusers import AutoencoderKLAllegro, AllegroPipeline
from transformers import T5EncoderModel, T5Tokenizer

# Allegro-specific imports
from allegro.pipelines.pipeline_allegro import AllegroPipeline
from allegro.models.vae.vae_allegro import AllegroAutoencoderKL3D
from allegro.models.transformers.transformer_3d_allegro import AllegroTransformer3DModel


# Try to suppress annoyingly persistent Windows asyncio proactor errors
if os.name == 'nt':  # Windows only
    import asyncio
    from functools import wraps
    
    # Replace the problematic proactor event loop with selector event loop
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Patch the base transport's close method
    def silence_event_loop_closed(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except RuntimeError as e:
                if str(e) != 'Event loop is closed':
                    raise
        return wrapper
    
    # Apply the patch
    if hasattr(asyncio.proactor_events._ProactorBasePipeTransport, '_call_connection_lost'):
        asyncio.proactor_events._ProactorBasePipeTransport._call_connection_lost = silence_event_loop_closed(
            asyncio.proactor_events._ProactorBasePipeTransport._call_connection_lost)

# existing warning filters as backup
warnings.filterwarnings('ignore', message='.*Exception in callback.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*ConnectionResetError:,*', category=UserWarning)


# Constants and configurations
device = devicetorch.get(torch)

SPEED_FACTOR_MIN = 0.25      # adjust the min-max values for the Video Speed adjustment slider
SPEED_FACTOR_MAX = 2.0       #
SPEED_FACTOR_STEP = 0.05     # granularity of speed control (0.05 = 5% steps)
FPS = 15                     # best not to touch unless purposefully!
VIDEO_QUALITY = 8            # imageio quality setting (0-10, higher is better)
save_path = "output_videos"  # Can be changed to a preferred directory: "C:\path\to\save_folder"
INTERPOLATED_PATH = os.path.join(save_path, "interpolated")


# Templates
POSITIVE_TEMPLATE = """
(masterpiece), (best quality), (ultra-detailed),  4k, epic, detailed, cinematic, film grain,  sharp focus"""

NEGATIVE_TEMPLATE = """lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."""


weights_dir = './allegro_weights'
os.makedirs(weights_dir, exist_ok=True)

def check_weights_exist():
    required_paths = [
        os.path.join(weights_dir, folder) for folder in [
            'scheduler',
            'text_encoder',
            'tokenizer',
            'transformer',
            'vae'
        ]
    ]
    return all(os.path.exists(path) for path in required_paths)

def get_dir_size(path):
    """Get directory size in GB"""
    size = 0
    for root, _, files in os.walk(path):
        size += sum(os.path.getsize(os.path.join(root, name)) for name in files)
    return size / (1024 * 1024 * 1024)  # Convert to GB

def download_model():
    """Download model with size-based progress tracking"""
    if not check_weights_exist():
        messages = [
            "\n🚀 DOWNLOADING MODEL WEIGHTS (40GB)",
            "═" * 25,
            "📂 Location: " + os.path.abspath(weights_dir),
            "═" * 25
        ]
        
        try:
            progress = gr.Progress()
            download_thread = threading.Thread(target=snapshot_download, kwargs={
                'repo_id': 'rhymes-ai/Allegro',
                'allow_patterns': [
                    'scheduler/**',
                    'text_encoder/**',
                    'tokenizer/**',
                    'transformer/**',
                    'vae/**',
                ],
                'local_dir': weights_dir,
                'max_workers': 4
            })
            download_thread.start()
            
            # Monitor download progress
            total_size = 41.2  # Expected size in GB
            while download_thread.is_alive():
                current_size = get_dir_size(weights_dir)
                progress_pct = min(current_size / total_size, 0.99)  # Cap at 99% until complete
                progress(progress_pct, f"Downloaded: {current_size:.1f}GB / {total_size}GB")
                time.sleep(2)  # Check every 2 seconds
                
            download_thread.join()
            progress(1.0, "Download complete!")
            
            messages.extend([
                "✨ Download complete!",
                f"✓ Final size: {get_dir_size(weights_dir):.1f}GB",
                "═" * 25
            ])
            return "\n".join(messages), None, gr.update(visible=False), gr.update(visible=True)  # Update both buttons
            
        except Exception as e:
            messages.extend([
                "❌ Download failed:",
                f"• {str(e)}",
                "Please check your connection and try again",
                "═" * 25
            ])
            return "\n".join(messages), None, gr.update(visible=True), gr.update(visible=False)  # Update both buttons
            
    return "Model weights already present, skipping download.", None, gr.update(visible=False), gr.update(visible=True)

def check_button_visibility():
    return not check_weights_exist()
def check_generate_button_visibility():
    return check_weights_exist()
    
def run_inference(user_prompt, negative_prompt, guidance_scale, num_sampling_steps, 
                 seed, enable_cpu_offload, target_fps=15, progress=gr.Progress()):
    output_path = generate_output_path(user_prompt)
    dtype = torch.float16
    messages = []
    generation_start = None

    def add_message(msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {msg}"
        messages.append(formatted_msg)
        print(formatted_msg)
        
    try:
        # Initial status block
        summary_messages = [
            "\n🚀 STARTING VIDEO GENERATION",
            "═" * 25,
            "📄 Generation Settings:",
            f"• Prompt: '{user_prompt}'",
            f"• Steps: {num_sampling_steps}",
            f"• Guidance: {guidance_scale}",
            f"• Seed: {seed}",
            "═" * 25
        ]
        for msg in summary_messages:
            messages.append(update_console(msg, add_timestamp=False))
        
        progress(0.05, desc="Loading models...")
        
        # Load VAE
        add_message("📥 Loading VAE model...")
        vae = AllegroAutoencoderKL3D.from_pretrained(
            "./allegro_weights/vae/", 
            torch_dtype=torch.float32
        ).to(device)
        vae.eval()
        add_message("✓ VAE loaded successfully")

        # Load Text Encoder
        progress(0.10, desc="Loading text encoder...")
        add_message("📥 Loading text encoder (this takes a while)...")
        text_encoder = T5EncoderModel.from_pretrained(
            "./allegro_weights/text_encoder/", 
            torch_dtype=dtype
        ).to(device)
        text_encoder.eval()
        add_message("✓ Text encoder loaded successfully")

        # Load remaining models
        progress(0.50, desc="Loading transformer...")
        add_message("📥 Loading transformer model...")
        tokenizer = T5Tokenizer.from_pretrained("./allegro_weights/tokenizer/")
        scheduler = EulerAncestralDiscreteScheduler()
        transformer = AllegroTransformer3DModel.from_pretrained(
            "./allegro_weights/transformer/", 
            torch_dtype=dtype
        ).to(device)
        transformer.eval()
        add_message("✓ Transformer model loaded successfully")

        # Initialize pipeline
        progress(0.80, desc="Initializing pipeline...")
        add_message("🔄 Initializing Allegro pipeline...")
        allegro_pipeline = AllegroPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            transformer=transformer
        ).to(device)

        if enable_cpu_offload:
            add_message("💾 Enabling CPU offload to manage VRAM usage...")
            allegro_pipeline.enable_sequential_cpu_offload()

        devicetorch.empty_cache(torch)
        
        # Generation phase
        progress(1, desc="Starting generation -- check terminal for progress...")
        generation_start = time.time()
        
        def progress_callback(iter_num: int, t: int, latents: torch.FloatTensor) -> None:
            nonlocal generation_start
            elapsed = time.time() - generation_start
            
            current_step = iter_num + 1
            remaining_steps = num_sampling_steps - current_step
            eta_seconds = (elapsed / current_step) * remaining_steps if current_step > 0 else 0
            
            eta_min = int(eta_seconds // 60)
            eta_sec = int(eta_seconds % 60)
            
            percent_complete = 0.05 + (current_step/num_sampling_steps * 0.75)
            progress(percent_complete, desc=f"Step {current_step}/{num_sampling_steps} • ETA: {eta_min}m {eta_sec}s")
            
            progress_msg = f"Step {current_step}/{num_sampling_steps} • ETA: {eta_min}m {eta_sec}s"
            add_message(progress_msg)

        out_video = allegro_pipeline(
            user_prompt, 
            negative_prompt=negative_prompt, 
            num_frames=88,
            height=720,
            width=1280,
            # num_frames=8,  #fast 1-minute test inference
            # height=256,
            # width=512,
            num_inference_steps=num_sampling_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=512,
            output_type="np",
            generator=torch.Generator(device=device).manual_seed(seed),
            callback=progress_callback,
            callback_steps=1
        ).video[0]        

        generation_time = time.time() - generation_start
        add_message(f"✨ Generation complete! Took {generation_time:.1f} seconds")
        
        progress(0.8, desc="Processing output...")
        add_message("💾 Processing generated frames...")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Parse the FPS
        fps = int(target_fps.split()[0]) if isinstance(target_fps, str) else int(target_fps)
        
        # Handle basic save or interpolation
        if fps <= 15:
            add_message("📼 Saving video at original 15 FPS...")
            imageio.mimwrite(output_path, out_video, fps=15, quality=VIDEO_QUALITY)
            
            # Save prompt info
            save_prompt_info(
                output_path,
                user_prompt,
                negative_prompt,
                guidance_scale,
                num_sampling_steps,
                seed
            )
            
            add_message(f"✅ Generation complete! Saved to: {output_path}")
            progress(1.0, desc="Complete!")
            return output_path, "\n".join(messages)
            
        # Handle interpolation
        add_message(f"\n🎯 Starting frame interpolation to {fps} FPS...")
        progress(0.85, desc="Interpolating frames...")

        interpolation_start = time.time()
        interpolator = VideoInterpolator()
        interpolated_frames, interp_messages = interpolator.process_video(out_video, target_fps=fps)

        # Add interpolator messages to our message log
        for msg in interp_messages:
            add_message(msg)

        # Check if we actually got frames back (not just the original frames)
        if (isinstance(interpolated_frames, (list, np.ndarray)) and 
            len(interpolated_frames) == int(len(out_video) * (fps/15))):
            
            interpolation_time = time.time() - interpolation_start
            interpolated_path = output_path.replace('.mp4', f'_{fps}fps.mp4')
            
            add_message(f"⚡ Frame interpolation complete! Took {interpolation_time:.1f} seconds")
            add_message(f"📼 Saving interpolated video at {fps} FPS...")
            
            imageio.mimwrite(interpolated_path, interpolated_frames, fps=fps, quality=VIDEO_QUALITY)
            
            # Save prompt info for interpolated version
            save_prompt_info(
                interpolated_path,
                user_prompt,
                negative_prompt,
                guidance_scale,
                num_sampling_steps,
                seed,
                target_fps=fps
            )
            
            add_message(f"✅ Processing complete! Final video saved to: {interpolated_path}")
            progress(1.0, desc="Complete!")
            return interpolated_path, "\n".join(messages)
        else:
            # Log the actual vs expected frame count
            expected_frames = int(len(out_video) * (fps/15))
            actual_frames = len(interpolated_frames) if isinstance(interpolated_frames, (list, np.ndarray)) else 0
            add_message(f"⚠️ Interpolation frame count mismatch: got {actual_frames}, expected {expected_frames}")
            
            # Fallback to original
            add_message("⚠️ Interpolation verification failed, saving original video...")
            imageio.mimwrite(output_path, out_video, fps=15, quality=VIDEO_QUALITY)
            
            # Save prompt info for original
            save_prompt_info(
                output_path,
                user_prompt,
                negative_prompt,
                guidance_scale,
                num_sampling_steps,
                seed
            )
            
            add_message(f"✅ Original video saved to: {output_path}")
            progress(1.0, desc="Complete!")
            return output_path, "\n".join(messages)

    except Exception as e:
        error_messages = [
            "\n❌ ERROR DURING GENERATION",
            "═" * 25,
            f"Details: {str(e)}",
            "═" * 25
        ]
        for msg in error_messages:
            messages.append(update_console(msg, add_timestamp=False))
        return None, "\n".join(messages)

    finally:
        try:
            # Cleanup section
            del vae
            del text_encoder
            del transformer
            del allegro_pipeline
            devicetorch.empty_cache(torch)
            gc.collect()
            
            messages.append(update_console("✓ Resources cleaned up successfully", add_timestamp=False))
            
        except Exception as e:
            messages.append(update_console(f"⚠️ Cleanup warning (non-critical): {str(e)}", add_timestamp=False))

     
class VideoInterpolator:
    def __init__(self):
        self.model = None
        self.model_dir = Path("model_rife")
        self.model_file = self.model_dir / "flownet.pkl"
        self.device = devicetorch.get(torch)
        self.messages = []
        
    def add_message(self, msg):
        """Helper method to add messages consistently"""
        self.messages.append(update_console(msg, add_timestamp=False))
        print(msg)
        
    def check_model_exists(self):
        return self.model_file.exists()
        
    def download_model(self):
        try:
            self.add_message("📥 Downloading RIFE model from HuggingFace...")
            snapshot_download(
                repo_id="AlexWortega/RIFE",
                local_dir=str(self.model_dir)
            )
            if self.check_model_exists():
                self.add_message("✓ Model downloaded successfully")
                return True
            else:
                self.add_message("❌ Model download completed but model file not found")
                return False
        except Exception as e:
            self.add_message(f"❌ Download failed: {str(e)}")
            return False
    
    def load_model(self):
        if self.model is None:
            try:
                if not self.check_model_exists():
                    if not self.download_model():
                        return False
                
                try:
                    self.model = EnhancedRIFEModel()
                    self.model.load_model(str(self.model_dir), -1)
                    self.model.eval()
                    return True
                except Exception as e:
                    self.add_message(f"❌ Model loading failed: {str(e)}")
                    return False
                    
            except Exception as e:
                self.add_message(f"❌ Failed to load RIFE model: {str(e)}")
                return False
        return True

    def interpolate_frames(self, frame1, frame2, target_fps=30, original_fps=15):
        if not self.load_model():
            return []
            
        frames = []
        try:
            n_frames = (target_fps // original_fps) - 1
            
            if not isinstance(frame1, torch.Tensor):
                frame1 = to_tensor(frame1).unsqueeze(0)
            if not isinstance(frame2, torch.Tensor):
                frame2 = to_tensor(frame2).unsqueeze(0)
                
            with torch.no_grad():
                if n_frames == 1:
                    middle = self.model.inference(frame1, frame2, scale=1.0)
                    middle = middle.cpu()
                    frames = [to_pil_image(middle[0])]
                elif n_frames > 1:
                    middle = self.model.inference(frame1, frame2, scale=1.0)
                    middle = middle.cpu()
                    
                    if n_frames == 3:  # 15->60 fps case
                        first_quarter = self.model.inference(frame1, middle, scale=1.0)
                        third_quarter = self.model.inference(middle, frame2, scale=1.0)
                        
                        frames = [
                            to_pil_image(first_quarter[0].cpu()),
                            to_pil_image(middle[0]),
                            to_pil_image(third_quarter[0].cpu())
                        ]
                
        except Exception as e:
            self.add_message(f"❌ Error during frame interpolation: {str(e)}")  
            return []
            
        return frames
        
    def process_video(self, video_frames, target_fps=30):
        original_fps = 15
        self.messages = []
        
        if isinstance(video_frames, torch.Tensor):
            video_frames = video_frames.cpu().numpy()
            
        if not isinstance(video_frames, (list, np.ndarray)):
            return video_frames, self.messages
            
        if isinstance(video_frames, np.ndarray) and video_frames.size == 0:
            return video_frames, self.messages
            
        if len(video_frames) < 2:
            return video_frames, self.messages
            
        if target_fps <= original_fps:
            return video_frames, self.messages
            
        original_duration = len(video_frames) / original_fps
        expected_total_frames = int(original_duration * target_fps)
        total_frames = len(video_frames)
        result_frames = []
        
        try:
            for i in range(len(video_frames) - 1):
                # Add original frame
                result_frames.append(video_frames[i])
                
                # Get interpolated frames
                interp_frames = self.interpolate_frames(
                    video_frames[i],
                    video_frames[i + 1],
                    target_fps=target_fps,
                    original_fps=original_fps
                )
                
                if isinstance(interp_frames, (list, np.ndarray)) and len(interp_frames) > 0:
                    result_frames.extend(interp_frames)
                else:
                    self.add_message("❌ Interpolation failed")
                    return None, self.messages
            
            # Add final frame
            result_frames.append(video_frames[-1])
            
            # Adjust frame count if needed
            if len(result_frames) != expected_total_frames:
                if len(result_frames) > expected_total_frames:
                    result_frames = result_frames[:expected_total_frames]
                else:
                    while len(result_frames) < expected_total_frames:
                        result_frames.append(result_frames[-1])
                        
            return result_frames, self.messages
            
        except Exception as e:
            self.add_message(f"❌ Error during video processing: {str(e)}")
            return None, self.messages


def randomize_seed():
    return random.randint(0, 10000)


def get_system_info():
    """Get detailed system status"""
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
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    return f"{save_path}/alle_{timestamp}.mp4"  


def save_prompt_info(video_path, user_prompt, negative_prompt, guidance_scale, steps, seed, test_mode=False, target_fps=None):
    info_path = video_path.replace('.mp4', '_info.txt')
    with open(info_path, 'w') as f:
        f.write(f"Prompt: {user_prompt}\n")
        f.write(f"Negative Prompt: {negative_prompt}\n")
        f.write(f"Guidance Scale: {guidance_scale}\n")
        f.write(f"Steps: {steps}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Test Mode: {'Yes' if test_mode else 'No'}\n")
        if target_fps:
            f.write(f"Target FPS: {target_fps}\n")
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
        
# Manual post-processing tools

def interpolate_frames(frame1, frame2, timestep):
    """Helper function to interpolate between two frames using RIFE"""
    try:
        if not hasattr(interpolate_frames, 'model'):
            # Check for model directory and file
            model_dir = Path("model_rife")
            model_file = model_dir / "flownet.pkl"
            
            if not model_file.exists():
                print("Downloading RIFE model for first use...")
                os.makedirs(model_dir, exist_ok=True)
                try:
                    snapshot_download(
                        repo_id="AlexWortega/RIFE",
                        local_dir=str(model_dir)
                    )
                    if not model_file.exists():
                        raise Exception("Model download completed but model file not found")
                except Exception as e:
                    raise Exception(f"Failed to download RIFE model: {str(e)}")

            # Initialize model
            interpolate_frames.model = EnhancedRIFEModel()
            interpolate_frames.model.load_model("model_rife", -1)
            interpolate_frames.model.eval()
    
        # Convert frames to tensors if needed
        if not isinstance(frame1, torch.Tensor):
            frame1 = to_tensor(frame1).unsqueeze(0)
        if not isinstance(frame2, torch.Tensor):
            frame2 = to_tensor(frame2).unsqueeze(0)
        
        with torch.no_grad():
            middle = interpolate_frames.model.inference(frame1, frame2, timestep=timestep)
            middle = middle.cpu()
            return to_pil_image(middle[0])
            
    except Exception as e:
        raise Exception(f"Frame interpolation failed: {str(e)}")
     
     
def process_loop_video(video_path, loop_type="none", num_loops=2, progress=gr.Progress()):
    """Loop video forwards or ping-pong it back and forth"""
    messages = []
    
    # Check for no input video
    if video_path is None:
        return None, "\n".join(update_console(msg, add_timestamp=False) for msg in [
            "═" * 25,
            "⚠️ NO INPUT VIDEO",
            "═" * 25,
            "Please upload or send a video to the input window first!"
        ])
    
    try:
        # Simple header
        messages.extend([
            "🔄 VIDEO LOOP PROCESSING",
            f"• Input: {os.path.basename(video_path)}",
            f"• Mode: {loop_type.title()}",
            f"• Loops: {num_loops}x"
        ])
        
        # Create output path
        os.makedirs(INTERPOLATED_PATH, exist_ok=True)
        filename = os.path.basename(video_path)
        name = clean_filename(os.path.splitext(filename)[0])
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        output_path = os.path.join(INTERPOLATED_PATH, f"{name}_{loop_type}_{num_loops}x_{timestamp}.mp4")
        
        
        if loop_type == "ping-pong":
            messages.append("Creating ping-pong effect...")
            filter_complex = f"[0:v]reverse[r];[0:v][r]concat=n=2:v=1[v];[v]loop={num_loops-1}:32767:0[final]"
            subprocess.run([
                'ffmpeg', '-y',
                '-i', video_path,
                '-filter_complex', filter_complex,
                '-map', '[final]',
                '-c:v', 'libx264',
                output_path
            ], capture_output=True)
        else:  # standard loop
            messages.append("Creating standard loop...")
            subprocess.run([
                'ffmpeg', '-y',
                '-stream_loop', str(num_loops - 1),
                '-i', video_path,
                '-c', 'copy',
                output_path
            ], capture_output=True)
            
        messages.extend([
            "✨ Processing complete!",
            f"📊 Output: {os.path.join('output_videos', 'interpolated', os.path.basename(output_path))}"
        ])
        
        return output_path, "\n".join(update_console(msg, add_timestamp=False) for msg in messages)
        
    except Exception as e:
        messages.extend([
            "❌ Error creating loop:",
            f"• {str(e)}"
        ])
        return None, "\n".join(update_console(msg, add_timestamp=False) for msg in messages)


def process_existing_video(video_path, target_fps, speed_factor=1.0, progress=gr.Progress(track_tqdm=False)):
    """Process an existing video file with RIFE interpolation and speed control"""
    messages = []
    
    def add_message(msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {msg}"
        messages.append(formatted_msg)
        print(formatted_msg)

    # Check for no input video
    if video_path is None:
        return None, "\n".join(update_console(msg, add_timestamp=False) for msg in [
            "═" * 25,
            "⚠️ NO INPUT VIDEO",
            "═" * 25,
            "Please upload or send a video to the input window first!"
        ])
        
    try:
        # Input summary block
        summary_messages = [
            "\n🎬  NEW VIDEO PROCESSING TASK",
            "─" * 25,
            "📥  Input Details:",
            f"• Source: {os.path.basename(video_path)}",
            f"• Settings: {target_fps}, Speed: {speed_factor}x",
            "─" * 25
        ]
        for msg in summary_messages:
            messages.append(update_console(msg, add_timestamp=False))

        # Extract audio from source if it exists
        audio_path = None
        try:
            temp_audio = video_path + '.temp.wav'
            add_message("📻 Extracting audio track...")
            subprocess.run([
                'ffmpeg', '-y', '-i', video_path, 
                '-vn', '-acodec', 'pcm_s16le', 
                '-ar', '44100', '-ac', '2',
                temp_audio
            ], capture_output=True)
            if os.path.exists(temp_audio) and os.path.getsize(temp_audio) > 0:
                audio_path = temp_audio
                add_message("✓ Audio track extracted")
        except Exception as e:
            add_message(f"Note: No audio track found or error extracting: {str(e)}")
            
        # Load video
        video_frames = imageio.mimread(video_path, memtest=False)
        if not video_frames:
            add_message("❌ Error: Could not load video frames")
            return None, "\n".join(messages)
            
        # Get video info
        reader = imageio.get_reader(video_path)
        original_fps = reader.get_meta_data()['fps']
        original_frame_count = len(video_frames)
        duration = original_frame_count / original_fps
        reader.close()
        
        # Parse multiplier from choice
        multiplier_map = {"0x fps": 0, "2x fps": 2, "3x fps": 3, "4x fps": 4}
        fps_multiplier = multiplier_map.get(target_fps, 0)
        
        # Step 1: Speed Adjustment
        if speed_factor != 1.0:
            progress(0.3, desc="Adjusting video speed...")
            add_message(f"⚡ Applying speed adjustment ({speed_factor}x)...")
            
            if speed_factor > 1.0:
                step = speed_factor
                indices = np.arange(0, len(video_frames), step)
                video_frames = [video_frames[int(i)] for i in indices if int(i) < len(video_frames)]
            else:
                new_frame_count = int(len(video_frames) / speed_factor)
                indices = np.linspace(0, len(video_frames) - 1, new_frame_count)
                video_frames = [video_frames[int(i)] for i in indices]
            
            add_message(f"✓ Speed adjustment complete: {len(video_frames):,} frames")
            
        # Step 2: Frame Interpolation
        if fps_multiplier > 0:
            progress(0.5, desc="Preparing interpolation...")
            
            # Check if RIFE model needs to be downloaded
            model_dir = Path("model_rife")
            model_file = model_dir / "flownet.pkl"
            
            if not model_file.exists():
                add_message("\n📥 Downloading RIFE interpolation model (first run only)...")
                try:
                    os.makedirs(model_dir, exist_ok=True)
                    try:
                        snapshot_download(
                            repo_id="AlexWortega/RIFE",
                            local_dir=str(model_dir)
                        )
                    except Exception as download_error:
                        error_msg = str(download_error).lower()
                        if any(term in error_msg for term in [
                            "connection", 
                            "access",
                            "maxretryerror",
                            "newconnectionerror",
                            "connectionerror",
                            "timeout",
                            "failed to establish",
                            "forbidden"
                        ]):
                            add_message("❌ Failed to download RIFE model: Check internet connection")
                        else:
                            add_message(f"❌ Failed to download RIFE model: {str(download_error)}")
                        return None, "\n".join(messages)
                        
                    if not model_file.exists():
                        add_message("❌ Model download was incomplete. Please try again.")
                        return None, "\n".join(messages)
                        
                    add_message("✓ RIFE model downloaded successfully")
                    
                except Exception as e:
                    add_message(f"❌ RIFE model installation error: {str(e)}")
                    return None, "\n".join(messages)
            
            add_message(f"🔄  Starting {fps_multiplier}x frame interpolation...")
            result_frames = []
            total_frames = len(video_frames) - 1
            
            for i in range(total_frames):
                current_progress = (i / total_frames) * 100
                if i % 10 == 0:  # Update every 10 frames
                    progress(0.5 + (0.3 * i / total_frames), desc="Interpolating frames...")
                    add_message(f" Frame {i+1:,}/{total_frames:,} ({current_progress:.0f}%)")
                
                try:
                    # Add original frame
                    result_frames.append(video_frames[i])
                    
                    # Add interpolated frames
                    for j in range(fps_multiplier - 1):
                        timestep = (j + 1) / fps_multiplier
                        interpolated = interpolate_frames(video_frames[i], video_frames[i + 1], timestep)
                        result_frames.append(interpolated)
                except Exception as e:
                    error_msg = str(e)
                    add_message(f"⚠️ Interpolation failed at frame {i}")
                    
                    # Check for tensor size mismatch
                    if "must match the size" in error_msg and "at non-singleton dimension" in error_msg:
                        error_messages = [
                            "═" * 25,
                            "📊  FRAME SIZE MISMATCH DETECTED",
                            "─" * 25,
                            "Found inconsistent frame sizes in the video",
                            "\n⚠️  Common Causes:",
                            "• Variable resolution videos",
                            "• Some specific video codecs",
                            "• Videos that have been edited or re-encoded",
                            "• AI-generated content",
                            "\n💡  Solutions:",
                            " 👉 Process the video at Interpolation 0x, and send back to input",
                            "\nOtherwise, try one of the following:",
                            "• Re-encode with constant frame size:",
                            "• Convert to MP4 with h264 codec",
                            "• Use a video editor to export at fixed dimensions",
                            "═" * 25
                        ]
                        
                        # Add messages without timestamps
                        for msg in error_messages:
                            messages.append(update_console(msg, add_timestamp=False))
                    else:
                        # Run the regular validation check for other types of errors
                        add_message("\n🔍 Running video integrity check...")
                        is_valid, issues = validate_video(video_path)
                        
                        if not is_valid and issues:
                            add_message("Found potential issues with the video file:")
                            for issue in issues:
                                add_message(f"• {issue}")
                            add_message("\nThese issues may have caused the interpolation failure.")
                        else:
                            add_message("✓ No obvious video integrity issues found")
                            add_message(f"Original error: {error_msg}")
                    
                    return None, "\n".join(messages)
                    
            # Add final frame
            result_frames.append(video_frames[-1])
            video_frames = result_frames
            
        # Save processed video
        progress(0.9, desc="Saving video...")
        
        # Create output path
        os.makedirs(INTERPOLATED_PATH, exist_ok=True)
        filename = os.path.basename(video_path)
        name, _ = os.path.splitext(filename)
        name = clean_filename(name)
        
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        speed_text = f"_{speed_factor:.1f}x" if speed_factor != 1.0 else ""
        fps_text = f"_{fps_multiplier}x" if fps_multiplier > 0 else ""
        output_path = os.path.join(INTERPOLATED_PATH, f"{name}{speed_text}{fps_text}_{timestamp}.mp4")
        
        add_message("💾  Saving processed video...")
        
        final_fps = original_fps * fps_multiplier if fps_multiplier > 0 else original_fps
        
        # First save without audio
        temp_video = output_path + '.temp.mp4'
        imageio.mimwrite(temp_video, video_frames, fps=final_fps, quality=VIDEO_QUALITY)
        
        # If we have audio, combine it with the video
        if audio_path:
            try:
                add_message("🔊 Processing audio track...")
                
                # Adjust audio speed to match video
                speed_adjusted_audio = audio_path + '.speed.wav'
                subprocess.run([
                    'ffmpeg', '-y', '-i', audio_path,
                    '-filter:a', f'atempo={speed_factor}',
                    speed_adjusted_audio
                ], capture_output=True)
                
                # Combine video and speed-adjusted audio
                subprocess.run([
                    'ffmpeg', '-y',
                    '-i', temp_video,
                    '-i', speed_adjusted_audio,
                    '-c:v', 'copy', '-c:a', 'aac',
                    output_path
                ], capture_output=True)
                
                # Cleanup temp files
                for f in [audio_path, speed_adjusted_audio, temp_video]:
                    try:
                        if f and os.path.exists(f):
                            os.remove(f)
                    except:
                        pass
                        
                add_message("✓ Audio track processed and merged")
                
            except Exception as e:
                add_message(f"⚠️ Audio processing failed: {str(e)}")
                # Fallback to video without audio
                os.rename(temp_video, output_path)
        else:
            # No audio to process, just rename the temp video
            os.rename(temp_video, output_path)
        
        # Final summary
        final_summary = [
            "\n✨  PROCESSING COMPLETE",
            "─" * 25,
            "📊  Saved to:",
            f"{os.path.abspath(output_path)}"
        ]
        for msg in final_summary:
            messages.append(update_console(msg, add_timestamp=False))
        
        return output_path, "\n".join(messages)
        
    except Exception as e:
        add_message(f"\n❌ Error processing video: {str(e)}")
        add_message("═" * 25)
        return None, "\n".join(messages)


def create_vignette_filter(strength):
    """Create a natural vignette effect using proper vignette filter parameters"""
    # Convert strength (0-100) to angle (0 to PI/2)
    # Higher strength = stronger vignette effect = lower angle
    # Map strength 0-100 to angle PI/2 (none) down to PI/6 (max)
    max_angle = math.pi / 2    # No vignette (strength = 0)
    min_angle = math.pi / 6    # Max vignette (strength = 100)
    
    # Reverse the strength mapping so 100 = strongest vignette
    angle = max_angle - ((100 - strength) / 100) * (max_angle - min_angle)
    
    return (
        f"vignette=angle={angle:.4f}:mode=forward:eval=init:aspect=1/1"
    )

# Video filter presets:
def update_sliders_for_preset(preset_name):
    preset_settings = {
        "none": {
            "contrast": 1,
            "saturation": 1,
            "brightness": 0,
            "temperature": 0,
            "vignette": 0,
            "sharpen": 0,
            "blur": 0,
            "denoise": 0
        },
        "cinematic": {
            "contrast": 1.3,
            "saturation": 0.9,
            "brightness": -5,
            "temperature": 20,
            "vignette": 10,
            "sharpen": 1.2,
            "blur": 0,
            "denoise": 0
        },
        "vintage": {
            "contrast": 1.1,
            "saturation": 0.7,
            "brightness": 5,
            "temperature": 15,
            "vignette": 30,
            "blur": 0.5,
            "denoise": 0,
            "sharpen": 0
        },
        "cool": {
            "contrast": 1.2,
            "saturation": 1.1,
            "brightness": 0,
            "temperature": -15,
            "vignette": 0,
            "sharpen": 1.0,
            "blur": 0,
            "denoise": 0
        },
        "warm": {
            "contrast": 1.1,
            "saturation": 1.2,
            "brightness": 5,
            "temperature": 20,
            "vignette": 0,
            "sharpen": 0,
            "blur": 0,
            "denoise": 0
        },
        "dramatic": {
            "contrast": 1.4,
            "saturation": 0.9,
            "brightness": -5,
            "temperature": 0,
            "vignette": 20,
            "sharpen": 1.5,
            "blur": 0,
            "denoise": 0
        }
    }
    
    # Get the settings for the selected preset
    settings = preset_settings.get(preset_name, preset_settings["none"])
    
    # Return values for all sliders in the order they're defined in the UI
    return [
        settings["brightness"],
        settings["contrast"],
        settings["saturation"],
        settings["temperature"],
        settings["sharpen"],
        settings["blur"],
        settings["denoise"],
        settings["vignette"]
    ]


def apply_video_filters(video_path, brightness=0, contrast=1, saturation=1, blur=0, 
                       sharpen=0, denoise=0, vignette=0, temperature=0, 
                       preset="none", progress=gr.Progress()):
    """Apply video filters using FFmpeg using current slider values"""
    messages = []
    
    def add_message(msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {msg}"
        messages.append(formatted_msg)
        print(formatted_msg)

    if video_path is None:
        return None, "\n".join(update_console(msg, add_timestamp=False) for msg in [
            "═" * 25,
            "⚠️ NO INPUT VIDEO",
            "═" * 25,
            "Please upload or send a video to the input window first!"
        ])

    try:
        # Input summary
        add_message("\n🎨 APPLYING VIDEO FILTERS")
        add_message("─" * 25)
        add_message(f"Source: {os.path.basename(video_path)}")

        # Build filter chain
        filters = []
        
        # 1. Start with denoise if enabled (clean up before other operations)
        if denoise > 0:
            filters.append(f"hqdn3d={denoise:.1f}")
            add_message(f"• Applying noise reduction (strength: {denoise:.1f})")

        # 2. Color corrections
        if temperature != 0:
            temp = temperature / 100
            if temp > 0:
                filters.append(f"colorbalance=rs=0.05*{temp}:bs=-0.05*{temp}")
            else:
                filters.append(f"colorbalance=rs={temp*0.05}:bs={-temp*0.05}")
            add_message(f"• Adjusting temperature ({temperature:+.0f})")

        # 3. Basic adjustments
        if brightness != 0 or contrast != 1:
            filters.append(f"eq=brightness={brightness/100:.2f}:contrast={contrast:.2f}")
            add_message(f"• Adjusting brightness ({brightness:+.0f}%) and contrast ({contrast:.1f}x)")

        if saturation != 1:
            filters.append(f"eq=saturation={saturation:.2f}")
            add_message(f"• Adjusting saturation ({saturation:.1f}x)")

        # 4. Sharpening/Blur (after basic adjustments, before vignette)
        if blur > 0:
            filters.append(f"boxblur={blur:.1f}:1")
            add_message(f"• Adding blur effect (strength: {blur:.1f})")

        if sharpen > 0:
            amount = sharpen * 0.8
            filters.append(f"unsharp=5:5:{amount}")
            add_message(f"• Adding sharpening (strength: {sharpen:.1f})")

        # 5. Vignette (apply last)
        if vignette > 0:
            filters.append(create_vignette_filter(vignette))
            add_message(f"• Adding vignette effect ({vignette:.0f}%)")

        # Create output path
        os.makedirs(INTERPOLATED_PATH, exist_ok=True)
        filename = os.path.basename(video_path)
        name = clean_filename(os.path.splitext(filename)[0])
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        output_path = os.path.join(INTERPOLATED_PATH, f"{name}_filtered_{timestamp}.mp4")

        # Build and execute FFmpeg command
        if filters:
            filter_str = ','.join(filters)
            add_message("\n🔄 Processing filters...")
            
            command = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-vf', filter_str,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '18',
                '-c:a', 'copy',
                output_path
            ]
            
            try:
                result = subprocess.run(command, capture_output=True, text=True)
                if result.returncode != 0:
                    add_message(f"\n⚠️ FFmpeg warning/error output:")
                    add_message(result.stderr)
                    return None, "\n".join(messages)
                
                add_message("\n✨ Filtering complete!")
                add_message(f"📊 Output: {os.path.join('output_videos', 'interpolated', os.path.basename(output_path))}")
                return output_path, "\n".join(messages)
            except subprocess.CalledProcessError as e:
                error_msg = e.stderr if e.stderr else str(e)
                add_message(f"\n❌ FFmpeg error: {error_msg}")
                return None, "\n".join(messages)
        else:
            add_message("\n⚠️ No filters selected!")
            return None, "\n".join(messages)

    except Exception as e:
        add_message(f"\n❌ Error processing video: {str(e)}")
        return None, "\n".join(messages)


def analyze_video_input(video_path):
    """Analyze uploaded video and return formatted information"""
    messages = []
    
    try:
        messages.append("═" * 25)
        messages.append("📽️ VIDEO ANALYSIS")
        messages.append("═" * 25)
        
        # Basic file info
        file_name = os.path.basename(video_path)
        file_size = os.path.getsize(video_path)
        file_ext = os.path.splitext(file_name)[1].lower()
        
        # Get detailed video info using imageio
        reader = imageio.get_reader(video_path)
        meta = reader.get_meta_data()
        frame_count = reader.count_frames()
        fps = meta.get('fps', 0)
        duration = frame_count / fps if fps else 0
        
        # Format file size nicely
        def format_size(size):
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size < 1024:
                    return f"{size:.1f} {unit}"
                size /= 1024
            return f"{size:.1f} TB"
        
        # Build information display
        messages.append("📄  File Details:")
        messages.append(f"• Name: {file_name}")
        messages.append(f"• Type: {file_ext.upper()[1:]} video")
        messages.append(f"• Size: {format_size(file_size)}")
        
        messages.append("\n🎬  Video Properties:")
        messages.append(f"• Duration: {duration:.2f} seconds")
        messages.append(f"• Frames: {frame_count:,}")
        messages.append(f"• FPS: {fps}")
        
        # Resolution check and warning
        if 'size' in meta:
            width, height = meta['size']
            messages.append(f"• Resolution: {width}x{height}")
            
            # Add resolution-based warnings
            if width >= 3840 or height >= 2160:  # 4K
                messages.append("\nHigh Resolution Video:")
                messages.append("• 4K video detected - expect long processing times")
            elif width >= 2560 or height >= 1440:  # 1440p
                messages.append("• High resolution video (1440p+)")
                messages.append("• Processing time will be increased")
            
        reader.close()
        return "\n".join(update_console(msg, add_timestamp=False) for msg in messages)
        
    except Exception as e:
        return "\n".join(update_console(msg, add_timestamp=False) for msg in [
            "❌ Error analyzing video file:",
            f"• {str(e)}",
            "• Please ensure the file is a valid video format",
            "\n• Ignore this if you just manually cleared the video window!"
        ])


def validate_video(video_path):
    """Enhanced video validation checking"""
    if not video_path:
        return False, ["No video file provided"]
        
    try:
        # First check with FFmpeg
        result = subprocess.run([
            'ffmpeg', 
            '-v', 'error',           
            '-i', video_path,        
            '-f', 'null',            
            '-'                      
        ], capture_output=True, text=True)
        
        # Then try to read with imageio for additional validation
        try:
            reader = imageio.get_reader(video_path)
            meta = reader.get_meta_data()
            
            # Check for valid metadata
            if not meta or 'fps' not in meta:
                reader.close()
                return False, ["Video metadata appears corrupted or unsupported"]
                
            # Check for empty video
            frame_count = reader.count_frames()
            if frame_count == 0:
                reader.close()
                return False, ["Video file contains no frames"]
                
            reader.close()
            
        except Exception as e:
            error_msg = str(e).lower()
            if any(term in error_msg for term in ['format', 'codec', 'decode', 'corrupt']):
                return False, [
                    f"Video format error: {str(e)}",
                    "Try converting to MP4 with H.264 codec"
                ]
            return False, [f"Error reading video: {str(e)}"]
            
        # Check FFmpeg output for errors
        if result.stderr:
            issues = [line.strip() for line in result.stderr.split('\n') if line.strip()]
            if issues:
                return False, issues
            
        return True, []
        
    except Exception as e:
        return False, [f"Error checking video: {str(e)}"]

        
def process_batch(prompts_text, negative_prompt, guidance_scale, num_sampling_steps, 
                 seed, enable_cpu_offload, target_fps, progress=gr.Progress()):
    messages = []
    def add_message(msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {msg}"
        messages.append(formatted_msg)
        print(formatted_msg)
        return "\n".join(messages)

    try:
        # Parse prompts
        prompts = [p.strip() for p in prompts_text.split('---') if p.strip()]
        if not prompts:
            return None, "No valid prompts found in batch input."

        # Initial batch status
        summary_messages = [
            "\n🎬 STARTING BATCH GENERATION",
            "═" * 25,
            f"📄 Batch Details:",
            f"• Total Jobs: {len(prompts)}",
            f"• Steps: {num_sampling_steps}",
            f"• Guidance: {guidance_scale}",
            f"• Base Seed: {seed}",
            "═" * 25
        ]
        for msg in summary_messages:
            messages.append(update_console(msg, add_timestamp=False))

        # Process each prompt
        for i, prompt in enumerate(prompts, 1):
            job_messages = [
                f"\n📌 JOB {i}/{len(prompts)}",
                "─" * 25,
                f"Prompt: {prompt}",
                f"Seed: {seed + i - 1}",  # Increment seed for each job
                "─" * 25
            ]
            for msg in job_messages:
                messages.append(update_console(msg, add_timestamp=False))

            try:
                # Run the generation
                video_path, job_messages = run_inference(
                    prompt, 
                    negative_prompt,
                    guidance_scale,
                    num_sampling_steps,
                    seed + i - 1,  # Use incremented seed
                    enable_cpu_offload,
                    target_fps,
                    progress
                )
                
                # Add job messages to batch messages
                messages.extend(job_messages.split('\n'))
                
                if video_path:
                    add_message(f"✅ Job {i}/{len(prompts)} completed successfully")
                else:
                    add_message(f"❌ Job {i}/{len(prompts)} failed")
                    
                # Force cleanup between jobs
                devicetorch.empty_cache(torch)
                gc.collect()
                
            except Exception as e:
                add_message(f"❌ Error in job {i}: {str(e)}")
                continue  # Continue with next job even if one fails

        # Final summary
        completion_messages = [
            "\n✨ BATCH PROCESSING COMPLETE",
            "═" * 25,
            f"📊 Results:",
            f"• Total Jobs: {len(prompts)}",
            f"• Check output folder for generated videos",
            "═" * 25
        ]
        for msg in completion_messages:
            messages.append(update_console(msg, add_timestamp=False))

        return None, "\n".join(messages)

    except Exception as e:
        error_msg = f"\n❌ Batch processing error: {str(e)}"
        messages.append(update_console(error_msg, add_timestamp=False))
        return None, "\n".join(messages)


def update_console(msg, console_output=None, add_timestamp=True):
    """Update the console output with optional timestamping"""
    if add_timestamp and not msg.startswith('['):
        timestamp = datetime.now().strftime("%H:%M:%S")
        msg = f"[{timestamp}] {msg}"
    
    print(msg) 
    
    if console_output is not None:
        current = console_output.value if console_output.value else ""
        return f"{current}\n{msg}".strip()
    return msg


def clean_filename(filename):
    """Clean filename while preserving processing markers"""
    # Remove timestamp patterns but keep processing markers
    filename = re.sub(r'_\d{6}_\d{6}', '', filename)  # Removes YYMMDD_HHMMSS
    filename = re.sub(r'_\d{6}_\d{4}', '', filename)  # Removes YYMMDD_HHMM
    return filename.strip('_')  # Remove any trailing underscores
        
        
        
def get_welcome_message():
    return """
🎬 What to expect:
• Generation takes about 1 hour per video (on a 3090)
• Output will be 720 x 1280
• Each video is ~88 frames 
• 15fps, 30fps, or 60fps depending on Interpolation setting (doesn't affect duration)
• The video models (40GB+) are downloaded manually via the big UI button

## 📝 Notes: 
• If you're only planning on using the manual tools in the Tool Box, ignore the Generate Video section and avoid the massive model downloads!


⚙️ Important Settings:
• "Enable CPU Offload" is ON by default - recommended for most users
• Only disable CPU Offload if you have a Workstation GPU with 30GB+ VRAM 
• Generation will fail if you run out of VRAM!
• You can adjust some of the default settings, like save path and default prompt, by editing the webui_app.py. Look for the # Constants and configurations section near the top (currently line: 66).


🎯 For best results:
• Experiment with creative prompting
• The default prompts are provided for general guidance. Feel free to change.
• Check out prompting info in the Gen Info tab ->
• The Tool Box is full of useful post-processing tools to play with!


Ready to generate? Enter your prompt above and click 'Generate Video'
"""

def get_toolbox_info():
    return """🧰 TOOL BOX GUIDE

## 📝 Overview
The Tool Box provides FFmpeg derived post-processing tools for polishing generated or imported videos. You can adjust speed, increase frame rates, add filters, and create loops.

## 🎯 Video Input/Output
• Drop videos directly into the Input window
• Processed videos appear in the Output window
• Use "⬅️ Use as Input" to chain multiple effects
• Generated videos can be sent to Tool Box using "⬇️ Send to Tool Box"

• If a video loaded into the Tool Box shows blank with NaN:NaN, just ignore or click the Process button to send it back to Input. It's just a gradio display bug.

## ⚡ Frame Interpolation
RIFE AI interpolation smooths motion by generating extra frames:
• 0x fps: No interpolation (original)
• 2x fps: 15fps → 30fps
• 3x fps: 15fps → 45fps
• 4x fps: 15fps → 60fps

## ⏱️ Speed Control
Adjust playback speed from 0.25x to 2.0x:
• <1.0: Slow motion effect
• >1.0: Speed up footage
• 1.0: Original speed
Tip: Combine with interpolation for smooth slow-mo

## 🔄 Video Loops
Three loop options:
• none: No looping
• loop: Standard repeat
• ping-pong: Forward then reverse
Adjust number of loops (1-5x)

## 🎨 Video Filters
Apply visual enhancements:

Style Presets:
• cinematic: Movie-like color grading
• vintage: Retro film look
• cool: Blue-tinted atmosphere
• warm: Golden/amber tones
• dramatic: High contrast, moody

Tip: Start with presets, then fine-tune with manual controls!

Note: I haven't really done much with these. Very, very easy to adjust them in the code to your tastes. 

Presets can be altered by editing gradio_app.py -> update_sliders_for_preset() [currently around line 930]
"""

def get_gen_info():
    return """🎬 GENERATION GUIDE

## 🎯 Core Settings
• Guidance Scale (0-20):
  - 7.5: Balanced (recommended)
  - Lower: More creative/abstract
  - Higher: More literal/accurate
  
• Sampling Steps (10-100):
  - 20: Quick results
  - 50: Good balance
  - 100: Maximum quality
  Note: More steps = longer generation time


## 💫 Prompting Tips
Strong prompts include:
1. Quality markers:
   • (masterpiece), (best quality), (ultra-detailed)
   
2. Visual style:
   • cinematic, film grain, sharp focus
   • moody, dramatic, atmospheric
   
3. Camera details:
   • wide shot, close-up, aerial view
   • tracking shot, dolly zoom
   
4. Lighting:
   • natural sunlight, neon glow
   • golden hour, blue hour
   
5. Motion descriptions:
   • slowly moving, floating
   • spinning, zooming, panning

Example Prompt:
"(masterpiece), (best quality), cinematic wide shot of a misty forest at dawn, golden sunlight filtering through trees, gentle wind moving leaves, atmospheric, moody lighting, sharp focus, 4k"


## 📦 Batch Processing
Process multiple videos overnight:

• Format:
  Separate prompts with "---"

  First prompt goes here...
  ---
  Second prompt goes here...
  ---
  Third prompt...

• Info:
  - WIP! I've tested it some, and seems to work as expected.
  - No hard limit to number of prompts.  Be careful!
  - CFG, Steps, and Interp settings currently apply to every prompt
  - Seeds auto-increment for variety
  - Generation continues if one fails
  - Check output folder for results
  - Use at your own risk! 
 
  
• Example Batch:

  (masterpiece), forest scene at dawn...
  ---
  (masterpiece), ocean waves at sunset...
  ---
  (masterpiece), city streets in rain...

"""
css = """
.video-natural-size {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
}

.video-natural-size video {
    max-width: 100% !important;
    max-height: 100vh !important;
    width: auto !important;
    height: auto !important;
    object-fit: contain !important;
}
"""

    
#UI title bar  
title = """<style>.allegro-banner{background:linear-gradient(to bottom,#162828,#101c1c);color:#fff;padding:0.5rem;border-radius:0.5rem;border:1px solid rgba(255,255,255,0.1);box-shadow:0 4px 6px rgba(0,0,0,0.1);margin-bottom:0.5rem;text-align:center}.allegro-banner h1{font-size:1.75rem;margin:0 0 0.25rem 0;font-weight:300;color:#ff6b35 !important}.allegro-banner p{color:#b0c4c4;font-size:1rem;margin:0 0 0.75rem 0}.allegro-banner .footer{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;font-size:0.875rem;color:#a0a0a0}.allegro-banner .powered-by{display:flex;align-items:center;gap:0.25rem}.allegro-banner .credits{display:flex;flex-direction:column;align-items:center;gap:0.25rem}.allegro-banner a{color:#4a9eff;text-decoration:none;transition:color 0.2s ease}.allegro-banner a:hover{color:#6db3ff}@media (max-width:768px){.allegro-banner .footer{flex-direction:column;gap:0.5rem;align-items:center}}</style><div class="allegro-banner"><h1>Allegro Text-to-Video</h1><p>Transform your prompts to video.</p><div class="footer"><div class="powered-by"><span>⚡ Powered by</span><a href="https://pinokio.computer/" target="_blank">Pinokio</a></div><div class="credits"><div>OG Project: <a href="https://github.com/rhymes-ai/Allegro" target="_blank">Rhymes</a></div><div>Code borrowed from: <a href="https://huggingface.co/spaces/fffiloni/allegro-text2video" target="_blank">fffiloni</a></div></div></div></div>"""


# Create Gradio interface
with gr.Blocks(css=css) as demo:
    #gr.HTML(title)
    with gr.Row():
        video_output = gr.Video(label="Generated Video", elem_classes="video-natural-size")
    with gr.Row():
        download_button = gr.Button("Download txt2vid Models First! (40GB) - not required for Tool Box.", variant="primary", visible=check_button_visibility())
        submit_btn = gr.Button("Generate Video", variant="primary", scale=4, visible=check_generate_button_visibility())
        transfer_to_toolbox_btn = gr.Button("⬇️ Send to Tool Box", visible=False, scale=1, variant="huggingface")
        
    with gr.Row():        
        with gr.Column():
            user_prompt = gr.Textbox(
                value=POSITIVE_TEMPLATE,
                label="Add your video prompt",
                lines=2
            )
            
            with gr.Row():
                guidance_scale = gr.Slider(minimum=1, maximum=20, step=0.5, 
                                       label="Guidance Scale", value=7.5)
                num_sampling_steps = gr.Slider(minimum=10, maximum=100, step=1, 
                                           label="Number of Sampling Steps", info="+quality ++inference time",value=20)    
                
            with gr.Row():
                seed = gr.Slider(minimum=0, maximum=10000, step=1, label="Seed", value=42, scale=3)
                random_seed = gr.Button("🎲 randomize seed", scale=1)
                
            with gr.Row():
                enable_cpu_offload = gr.Checkbox(label="Enable CPU Offload", info="Don't touch unless certain!", value=True)
                target_fps = gr.Radio(
                    choices=["15 FPS (Original)", "30 FPS", "60 FPS"],
                    value="60 FPS",
                    label="Interpolation Options",
                )

            # Dev tools in accordion
            with gr.Accordion("Tool Box", open=False):        
                with gr.Row():
                    input_video = gr.Video(label="Upload Video for Processing")
                    output_processed = gr.Video(label="Processed Video", interactive=False)

                with gr.Row():
                    process_fps = gr.Radio(
                        choices=["0x fps", "2x fps", "3x fps", "4x fps"],
                        value="0x fps",
                        label="RIFE Frame Interpolation",
                        info="Smooth motion by increasing fps"
                    )
                with gr.Row():
                    send_to_main_btn = gr.Button("⬆️ Send to Main Display", visible=False, variant="huggingface")
                    send_to_input_btn = gr.Button("⬅️ Use as Input", visible=False, variant="huggingface")
                    
                with gr.Row():                    
                    speed_factor = gr.Slider(
                        minimum=SPEED_FACTOR_MIN,
                        maximum=SPEED_FACTOR_MAX,
                        step=SPEED_FACTOR_STEP,
                        value=1.0,
                        label="Adjust Video Speed",
                        info="Slow-mo (0.25x) to speed-up (2x)"
                    )
                with gr.Row():    
                    process_btn = gr.Button("Process Frame Adjustments", variant="primary")
                gr.HTML('<hr style="margin: 20px 0; border: none; border-top: 1px solid rgba(128, 128, 128, 0.2);">') 
                
                with gr.Accordion("🔄 Video Loop", open=True):        
                    with gr.Row():
                        loop_type = gr.Radio(
                            choices=["none", "loop", "ping-pong"],
                            value="none",
                            label="Loop Type",
                            info="Loop or ping-pong the video"
                        )
                        num_loops = gr.Slider(
                            minimum=1,
                            maximum=5,
                            step=1,
                            value=2,
                            label="Number of Loops"
                        )
                    with gr.Row():
                        loop_btn = gr.Button("Create Loop (⬅️ on Input Video)", variant="primary")
                    gr.HTML('<hr style="margin: 20px 0; border: none; border-top: 1px solid rgba(128, 128, 128, 0.2);">')

                with gr.Accordion("🎨 Video Filters - FFmpeg", open=True):
                    with gr.Row():
                        preset = gr.Radio(
                            choices=["none", "cinematic", "vintage", "cool", "warm", "dramatic"],
                            value="none",
                            label="Style Presets",
                            info="Quick style applications"
                        )
                    
                    with gr.Row():
                        brightness = gr.Slider(-100, 100, value=0, step=1, label="Brightness", info="Adjust video brightness")
                        contrast = gr.Slider(0, 2, value=1, step=0.1, label="Contrast", info="Adjust video contrast")
                        
                    with gr.Row():
                        saturation = gr.Slider(0, 2, value=1, step=0.1, label="Saturation", info="Adjust color intensity")
                        temperature = gr.Slider(-50, 50, value=0, step=1, label="Temperature", info="Adjust color temperature (cool/warm)")
                        
                    with gr.Row():
                        sharpen = gr.Slider(0, 5, value=0, step=0.1, label="Sharpen", info="Add sharpening effect")
                        blur = gr.Slider(0, 5, value=0, step=0.1, label="Blur", info="Add blur effect")
                        
                    with gr.Row():
                        denoise = gr.Slider(0, 5, value=0, step=0.1, label="Denoise", info="Reduce video noise")
                        vignette = gr.Slider(0, 100, value=0, step=1, label="Vignette", info="Add dark corners effect")
                        
                    with gr.Row():
                        apply_filters_btn = gr.Button("Apply Filters (⬅️ on Input Video)", variant="primary")



                        
                # gr.HTML('<hr style="margin: 20px 0; border: none; border-top: 1px solid rgba(128, 128, 128, 0.2);">')            
            # with gr.Accordion("Dev Tools", open=False):
                # with gr.Row():
                    # test_btn = gr.Button("🧪 Two Minute Test Generation", variant="primary")
                # with gr.Row():
                    # num_frames = gr.Slider(minimum=8, maximum=88, step=1, 
                                           # label="Number of Frames", info="for test purposes only",value=88)
                    # height = gr.Slider(minimum=360, maximum=720, step=1, 
                                           # label="Video Height", info="for test purposes only",value=720)
                    # width = gr.Slider(minimum=640, maximum=1280, step=1, 
                                           # label="Video Width", info="for test purposes only",value=1280)
                    
        with gr.Column():    
            negative_prompt = gr.Textbox(
                value=NEGATIVE_TEMPLATE,
                label="Negative Prompt",
                placeholder="Enter negative prompt",
                lines=2
            )
            # batch generations
            with gr.Row():
                with gr.Accordion("Batch Processing", open=False):
                    batch_prompts = gr.TextArea(
                        label="Batch Prompts (separate with ---)",
                        placeholder="First prompt here...\n---\nSecond prompt here...\n---\nThird prompt...",
                        lines=5
                    )
                    batch_btn = gr.Button("Start Batch Generation", variant="primary")  
                    
            # Welcome & monitor Info
            with gr.Row():
                with gr.Accordion("Helpful Docs", open=True):   
                    with gr.Tabs() as info_tabs:
                        with gr.Tab("Welcome"):
                            welcome_info = gr.Textbox(
                                value=get_welcome_message(),
                                label="Welcome to Allegro txt2vid!",
                                lines=20,
                                interactive=False
                            )
                        with gr.Tab("Tool Box info"):
                            toolbox_info = gr.Textbox(
                                value=get_toolbox_info(),  
                                label="Tool Box",
                                lines=20,
                                interactive=False
                            )
                        with gr.Tab("Prompting and Batch"):
                            gen_info = gr.Textbox(
                                value=get_gen_info(),  
                                label="Prompting and Batch",
                                lines=20,
                                interactive=False
                            )
            with gr.Row():
                open_folder_btn = gr.Button("📁 Open Output Folder")  
                
            with gr.Row():       
                # Status Info (for cpu/gpu monitor)
                status_info = gr.Textbox(
                    label="Monitor",
                    lines=5,
                    interactive=False,
                    value=get_welcome_message()
                )    
                # System Messages Console
            with gr.Row():
                console_out = gr.Textbox(
                    label="System Messages",
                    lines=8,
                    interactive=False,
                    show_copy_button=True
                )


    # Event handlers
    
    download_button.click(
        fn=download_model,
        outputs=[console_out, video_output, download_button, submit_btn],  # Add submit_btn
        show_progress=True
    )
    
    random_seed.click(fn=randomize_seed, outputs=seed)
    
    # Timer that updates system info
    timer = gr.Timer(value=1)
    timer.tick(
        fn=lambda: get_system_info(),
        outputs=status_info
    )
    
    open_folder_btn.click(
        fn=open_output_folder,
        inputs=None,
        outputs=console_out
    )
    
    submit_btn.click(
        fn=run_inference,
        inputs=[
            user_prompt, 
            negative_prompt, 
            guidance_scale, 
            num_sampling_steps, 
            seed, 
            enable_cpu_offload,
            target_fps
        ],
        outputs=[video_output, console_out],
        show_progress=True
    )

    # test_btn.click(
        # fn=run_inference,
        # inputs=[
            # user_prompt, 
            # negative_prompt, 
            # guidance_scale, 
            # num_sampling_steps, 
            # seed, 
            # enable_cpu_offload, 
            # target_fps, 
            # height, 
            # width, 
            # num_frames
        # ],
        # outputs=[video_output, console_out],
        # show_progress=True
    # )
   
    def reset_processing_controls():
        return "0x fps", 1.0  # Default values for process_fps and speed_factor

    # reset post-processing controls
    input_video.change(
        fn=reset_processing_controls,
        inputs=None,
        outputs=[process_fps, speed_factor]
    )
    
    def clear_console():
        return gr.update(value="")
    
    # Update the process button click handler
    process_btn.click(
        fn=lambda: clear_console(),  # First clear the console
        outputs=[console_out],
    ).then(  # Then start processing
        fn=process_existing_video,
        inputs=[input_video, process_fps, speed_factor],
        outputs=[output_processed, console_out],
        show_progress="minimal"
    )
    
    # loop functions
    loop_btn.click(
        fn=process_loop_video,
        inputs=[input_video, loop_type, num_loops],
        outputs=[output_processed, console_out]
    )
    
    # function for when video is manually cleared
    def on_video_clear():
        return "\n".join(update_console(msg, add_timestamp=False) for msg in [
            "═" * 25,
            "📽️ VIDEO INPUT CLEARED",
            "═" * 25,
            "Ready for new video input"
        ])
    input_video.clear(
        fn=on_video_clear,
        outputs=[console_out]
    )
    
    # Analyze input video when loaded or received  
    input_video.change(
        fn=analyze_video_input,
        inputs=[input_video],
        outputs=[console_out],
    ).then(  # Chain with the existing reset controls
        fn=reset_processing_controls,
        outputs=[process_fps, speed_factor]
    )
    
    # Show/hide "Send to Input" button based on output_processed state
    output_processed.change(
        fn=lambda x: [gr.update(visible=bool(x)), gr.update(visible=bool(x))],
        inputs=[output_processed],
        outputs=[send_to_input_btn, send_to_main_btn]
    )
    
    # click handler for the filters button
    apply_filters_btn.click(
        fn=apply_video_filters,
        inputs=[
            input_video,
            brightness,
            contrast, 
            saturation,
            blur,
            sharpen,
            denoise,
            vignette,
            temperature,
            preset
        ],
        outputs=[output_processed, console_out],
        show_progress="minimal"
    )
    
   
    # Show/hide transfer buttons based on video_output state
    output_processed.change(
        fn=lambda x: gr.update(visible=bool(x)), 
        inputs=[output_processed],
        outputs=send_to_main_btn
    )
    # Show/hide transfer button based on video_output state
    video_output.change(
        fn=lambda x: gr.update(visible=bool(x)),
        inputs=[video_output],
        outputs=[transfer_to_toolbox_btn]
    )

    
    def transfer_to_toolbox(video):
        """Transfer generated video to toolbox input and reset processing controls"""
        return (
            video,      # input_video
            "0x fps",   # process_fps
            1.0,        # speed_factor
            None,       # output_processed (clear it)
         )
    # main video to toolbox
    transfer_to_toolbox_btn.click(
        fn=transfer_to_toolbox,
        inputs=[video_output],
        outputs=[
            input_video,
            process_fps,
            speed_factor,
            output_processed
        ]
    )
    # send processed video back to input
    def send_to_input(video):
        return video, "0x fps", 1.0, None  # Return as tuple in order of outputs

    # send toolbox result to input
    send_to_input_btn.click(
        fn=send_to_input,
        inputs=[output_processed],
        outputs=[input_video, process_fps, speed_factor, output_processed]
    )
    
    # Transfer processed video to main display
    def send_to_main_display(video):
        return video

    send_to_main_btn.click(
        fn=send_to_main_display,
        inputs=[output_processed],
        outputs=[video_output]
    )       

    # handler for the preset radio button:
    preset.change(
        fn=update_sliders_for_preset,
        inputs=[preset],
        outputs=[
            brightness,
            contrast,
            saturation,
            temperature,
            sharpen,
            blur,
            denoise,
            vignette
        ]
    )
    
    # batch run button:
    batch_btn.click(
        fn=process_batch,
        inputs=[
            batch_prompts,
            negative_prompt,
            guidance_scale,
            num_sampling_steps,
            seed,
            enable_cpu_offload,
            target_fps
        ],
        outputs=[video_output, console_out],
        show_progress=True
    )


# Launch the interface
demo.launch(share=False)
