# Standard library imports
import os
import gc
import re  # added for filename cleaning
import time
import random
import warnings
import subprocess
import psutil
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
from transformers import T5EncoderModel, T5Tokenizer

# Allegro-specific imports
from allegro.pipelines.pipeline_allegro import AllegroPipeline
from allegro.models.vae.vae_allegro import AllegroAutoencoderKL3D
from allegro.models.transformers.transformer_3d_allegro import AllegroTransformer3DModel


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
POSITIVE_TEMPLATE = """(masterpiece), (best quality), (ultra-detailed), (unwatermarked),

emotional, harmonious, vignette, 4k epic detailed, shot on kodak, 35mm photo, sharp focus, high budget, cinemascope, moody, epic, gorgeous"""

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
        devicetorch.empty_cache(torch)
#        torch.cuda.empty_cache()

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
            devicetorch.empty_cache(torch)
            #torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            print(f"Cleanup warning (non-critical): {str(e)}")


def run_inference(user_prompt, negative_prompt, guidance_scale, num_sampling_steps, 
                 seed, enable_cpu_offload, target_fps=15, progress=gr.Progress(track_tqdm=True)):
    console_text = ""  # Initialize empty console text
    output_path = generate_output_path(user_prompt)
    dtype = torch.float16 
    try:
        msg = "Starting video generation..."
        print(msg)
        console_text = log_to_console(msg, console_text)
        
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

        #torch.cuda.empty_cache()
        devicetorch.empty_cache(torch)

        msg = "Loading complete. Starting pipeline..."
        print(msg)
        console_text = log_to_console(msg, console_text)
        
        out_video = allegro_pipeline(
            user_prompt, 
            negative_prompt=negative_prompt, 
            num_frames=88,
            height=720,
            width=1280,
            num_inference_steps=num_sampling_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=512,
            output_type="np",  # Ensure consistent numpy output for interpolator
            generator=torch.Generator(device=device).manual_seed(seed)
        ).video[0]

        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Parse the FPS from the radio button choice if needed
        fps = int(target_fps.split()[0]) if isinstance(target_fps, str) else int(target_fps)
        
        # If no interpolation needed, save and return original
        if fps <= 15:
            msg = "Saving video at original 15 FPS..."
            print(msg)
            console_text = log_to_console(msg, console_text)
            
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
            
            msg = "✨ Generation complete!"
            print(msg)
            console_text = log_to_console(msg, console_text)
            return output_path, console_text
            
        # Proceed with interpolation for higher FPS
        msg = f"Starting interpolation process for {fps} FPS..."
        print(msg)
        console_text = log_to_console(msg, console_text)
        
        interpolator = VideoInterpolator()
        interpolated_frames = interpolator.process_video(out_video, target_fps=fps)
        
        if isinstance(interpolated_frames, (list, np.ndarray)) and len(interpolated_frames) > 0:
            interpolated_path = output_path.replace('.mp4', f'_{fps}fps.mp4')
            msg = f"Saving interpolated video at {fps} FPS..."
            print(msg)
            console_text = log_to_console(msg, console_text)
            
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
            
            msg = "✨ Generation and interpolation complete!"
            print(msg)
            console_text = log_to_console(msg, console_text)
            return interpolated_path, console_text
        else:
            # Fallback to original if interpolation fails
            msg = "Interpolation failed, saving original video..."
            print(msg)
            console_text = log_to_console(msg, console_text)
            
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
            
            msg = "✨ Generation complete (fell back to original FPS)"
            print(msg)
            console_text = log_to_console(msg, console_text)
            return output_path, console_text

    except Exception as e:
        msg = f"Error during generation: {str(e)}"
        print(msg)
        console_text = log_to_console(msg, console_text)
        return None, console_text
        
    finally:
        try:
            # Cleanup section - runs even if there's an error
            del vae
            del text_encoder
            del transformer
            del allegro_pipeline
            #torch.cuda.empty_cache()
            devicetorch.empty_cache(torch)
            gc.collect()
        except Exception as e:
            msg = f"Cleanup warning (non-critical): {str(e)}"
            print(msg)
            console_text = log_to_console(msg, console_text)
     
     
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

        
def update_info_display(display_type):
    if display_type == "welcome":
        return get_welcome_message()
    else:
        return get_system_info()
        

class VideoInterpolator:
    def __init__(self):
        self.model = None
        self.model_dir = Path("model_rife")
        self.model_file = self.model_dir / "flownet.pkl"
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = devicetorch.get(torch)
        self.console_text = ""  # Initialize empty console text
        
    def log(self, msg):
        """Helper method for logging"""
        print(msg)
        self.console_text = log_to_console(msg, self.console_text)
        
    def check_model_exists(self):
        """Check if the model files already exist"""
        return self.model_file.exists()
        
    def download_model(self):
        """Download the model files if they don't exist"""
        try:
            self.log("Downloading RIFE model from HuggingFace...")
            snapshot_download(
                repo_id="AlexWortega/RIFE",
                local_dir=str(self.model_dir)
            )
            if self.check_model_exists():
                self.log(f"✔️ Model downloaded successfully to: {self.model_dir}")
                return True
            else:
                self.log("❌ Model download completed but model file not found")
                return False
        except Exception as e:
            self.log(f"❌ Download failed: {str(e)}")
            return False
    
    def load_model(self):
        """Lazy loading of RIFE model with smart download management"""
        if self.model is None:
            try:
                self.log("Loading RIFE interpolation model...")
                
                if not self.check_model_exists():
                    if not self.download_model():
                        return False
                else:
                    self.log("Using cached RIFE model")
                
                try:
                    self.log(f"Initializing RIFE model from {self.model_dir}")
                    self.model = EnhancedRIFEModel()
                    self.model.load_model(str(self.model_dir), -1)
                    self.model.eval()
                    self.log(f"✨ RIFE model loaded successfully on {self.device}!")
                    return True
                except Exception as e:
                    self.log(f"❌ Model loading failed: {str(e)}")
                    return False
                    
            except Exception as e:
                self.log(f"❌ Failed to load RIFE model: {str(e)}")
                return False
        return True

    def interpolate_frames(self, frame1, frame2, target_fps=30, original_fps=15):
        """Generate intermediate frames between two frames based on FPS ratio"""
        if not self.load_model():
            return []
            
        frames = []
        try:
            # Calculate how many frames we need between these two frames
            # For example: 15->30 fps needs 1 frame, 15->60 fps needs 3 frames
            n_frames = (target_fps // original_fps) - 1
            
            self.log(f"Interpolating frame pair (generating {n_frames} intermediate frames)...")
            
            # Convert frames to tensors if needed
            if not isinstance(frame1, torch.Tensor):
                frame1 = to_tensor(frame1).unsqueeze(0)
            if not isinstance(frame2, torch.Tensor):
                frame2 = to_tensor(frame2).unsqueeze(0)
                
            with torch.no_grad():
                if n_frames == 1:
                    # Simple case: just one intermediate frame
                    middle = self.model.inference(frame1, frame2, scale=1.0)
                    middle = middle.cpu()
                    frames = [to_pil_image(middle[0])]
                elif n_frames > 1:
                    # For higher frame rates, we need multiple intermediate frames
                    # First get the middle frame
                    middle = self.model.inference(frame1, frame2, scale=1.0)
                    middle = middle.cpu()
                    
                    # If we need more frames, interpolate between original and middle,
                    # and between middle and end
                    if n_frames == 3:  # 15->60 fps case
                        # Get frame between start and middle
                        first_quarter = self.model.inference(frame1, middle, scale=1.0)
                        # Get frame between middle and end
                        third_quarter = self.model.inference(middle, frame2, scale=1.0)
                        
                        frames = [
                            to_pil_image(first_quarter[0].cpu()),
                            to_pil_image(middle[0]),
                            to_pil_image(third_quarter[0].cpu())
                        ]
                
        except Exception as e:
            self.log(f"❌ Error during frame interpolation: {str(e)}")  
            return []
            
        return frames
        
    def process_video(self, video_frames, target_fps=30):
        """Process entire video with RIFE interpolation"""
        original_fps = 15  # Base FPS for generated videos
        
        self.log(f"Received video frames type: {type(video_frames)}")
        self.log(f"Shape/size of input: {video_frames.shape if hasattr(video_frames, 'shape') else len(video_frames)}")
        
        # Convert tensor to numpy if needed
        if isinstance(video_frames, torch.Tensor):
            self.log("Converting tensor to numpy array...")
            video_frames = video_frames.cpu().numpy()
            self.log(f"Converted shape: {video_frames.shape}")
        
        # Validate input
        if not isinstance(video_frames, (list, np.ndarray)):
            self.log(f"❌ Unexpected frame type after conversion: {type(video_frames)}")
            return video_frames
        if isinstance(video_frames, np.ndarray) and video_frames.size == 0:
            self.log("❌ Empty array received")
            return video_frames
        if len(video_frames) < 2:
            self.log(f"❌ Not enough frames: {len(video_frames)}")
            return video_frames
            
        if target_fps <= original_fps:
            self.log("No interpolation needed")
            return video_frames
            
        self.log(f"Starting RIFE interpolation:")
        self.log(f"• Input frames: {len(video_frames)}")
        self.log(f"• Original FPS: {original_fps}")
        self.log(f"• Target FPS: {target_fps}")
        
        # Calculate expected output frame count to maintain duration
        original_duration = len(video_frames) / original_fps
        expected_total_frames = int(original_duration * target_fps)
        
        self.log(f"• Original duration: {original_duration:.2f}s")
        self.log(f"• Expected output frames: {expected_total_frames}")
            
        total_frames = len(video_frames)
        result_frames = []
        
        try:
            for i in range(len(video_frames) - 1):
                self.log(f"Processing frame pair {i+1}/{total_frames-1}")
                
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
                    self.log("❌ Interpolation failed, returning original video")
                    return video_frames
            
            # Add final frame
            result_frames.append(video_frames[-1])
            
            # Verify frame count matches expected
            if len(result_frames) != expected_total_frames:
                self.log(f"⚠️ Frame count mismatch. Expected: {expected_total_frames}, Got: {len(result_frames)}")
                # Adjust if necessary by trimming or duplicating last frame
                if len(result_frames) > expected_total_frames:
                    result_frames = result_frames[:expected_total_frames]
                else:
                    while len(result_frames) < expected_total_frames:
                        result_frames.append(result_frames[-1])
                        
            self.log(f"✨ Interpolation complete! Generated {len(result_frames)} total frames")
            
        except Exception as e:
            self.log(f"❌ Error during video processing: {str(e)}")
            return video_frames
            
        return result_frames
        

def log_to_console(msg, console_text):
    """Add new message to console text with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    updated = f"{console_text}\n[{timestamp}] {msg}" if console_text else f"[{timestamp}] {msg}"
    return updated.strip()
    
    
## Test Zone!

def test_inference(user_prompt, negative_prompt, guidance_scale, num_sampling_steps, seed, enable_cpu_offload, target_fps):
    console_text = ""  # Initialize empty console text
    output_path = generate_output_path(user_prompt)
    dtype = torch.float16 
    try:
        msg = "Starting test inference..."
        print(msg)
        console_text = log_to_console(msg, console_text)
        # Load models same as normal
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

        #torch.cuda.empty_cache()
        devicetorch.empty_cache(torch)

        # Super quick test settings
        out_video = allegro_pipeline(
            user_prompt, 
            negative_prompt=negative_prompt, 
            num_frames=8,        # Minimum frames
            height=256,          # Smaller
            width=512,           # Smaller
            num_inference_steps=8,   # Faster
            guidance_scale=guidance_scale,
            max_sequence_length=512,
            output_type="np",
            generator=torch.Generator(device=device).manual_seed(int(seed))
        ).video[0]
        
        # Parse the FPS from the radio button choice
        fps = int(target_fps.split()[0])  # Converts "30 FPS" to 30
        
        # Save paths
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        test_path = output_path.replace('.mp4', '_test.mp4')
        
        # If we don't need interpolation, save and return original
        if fps <= 15:
            msg = "Saving test video at original FPS..."
            print(msg)
            console_text = log_to_console(msg, console_text)
            imageio.mimwrite(test_path, out_video, fps=15, quality=VIDEO_QUALITY)
            return test_path, console_text
            
        # Only proceed with interpolation if fps > 15
        msg = f"Starting interpolation process for {fps}fps..."
        print(msg)
        console_text = log_to_console(msg, console_text)
        interpolator = VideoInterpolator()
        interpolated_frames = interpolator.process_video(out_video, target_fps=fps)
        
        if isinstance(interpolated_frames, (list, np.ndarray)) and len(interpolated_frames) > 0:
            test_path_interp = output_path.replace('.mp4', f'_test_{fps}fps.mp4')
            msg = f"\nSaving interpolated video at {fps}fps..."
            print(msg)
            console_text = log_to_console(msg, console_text)
            imageio.mimwrite(test_path_interp, interpolated_frames, fps=fps, quality=VIDEO_QUALITY)
            msg = "Interpolation complete!"
            print(msg)
            console_text = log_to_console(msg, console_text)
            return test_path_interp, console_text
        else:
            # Fallback to original if interpolation fails
            msg = "\nInterpolation failed, saving original video..."
            print(msg)
            console_text = log_to_console(msg, console_text)
            imageio.mimwrite(test_path, out_video, fps=15, quality=VIDEO_QUALITY)
            return test_path, console_text

    except Exception as e:
        msg = f"Error in test_inference: {str(e)}"
        print(msg)
        console_text = log_to_console(msg, console_text)
        return None, console_text
    finally:
        try:
            del vae
            del text_encoder
            del transformer
            del allegro_pipeline
            #torch.cuda.empty_cache()
            devicetorch.empty_cache(torch)
            gc.collect()
        except Exception as e:
            print(f"Cleanup warning (non-critical): {str(e)}")

# Manual post-processing tools
def process_existing_video(video_path, target_fps, speed_factor=1.0, progress=gr.Progress(track_tqdm=True)):
    """Process an existing video file with RIFE interpolation and speed control
    
    Processing order:
    1. Apply speed adjustment first (if any) - affects duration only
    2. Apply frame interpolation second (if selected) - affects FPS only
    """
    console_text = ""
    
    msg = f"Loading video: {video_path}"
    print(msg)
    console_text = log_to_console(msg, console_text)
        
    try:
        # Extract audio from source if it exists
        audio_path = None
        try:
            temp_audio = video_path + '.temp.wav'
            subprocess.run([
                'ffmpeg', '-y', '-i', video_path, 
                '-vn', '-acodec', 'pcm_s16le', 
                '-ar', '44100', '-ac', '2',
                temp_audio
            ], capture_output=True)
            if os.path.exists(temp_audio) and os.path.getsize(temp_audio) > 0:
                audio_path = temp_audio
                msg = "✓ Audio track extracted"
                print(msg)
                console_text = log_to_console(msg, console_text)
        except Exception as e:
            msg = f"Note: No audio track found or error extracting: {str(e)}"
            print(msg)
            console_text = log_to_console(msg, console_text)
            
        # Load the video
        video_frames = imageio.mimread(video_path, memtest=False)
        if not video_frames:
            msg = "Error: Could not load video frames"
            print(msg)
            console_text = log_to_console(msg, console_text)
            return None, console_text
            
        # Get original video info
        reader = imageio.get_reader(video_path)
        original_fps = reader.get_meta_data()['fps']
        original_frame_count = len(video_frames)
        duration = original_frame_count / original_fps
        reader.close()
        
        # Parse multiplier from choice
        multiplier_map = {"0x fps": 0, "2x fps": 2, "3x fps": 3, "4x fps": 4}
        fps_multiplier = multiplier_map.get(target_fps, 0)
        
        msg = (f"Starting processing:\n"
               f"• Input: {original_frame_count} frames @ {original_fps}fps ({duration:.2f}s)\n"
               f"• Speed adjustment: {speed_factor:.2f}x\n"
               f"• Frame interpolation: {fps_multiplier}x")
        print(msg)
        console_text = log_to_console(msg, console_text)
        
        # Step 1: Speed Adjustment - affects duration only
        if speed_factor != 1.0:
            msg = f"Applying speed adjustment ({speed_factor}x)..."
            print(msg)
            console_text = log_to_console(msg, console_text)
            
            if speed_factor > 1.0:
                # Speed up: take fewer frames but keep original fps
                step = speed_factor
                indices = np.arange(0, len(video_frames), step)
                video_frames = [video_frames[int(i)] for i in indices if int(i) < len(video_frames)]
            else:
                # Slow down: duplicate frames for smoother slow motion
                new_frame_count = int(len(video_frames) / speed_factor)
                indices = np.linspace(0, len(video_frames) - 1, new_frame_count)
                video_frames = [video_frames[int(i)] for i in indices]
            
            # Speed adjustment doesn't change FPS
            speed_adjusted_fps = original_fps
            speed_adjusted_duration = len(video_frames) / speed_adjusted_fps
            
            msg = (f"• Speed adjustment complete: {len(video_frames)} frames @ {speed_adjusted_fps:.1f}fps\n"
                  f"• New duration: {speed_adjusted_duration:.2f}s")
            print(msg)
            console_text = log_to_console(msg, console_text)
        else:
            speed_adjusted_fps = original_fps
            speed_adjusted_duration = duration
            
        # Step 2: Frame Interpolation - affects FPS only
        if fps_multiplier > 0:
            msg = f"Applying frame interpolation ({fps_multiplier}x)..."
            print(msg)
            console_text = log_to_console(msg, console_text)
            
            result_frames = []
            for i in range(len(video_frames) - 1):
                # Add original frame
                result_frames.append(video_frames[i])
                
                # Generate intermediate frames
                for j in range(fps_multiplier - 1):
                    timestep = (j + 1) / fps_multiplier
                    try:
                        interpolated = interpolate_frames(
                            video_frames[i], 
                            video_frames[i + 1], 
                            timestep
                        )
                        result_frames.append(interpolated)
                    except Exception as e:
                        msg = f"Warning: Frame interpolation failed at frame {i}: {str(e)}"
                        print(msg)
                        console_text = log_to_console(msg, console_text)
                        continue
                        
            # Add final frame
            result_frames.append(video_frames[-1])
            final_fps = speed_adjusted_fps * fps_multiplier
        else:
            result_frames = video_frames
            final_fps = speed_adjusted_fps
            
        # Convert frames if needed
        if hasattr(result_frames[0], 'convert'):
            result_frames = [np.array(frame.convert('RGB')) for frame in result_frames]
        
        # Create output path
        os.makedirs(INTERPOLATED_PATH, exist_ok=True)
        filename = os.path.basename(video_path)
        name, _ = os.path.splitext(filename)
        name = clean_filename(name)  # Remove timestamps but keep processing markers
        
        # Create descriptive filename
        timestamp = datetime.now().strftime("%y%m%d_%H%M")
        speed_text = f"_{speed_factor:.1f}x" if speed_factor != 1.0 else ""
        fps_text = f"_{fps_multiplier}x" if fps_multiplier > 0 else ""
        output_path = os.path.join(INTERPOLATED_PATH, f"{name}{speed_text}{fps_text}_{timestamp}.mp4")
        
        msg = f"Saving processed video to: {output_path}"
        print(msg)
        console_text = log_to_console(msg, console_text)
        
        # First save without audio
        temp_video = output_path + '.temp.mp4'
        imageio.mimwrite(temp_video, result_frames, fps=final_fps, quality=VIDEO_QUALITY)
        
        # If we have audio, combine it with the video
        if audio_path:
            try:
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
                        
                msg = "✓ Audio track processed and merged"
                print(msg)
                console_text = log_to_console(msg, console_text)
                
            except Exception as e:
                msg = f"Warning: Audio processing failed: {str(e)}"
                print(msg)
                console_text = log_to_console(msg, console_text)
                # Fallback to video without audio
                os.rename(temp_video, output_path)
        else:
            # No audio to process, just rename the temp video
            os.rename(temp_video, output_path)
        
        final_frame_count = len(result_frames)
        final_duration = final_frame_count / final_fps
        final_msg = (f"✨ Processing complete!\n"
                    f"• Original: {original_frame_count} frames @ {original_fps}fps ({duration:.2f}s)\n"
                    f"• Final: {final_frame_count} frames @ {final_fps:.1f}fps ({final_duration:.2f}s)")
        print(final_msg)
        console_text = log_to_console(final_msg, console_text)
        return output_path, console_text
        
    except Exception as e:
        msg = f"Error processing video: {str(e)}"
        print(msg)
        console_text = log_to_console(msg, console_text)
        return None, console_text
        

def interpolate_frames(frame1, frame2, timestep):
    """Helper function to interpolate between two frames using RIFE"""
    if not hasattr(interpolate_frames, 'model'):
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
     
     
def process_loop_video(video_path, loop_type="none", num_loops=2, progress=gr.Progress(track_tqdm=True)):
    """Loop video forwards or ping-pong it back and forth"""
    console_text = ""
    
    msg = f"Processing video loop: {video_path}"
    print(msg)
    console_text = log_to_console(msg, console_text)
    
    try:
        # Create output path
        filename = os.path.basename(video_path)
        name, ext = os.path.splitext(filename)
        timestamp = datetime.now().strftime("%y%m%d_%H%M")
        output_path = os.path.join(INTERPOLATED_PATH, f"{name}_{loop_type}_{num_loops}x_{timestamp}.mp4")
        
        if loop_type == "ping-pong":
            # Create palindrome effect and repeat it
            filter_complex = f"[0:v]reverse[r];[0:v][r]concat=n=2:v=1[v];[v]loop={num_loops-1}:32767:0[final]"
            subprocess.run([
                'ffmpeg', '-y',
                '-i', video_path,
                '-filter_complex', filter_complex,
                '-map', '[final]',  # Use the looped output
                '-c:v', 'libx264',
                output_path
            ], capture_output=True)
        else:  # standard loop
            subprocess.run([
                'ffmpeg', '-y',
                '-stream_loop', str(num_loops - 1),  # -1 because first play counts as 1
                '-i', video_path,
                '-c', 'copy',
                output_path
            ], capture_output=True)
            
        msg = f"✨ Loop processing complete: {output_path}"
        print(msg)
        console_text = log_to_console(msg, console_text)
        return output_path, console_text
        
    except Exception as e:
        msg = f"Error creating loop: {str(e)}"
        print(msg)
        console_text = log_to_console(msg, console_text)
        return None, console_text


# managing video naming over repeated runs       
def clean_filename(filename):
    """Remove only timestamp patterns from filename, preserve processing markers"""
    # Remove timestamp patterns (yymmmdd_HHMM)
    filename = re.sub(r'_\d{6}_\d{4}', '', filename)
    return filename       
        
        
        
        
def get_welcome_message():
    return """Welcome to Allegro Text-to-Video!

🎬 What to expect:
• Generation takes about 1 hour per video (on a 3090)
• Output will be 720p at 15fps
• Each video is ~88 frames long
• The initial load time can be quite lengthly, as it (down)loads the massive TE (20GB) from HD to RAM

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
title = """<style>.allegro-banner{background:linear-gradient(to bottom,#162828,#101c1c);color:#fff;padding:0.5rem;border-radius:0.5rem;border:1px solid rgba(255,255,255,0.1);box-shadow:0 4px 6px rgba(0,0,0,0.1);margin-bottom:0.5rem;text-align:center}.allegro-banner h1{font-size:1.75rem;margin:0 0 0.25rem 0;font-weight:300;color:#ff6b35 !important}.allegro-banner p{color:#b0c4c4;font-size:1rem;margin:0 0 0.75rem 0}.allegro-banner .footer{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;font-size:0.875rem;color:#a0a0a0}.allegro-banner .powered-by{display:flex;align-items:center;gap:0.25rem}.allegro-banner .credits{display:flex;flex-direction:column;align-items:center;gap:0.25rem}.allegro-banner a{color:#4a9eff;text-decoration:none;transition:color 0.2s ease}.allegro-banner a:hover{color:#6db3ff}@media (max-width:768px){.allegro-banner .footer{flex-direction:column;gap:0.5rem;align-items:center}}</style><div class="allegro-banner"><h1>Allegro Text-to-Video</h1><p>Transform your prompts to video.</p><div class="footer"><div class="powered-by"><span>⚡ Powered by</span><a href="https://pinokio.computer/" target="_blank">Pinokio</a></div><div class="credits"><div>OG Project: <a href="https://github.com/rhymes-ai/Allegro" target="_blank">Rhymes</a></div><div>Code borrowed from: <a href="https://huggingface.co/spaces/fffiloni/allegro-text2video" target="_blank">fffiloni</a></div></div></div></div>"""


# Create Gradio interface
with gr.Blocks() as demo:
    #gr.HTML(title)
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
                    input_video = gr.Video(label="Upload Video for Interpolation")
                    output_processed = gr.Video(label="Processed Video")

                with gr.Row():
                    process_fps = gr.Radio(
                        choices=["0x fps", "2x fps", "3x fps", "4x fps"],
                        value="0x fps",
                        label="RIFE Frame Interpolation",
                        info="Smooth motion by increasing fps"
                    )
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
                    process_btn = gr.Button("Process Video Interpolation", variant="primary")
                gr.HTML('<hr style="margin: 20px 0; border: none; border-top: 1px solid rgba(128, 128, 128, 0.2);">') 
                
                with gr.Accordion("🔄 Video Loop ~ WIP ~ Feature Creep is Real", open=False):        
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
                        loop_btn = gr.Button("🔄 Create Loop", variant="primary")
                        
                gr.HTML('<hr style="margin: 20px 0; border: none; border-top: 1px solid rgba(128, 128, 128, 0.2);">')            
            with gr.Accordion("Dev Tools", open=False):
                with gr.Row():
                    test_btn = gr.Button("🧪 Two Minute Test Generation", variant="primary")
                  
                    
        with gr.Column():    
            negative_prompt = gr.Textbox(
                value=NEGATIVE_TEMPLATE,
                label="Negative Prompt",
                placeholder="Enter negative prompt",
                lines=2
            )
               
            # Information display section
            with gr.Row():
                info_type = gr.Radio(
                    choices=["welcome", "monitor"],
                    value="welcome",
                    label="Information Display",
                    interactive=True
                )
                open_folder_btn = gr.Button("📁 Open Output Folder")
            
            # Welcome & monitor Info
            with gr.Row():
                status_info = gr.Textbox(
                    label="Status",
                    lines=5,
                    interactive=False,
                    value=get_welcome_message()
                )
            
            # System Messages Console
            with gr.Row():
                console_out = gr.TextArea(
                    label="System Messages",
                    lines=8,
                    interactive=False,
                    autoscroll=True,
                    show_copy_button=True
                )


    # Event handlers
    
    random_seed.click(fn=randomize_seed, outputs=seed)
    
    # Timer that updates system info if monitor view is selected
    timer = gr.Timer(value=2)
    timer.tick(
        fn=lambda display_type: get_system_info() if display_type == "monitor" else status_info.value,
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

    test_btn.click(
        fn=test_inference,
        inputs=[
            user_prompt, 
            negative_prompt, 
            guidance_scale, 
            num_sampling_steps, 
            seed, 
            enable_cpu_offload, 
            target_fps 
        ],
        outputs=[video_output, console_out]
    )
    # manual interpolation
    process_btn.click(
        fn=process_existing_video,
        inputs=[input_video, process_fps, speed_factor],  
        outputs=[output_processed, console_out]
    )
    
    def reset_processing_controls():
        return "0x fps", 1.0  # Default values for process_fps and speed_factor

    # reset post-processing controls
    input_video.change(
        fn=reset_processing_controls,
        inputs=None,
        outputs=[process_fps, speed_factor]
    )
    # loop functions
    loop_btn.click(
        fn=process_loop_video,
        inputs=[input_video, loop_type, num_loops],
        outputs=[output_processed, console_out]
    )
# Launch the interface
demo.launch(share=False)
