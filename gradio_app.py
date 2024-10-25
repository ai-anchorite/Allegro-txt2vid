import torch
import imageio
import os
import gradio as gr
import subprocess
from subprocess import getoutput

from diffusers.schedulers import EulerAncestralDiscreteScheduler
from transformers import T5EncoderModel, T5Tokenizer
from allegro.pipelines.pipeline_allegro import AllegroPipeline
from allegro.models.vae.vae_allegro import AllegroAutoencoderKL3D
from allegro.models.transformers.transformer_3d_allegro import AllegroTransformer3DModel
# from allegro.models.transformers.block import AttnProcessor2_0

from huggingface_hub import snapshot_download

# # Override attention processor initialization
# AttnProcessor2_0.__init__ = lambda self, *args, **kwargs: super(AttnProcessor2_0, self).__init__()

weights_dir = './allegro_weights'
os.makedirs(weights_dir, exist_ok=True)

is_shared_ui = False
is_gpu_associated = torch.cuda.is_available()

# Download weights if not present
if not os.path.exists(weights_dir):
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

def single_inference(user_prompt, save_path, guidance_scale, num_sampling_steps, seed, enable_cpu_offload):
    dtype = torch.float16  # Changed from torch.bfloat16

    # Load models
    vae = AllegroAutoencoderKL3D.from_pretrained(
        "./allegro_weights/vae/", 
        torch_dtype=torch.float32
    ).cuda()
    vae.eval()

    text_encoder = T5EncoderModel.from_pretrained("./allegro_weights/text_encoder/", torch_dtype=dtype)
    text_encoder.eval()

    tokenizer = T5Tokenizer.from_pretrained("./allegro_weights/tokenizer/")

    scheduler = EulerAncestralDiscreteScheduler()

    transformer = AllegroTransformer3DModel.from_pretrained("./allegro_weights/transformer/", torch_dtype=dtype).cuda()
    transformer.eval()

    allegro_pipeline = AllegroPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        transformer=transformer
    ).to("cuda:0")

    positive_prompt = """
    (masterpiece), (best quality), (ultra-detailed), (unwatermarked), 
    {} 
    emotional, harmonious, vignette, 4k epic detailed, shot on kodak, 35mm photo, 
    sharp focus, high budget, cinemascope, moody, epic, gorgeous
    """

    negative_prompt = """
    nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, 
    low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry.
    """

    # Process user prompt
    user_prompt = positive_prompt.format(user_prompt.lower().strip())

    if enable_cpu_offload:
        allegro_pipeline.enable_sequential_cpu_offload()

    # Clear memory before generation
    # torch.cuda.empty_cache()

    out_video = allegro_pipeline(
        user_prompt, 
        negative_prompt=negative_prompt, 
        num_frames=88,
        height=720,
        width=1280,
        num_inference_steps=num_sampling_steps,
        guidance_scale=guidance_scale,
        max_sequence_length=512,
        generator=torch.Generator(device="cuda:0").manual_seed(seed)
    ).video[0]

    # Save video
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.mimwrite(save_path, out_video, fps=15, quality=8)

    return save_path


# Gradio interface function
def run_inference(user_prompt, guidance_scale, num_sampling_steps, seed, enable_cpu_offload, progress=gr.Progress(track_tqdm=True)):
    save_path = "./output_videos/generated_video.mp4"
    result_path = single_inference(user_prompt, save_path, guidance_scale, num_sampling_steps, seed, enable_cpu_offload)
    return result_path

# Create Gradio interface
with gr.Blocks() as demo:
    with gr.Column():
        gr.Markdown("# Allegro Video Generation")
        gr.Markdown("Generate a video based on a text prompt using the Allegro pipeline.")
        
        user_prompt = gr.Textbox(label="User Prompt")
        with gr.Row():
            guidance_scale = gr.Slider(minimum=0, maximum=20, step=0.1, label="Guidance Scale", value=7.5)
            num_sampling_steps = gr.Slider(minimum=10, maximum=100, step=1, label="Number of Sampling Steps", value=20)
        with gr.Row():
            seed = gr.Slider(minimum=0, maximum=10000, step=1, label="Random Seed", value=42)
            enable_cpu_offload = gr.Checkbox(label="Enable CPU Offload", value=True, scale=1)
            
        submit_btn = gr.Button("Generate Video")
        video_output = gr.Video(label="Generated Video")

        gr.Examples(
            examples=[
                ["A Monkey is playing bass guitar."],
                ["An astronaut riding a horse."],
                ["A tiny finch on a branch with spring flowers on background."]
            ],
            inputs=[user_prompt],
            outputs=video_output,
            fn=lambda x: None,
            cache_examples=False
        )

    submit_btn.click(
        fn=run_inference,
        inputs=[user_prompt, guidance_scale, num_sampling_steps, seed, enable_cpu_offload],
        outputs=video_output
    )

# Launch the interface
demo.launch(show_error=True)
