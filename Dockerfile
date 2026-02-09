# clean base image containing only comfyui, comfy-cli and comfyui-manager
FROM runpod/worker-comfyui:5.5.1-base

# download models into comfyui (note: use /resolve/main/ not /blob/main/ for HuggingFace)
RUN comfy model download --url https://huggingface.co/lightx2v/Qwen-Image-2512-Lightning/resolve/main/qwen_image_2512_fp8_e4m3fn_scaled_comfyui_4steps_v1.0.safetensors --relative-path models/diffusion_models --filename qwen_image_2512_fp8_e4m3fn_scaled_comfyui_4steps_v1.0.safetensors
RUN comfy model download --url https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors --relative-path models/vae --filename qwen_image_vae.safetensors
RUN comfy model download --url https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors --relative-path models/clip --filename qwen_2.5_vl_7b_fp8_scaled.safetensors

# Install Python dependencies for custom handler
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Copy custom handler (overrides base image handler)
COPY handler.py /handler.py

# Run custom handler
CMD ["python", "-u", "/handler.py"]
