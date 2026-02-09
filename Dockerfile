# clean base image containing only comfyui, comfy-cli and comfyui-manager
FROM runpod/worker-comfyui:5.5.1-base

# install custom nodes into comfyui (first node with --mode remote to fetch updated cache)
RUN comfy node install --exit-on-fail comfyui_resolutionselectorplus@2.0.5 --mode remote
RUN comfy node install --exit-on-fail rgthree-comfy@1.0.2512112053
# The following custom nodes are in an unknown registry and have no aux_id provided, so they could not be resolved and were skipped:
# GetNode
# SetNode

# download models into comfyui
RUN comfy model download --url https://huggingface.co/lightx2v/Qwen-Image-2512-Lightning/blob/main/qwen_image_2512_fp8_e4m3fn_scaled_comfyui_4steps_v1.0.safetensors --relative-path models/diffusion_models --filename qwen_image_2512_fp8_e4m3fn_scaled_comfyui_4steps_v1.0.safetensors
RUN comfy model download --url https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors --relative-path models/vae --filename qwen_image_vae.safetensors
RUN comfy model download --url https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors --relative-path models/clip --filename qwen_2.5_vl_7b_fp8_scaled.safetensors

# copy all input data (like images or videos) into comfyui (uncomment and adjust if needed)
# COPY input/ /comfyui/input/
