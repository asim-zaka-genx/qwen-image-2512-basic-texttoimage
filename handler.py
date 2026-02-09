"""
RunPod Serverless Handler for Qwen Image 2512 ComfyUI Worker
"""

import runpod
import json
import urllib.request
import urllib.error
import time
import os
import base64
import subprocess
import threading
import uuid

# Configuration
COMFY_HOST = "127.0.0.1"
COMFY_PORT = 8188
COMFY_API_URL = f"http://{COMFY_HOST}:{COMFY_PORT}"
COMFY_OUTPUT_DIR = "/comfyui/output"
COMFY_INPUT_DIR = "/comfyui/input"

# Polling settings
COMFY_API_AVAILABLE_INTERVAL_MS = int(os.environ.get("COMFY_API_AVAILABLE_INTERVAL_MS", 500))
COMFY_POLLING_INTERVAL_MS = int(os.environ.get("COMFY_POLLING_INTERVAL_MS", 250))
COMFY_POLLING_MAX_RETRIES = int(os.environ.get("COMFY_POLLING_MAX_RETRIES", 500))


def start_comfyui():
    """Start ComfyUI server in background"""
    print("Starting ComfyUI server...")
    subprocess.Popen(
        ["python", "main.py", "--listen", COMFY_HOST, "--port", str(COMFY_PORT)],
        cwd="/comfyui",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )


def wait_for_comfyui(timeout=120):
    """Wait for ComfyUI API to become available"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            urllib.request.urlopen(f"{COMFY_API_URL}/system_stats", timeout=5)
            print("ComfyUI is ready!")
            return True
        except (urllib.error.URLError, urllib.error.HTTPError):
            time.sleep(COMFY_API_AVAILABLE_INTERVAL_MS / 1000)
    print("ComfyUI failed to start within timeout")
    return False


def queue_prompt(workflow):
    """Queue a workflow prompt to ComfyUI"""
    data = json.dumps({"prompt": workflow}).encode("utf-8")
    req = urllib.request.Request(
        f"{COMFY_API_URL}/prompt",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    try:
        response = urllib.request.urlopen(req)
        result = json.loads(response.read().decode("utf-8"))
        return result.get("prompt_id")
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        raise Exception(f"Failed to queue prompt: {error_body}")


def get_history(prompt_id):
    """Get execution history for a prompt"""
    try:
        response = urllib.request.urlopen(f"{COMFY_API_URL}/history/{prompt_id}")
        return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError:
        return {}


def poll_for_completion(prompt_id):
    """Poll until workflow execution completes"""
    retries = 0
    while retries < COMFY_POLLING_MAX_RETRIES:
        history = get_history(prompt_id)
        
        if prompt_id in history:
            status = history[prompt_id].get("status", {})
            
            # Check for errors
            if status.get("status_str") == "error":
                messages = status.get("messages", [])
                error_msg = messages[-1] if messages else "Unknown error"
                raise Exception(f"Workflow execution failed: {error_msg}")
            
            # Check for completion
            if history[prompt_id].get("outputs"):
                return history[prompt_id]
        
        time.sleep(COMFY_POLLING_INTERVAL_MS / 1000)
        retries += 1
    
    raise Exception("Polling timeout: workflow did not complete in time")


def get_output_images(history):
    """Extract output images from workflow history"""
    images = []
    outputs = history.get("outputs", {})
    
    for node_id, node_output in outputs.items():
        if "images" in node_output:
            for img_info in node_output["images"]:
                filename = img_info.get("filename")
                subfolder = img_info.get("subfolder", "")
                img_type = img_info.get("type", "output")
                
                # Construct image path
                if subfolder:
                    img_path = os.path.join(COMFY_OUTPUT_DIR, subfolder, filename)
                else:
                    img_path = os.path.join(COMFY_OUTPUT_DIR, filename)
                
                if os.path.exists(img_path):
                    with open(img_path, "rb") as f:
                        img_data = base64.b64encode(f.read()).decode("utf-8")
                    
                    images.append({
                        "filename": filename,
                        "type": "base64",
                        "data": img_data
                    })
    
    return images


def upload_input_images(images):
    """Upload base64 images to ComfyUI input directory"""
    uploaded = []
    
    for idx, img in enumerate(images):
        if isinstance(img, dict):
            img_data = img.get("data", img.get("image", ""))
            img_name = img.get("name", f"input_{uuid.uuid4().hex[:8]}.png")
        else:
            img_data = img
            img_name = f"input_{uuid.uuid4().hex[:8]}.png"
        
        # Decode base64 and save
        try:
            img_bytes = base64.b64decode(img_data)
            img_path = os.path.join(COMFY_INPUT_DIR, img_name)
            
            with open(img_path, "wb") as f:
                f.write(img_bytes)
            
            uploaded.append(img_name)
            print(f"Uploaded input image: {img_name}")
        except Exception as e:
            print(f"Failed to upload image {idx}: {e}")
    
    return uploaded


def inject_prompt_into_workflow(workflow, prompt, negative_prompt="", seed=None):
    """Inject prompt text into workflow's CLIPTextEncode nodes"""
    modified = json.loads(json.dumps(workflow))  # Deep copy
    
    for node_id, node in modified.items():
        class_type = node.get("class_type", "")
        
        # Update positive prompt (node 6 typically)
        if class_type == "CLIPTextEncode":
            inputs = node.get("inputs", {})
            current_text = inputs.get("text", "")
            
            # Check if this is positive or negative prompt node
            # Usually negative prompt nodes have empty or negative keywords
            if current_text == "" or any(neg in current_text.lower() for neg in ["blurry", "bad", "ugly", "distorted"]):
                if negative_prompt:
                    inputs["text"] = negative_prompt
            else:
                if prompt:
                    inputs["text"] = prompt
        
        # Update seed in KSampler
        if class_type == "KSampler" and seed is not None:
            node.get("inputs", {})["seed"] = seed
    
    return modified


def handler(event):
    """
    Main handler for RunPod serverless requests
    
    Expected input format:
    {
        "workflow": { ... ComfyUI API workflow JSON ... },
        "prompt": "optional text prompt to inject",
        "negative_prompt": "optional negative prompt",
        "seed": 12345,  # optional
        "images": [     # optional input images
            {"name": "image.png", "data": "base64..."}
        ]
    }
    """
    try:
        job_input = event.get("input", {})
        
        # Validate input
        workflow = job_input.get("workflow")
        if not workflow:
            return {"error": "No workflow provided in input"}
        
        # Handle optional prompt injection
        prompt = job_input.get("prompt")
        negative_prompt = job_input.get("negative_prompt", "")
        seed = job_input.get("seed")
        
        if prompt:
            workflow = inject_prompt_into_workflow(workflow, prompt, negative_prompt, seed)
        elif seed is not None:
            # Just update seed if provided without prompt
            for node_id, node in workflow.items():
                if node.get("class_type") == "KSampler":
                    node.get("inputs", {})["seed"] = seed
        
        # Handle input images if provided
        input_images = job_input.get("images", [])
        if input_images:
            uploaded_names = upload_input_images(input_images)
            print(f"Uploaded {len(uploaded_names)} input images")
        
        # Queue the workflow
        print("Queueing workflow...")
        prompt_id = queue_prompt(workflow)
        print(f"Queued with prompt_id: {prompt_id}")
        
        # Poll for completion
        print("Waiting for completion...")
        history = poll_for_completion(prompt_id)
        print("Workflow completed!")
        
        # Extract output images
        images = get_output_images(history)
        print(f"Generated {len(images)} images")
        
        if not images:
            return {"error": "No images were generated"}
        
        return {
            "output": {
                "images": images,
                "prompt_id": prompt_id
            }
        }
        
    except Exception as e:
        print(f"Handler error: {str(e)}")
        return {"error": str(e)}


# Initialize ComfyUI on cold start
print("Initializing worker...")
threading.Thread(target=start_comfyui, daemon=True).start()

if not wait_for_comfyui(timeout=120):
    print("WARNING: ComfyUI may not be ready")

# Start the serverless handler
runpod.serverless.start({"handler": handler})
