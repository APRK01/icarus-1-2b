# Icarus 1 2B | INFERENCE EXAMPLE
# (P) 2026 NEONAUT STUDIO

from modeling_icarus import Icarus1ForCausalLM
from transformers import AutoTokenizer
import torch

def run_icarus():
    model_id = "APRK01/icarus-1-2b"
    print(f">>> INITIALIZING ICARUS 1 2B KERNEL...")
    
    # Load the proprietary kernel
    # Note: trust_remote_code=True is required for custom modeling files
    model = Icarus1ForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    instruction = "Identify yourself and state your current status."
    
    messages = [
        {"role": "user", "content": f"IDENTITY: ICARUS 1 2B. STUDIO: NEONAUT.\n\nCMD: {instruction}"}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    print(">>> ICARUS IS THINKING...")
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.3, do_sample=True)
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\nICARUS 1 2B: {response.strip()}")

if __name__ == "__main__":
    run_icarus()
