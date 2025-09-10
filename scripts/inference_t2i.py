# -*- coding: utf-8 -*-
"""
Text-to-image inference script
"""
import os
import json
import argparse
import time
import torch
from transformers import AutoConfig, AutoTokenizer
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import SPECIAL_TOKENS
from model import LLaDAForMultiModalGeneration
from utils.generation_utils import setup_seed
from utils.image_utils import decode_vq_to_image, calculate_vq_params, add_break_line
from generators.image_generation_generator import generate_image
from utils.prompt_utils import generate_text_to_image_prompt, create_prompt_templates



def main():
    parser = argparse.ArgumentParser(description="Text-to-image inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Fine-tuned checkpoint path")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--timesteps", type=int, default=64, help="Number of timesteps")
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="CFG scale")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--vae_ckpt", type=str, default="./vae_ckpt", help="VAE checkpoint path")
    parser.add_argument("--output_dir", type=str, default="results_text_to_image", help="Output directory")
    
    args = parser.parse_args()
    
    # Special tokens
    MASK = SPECIAL_TOKENS["mask_token"]
    NEW_LINE = SPECIAL_TOKENS["newline_token"]
    BOA = SPECIAL_TOKENS["answer_start"]  # Begin of Answer
    EOA = SPECIAL_TOKENS["answer_end"]    # End of Answer
    BOI = SPECIAL_TOKENS["boi"]           # Begin of Image
    EOI = SPECIAL_TOKENS["eoi"]           # End of Image

    # Set Random seed
    if args.seed != 0:
        setup_seed(args.seed)
    
    # Create Output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = LLaDAForMultiModalGeneration.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, device_map="auto",
    )
    
    # Load VQ-VAE
    from diffusers import VQModel
    vqvae = VQModel.from_pretrained(args.vae_ckpt, subfolder="vqvae").to(device)
    # Calculate VQ parameters
    seq_len, newline_every, token_grid_height, token_grid_width = calculate_vq_params(args.height, args.width)
    
    print(f"Generate image size: {args.height}x{args.width}")
    print(f"Calculated VQ sequence length: {seq_len}")
    print(f"Tokens per line (newline_every): {newline_every}")
    
    # Get prompt templates
    templates = create_prompt_templates()

    # Get prompt
    prompt_text = args.prompt

    # Generate prompts using utility function
    input_prompt, uncon_prompt = generate_text_to_image_prompt(prompt_text, templates)

    # build initial sequence
    con_prompt_token = tokenizer(input_prompt)["input_ids"]
    uncon_prompt_token = tokenizer(uncon_prompt)["input_ids"]
    
    # build image mask predition
    img_mask_token = add_break_line([MASK] * seq_len, token_grid_height, token_grid_width, new_number = NEW_LINE)
    img_pred_token = [BOA] + [BOI] + img_mask_token + [EOI] + [EOA]

    prompt_ids = torch.tensor(con_prompt_token + img_pred_token, device=device).unsqueeze(0)
    uncon_ids = torch.tensor(uncon_prompt_token, device=device).unsqueeze(0)

    # image satrt index
    code_start = len(con_prompt_token) + 2 
    
    # Generate VQ tokens
    start_time = time.time()
    vq_tokens = generate_image(
        model,
        prompt_ids,
        seq_len=seq_len,
        newline_every=newline_every,
        timesteps=args.timesteps,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        uncon_ids=uncon_ids,
        code_start=code_start
    )
    
    # Generate filename
    words = prompt_text.split()
    filename_words = words[:10] if len(words) > 10 else words
    filename = "_".join(filename_words)
    filename = "".join(c for c in filename if c.isalnum() or c in ('_', '-'))
    filename = f"{filename}_{args.height}x{args.width}_t{args.timesteps}_cfg{args.cfg_scale}_seed{args.seed}.png"
    save_path = os.path.join(args.output_dir, filename)
    
    # Decode VQ codes to PNG and save
    decode_vq_to_image(
        vq_tokens, save_path, 
        vae_ckpt=args.vae_ckpt, 
        image_height=args.height, 
        image_width=args.width,
        vqvae=vqvae
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Time: {elapsed_time:.2f}s")
       


if __name__ == '__main__':
    main()
