from transformers import AutoTokenizer, Llama4ForCausalLM                                                                  
import torch
                                                                                                                           
try:                                                                                                                       
    from modeling_llama4_17x16_kv import Llama4ForCausalLM as CustomLlama4ForCausalLM                                      
except ImportError as e:                                                                                                   
    print(f"Import error: {e}")                                                                                            
    print("Trying alternative import method...")                                                                           
    import modeling_llama4_17x16_kv                                                                                        
    CustomLlama4ForCausalLM = modeling_llama4_17x16_kv.Llama4ForCausalLM                                                   
                                                                                                                           
                                                                                                                           
model_path="/data/eagle_data/shenggui/models/Llama-4-Scout-17B-16E-Instruct"                                               
#model = CustomLlama4ForCausalLM.from_pretrained(model_path)                                                               
# --- Load Tokenizer and Model ---
# The tokenizer is loaded onto the CPU.
tokenizer = AutoTokenizer.from_pretrained(model_path)

# The model is loaded and automatically distributed across all available GPUs.
# Ensure that `torch.cuda.is_available()` is true and `torch.cuda.device_count()` returns 8.
model = CustomLlama4ForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    # torch_dtype=torch.bfloat16  # Using bfloat16 for better performance on compatible hardware.
)

# --- Interactive Loop for Experiments ---
while True:
    # --- User Input ---
    user_input = input("Enter your prompt (or 'quit' to exit): ")
    if user_input.lower() == 'quit':
        break

    # --- Formatting the prompt using a chat template ---
    # This creates a structured conversation history, which is crucial for instruction-tuned models.
    messages = [
        {"role": "system", "content": "You are a friendly chatbot",},
        {"role": "user", "content": user_input}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    breakpoint() 

    # --- Generate a Response ---
    generate_ids = model.generate(
        inputs.input_ids,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        # use_cache=False
    )
    
    # Decode the generated token IDs into text.
    result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # --- Print the Result ---
    print("\n--- Model Output ---")
    print(result)
    print("--------------------\n")

print("Exiting the script.")