import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Starting test...")

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)

print("MODEL LOADED SUCCESSFULLY!")

question = "What are common risk factors for heart disease?"

messages = [
    {
        "role": "system",
        "content": "You are a helpful medical information assistant. Give general educational information only."
    },
    {
        "role": "user",
        "content": question
    }
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print("Tokenizing...")

inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

print("Starting generation...")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

print("GENERATION COMPLETED!")

answer = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True
)

print("\nANSWER:")
print(answer)

print("\nTEST FINISHED!")