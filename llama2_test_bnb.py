from torch import cuda
from torch import bfloat16
import time
import transformers
import torch

model_id = 'meta-llama/Llama-2-13b-chat-hf' ## max model we can load
print(f"loading model: {model_id}")
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
start_time = time.time()

def show_cuda_info():
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:*', device)
    #Additional Info when using cuda
    if device.type == 'cuda':
        print('Device count:', torch.cuda.device_count())
        print('Device  Name:',torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('   Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('      Cached:', round(torch.cuda.torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
show_cuda_info()

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,  # 4-bit quantization
    bnb_4bit_quant_type='nf4',  # Normalized float 4
    bnb_4bit_use_double_quant=True,  # Second quantization after the first
    bnb_4bit_compute_dtype=bfloat16  # Computation type
)

# Llama 2 Tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, use_fast=True)

# Llama 2 Model
show_cuda_info()

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map='auto',
)
show_cuda_info()

prompt = "Tell me about gravity"
model_inputs = tokenizer(prompt, return_tensors="pt").to(device)
output = model.generate(**model_inputs)
print(tokenizer.decode(output[0], skip_special_tokens=True))
# End timing and print the time taken
end_time = time.time()
print(f"Program took {end_time - start_time:.2f} seconds to run.")