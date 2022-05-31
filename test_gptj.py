from transformers import AutoTokenizer, GPTJForCausalLM, pipeline
import torch


# load fp 16 model
model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

# create pipeline
gen = pipeline("text-generation",model=model,tokenizer=tokenizer,device=0)

# run prediction
gen_result = gen("My Name is philipp")

print(gen_result)