from transformers import GPTNeoForCausalLM, GPT2Tokenizer

model_name = "EleutherAI/gpt-neox-20b"
model = GPTNeoForCausalLM.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

prompt = (
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
    "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
    "researchers was the fact that the unicorns spoke perfect English."
)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    max_length=100,
)
gen_text = tokenizer.batch_decode(gen_tokens)[0]

from transformers import GPTNeoXTokenizer, GPTNeoXForCausalLM, GPTNeoXConfig
import torch

tokenizer = GPTNeoXTokenizer.from_pretrained("gpt-neox-20b")
config = GPTNeoXConfig.from_pretrained("gpt-neox-20b")
config.is_decoder = True
model = GPTNeoXForCausalLM.from_pretrained("gpt-neox-20b", config=config)

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

prediction_logits = outputs.logits