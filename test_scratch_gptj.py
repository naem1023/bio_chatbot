from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

prompt = (
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
    "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
    "researchers was the fact that the unicorns spoke perfect English."
)

print(type(prompt))
print(prompt)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
print(input_ids.shape)

model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    max_length=100,
)
gened = tokenizer.decode(gen_tokens)
print(gened)
gen_text = tokenizer.batch_decode(gen_tokens)[0]

print(gen_text)