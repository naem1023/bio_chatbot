from turtle import fd
from transformers import GPTNeoXTokenizerFast, GPTNeoXForCausalLM, GPTNeoXConfig
import torch, os, pickle

tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
config = GPTNeoXConfig.from_pretrained("EleutherAI/gpt-neox-20b")
config.is_decoder = True
print("======")
if os.path.exists("o.pickle"):
    with open('o.pickle', 'rb') as f:
        outputs = pickle.load(f)
        
    gen_text = tokenizer.decode(outputs)

    print("======"*100)
else:
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b", config=config)
    # outputs = model(**inputs)
    outputs = model.generate(
        input,
        do_sample=True,
        temperature=0.9,
        max_length=100,
    )
    with open('o.pickle', 'wb') as f:
        pickle.dump(outputs, f, pickle.HIGHEST_PROTOCOL)

    prediction_logits = outputs.logits

    # print(outputs)

    gen_text = tokenizer.batch_decode(outputs)[0]

    # print(gen_text)
    print("======")
print("======")