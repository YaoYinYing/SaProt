import os
from SaProt.utils.weights import PretrainedModel

model, tokenizer = PretrainedModel(
    dir=os.path.abspath("./weights/SaProt"), model_name="SaProt_35M_AF2"
).load_model()


device = "cpu"
model.to(device)

seq = "MdEvVpQpLrVyQdYaKv"
tokens = tokenizer.tokenize(seq)
print(tokens)

inputs = tokenizer(seq, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

outputs = model(**inputs)
print(outputs.logits.shape)
