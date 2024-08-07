import os
from SaProt.model.saprot.saprot_foldseek_mutation_model import (
    SaprotFoldseekMutationModel,
)
from SaProt.utils.foldseek_util import FoldSeekSetup
from SaProt.utils.weights import PretrainedModel

model_loader = PretrainedModel(
    dir=os.path.abspath("./weights/SaProt"),
    model_name="SaProt_650M_AF2",
)

foldseek = FoldSeekSetup(
    bin_dir="./foldseek/bin",
    base_url="https://github.com/steineggerlab/foldseek/releases/download/9-427df8a/",
).foldseek


config = {
    "foldseek_path": foldseek,
    "config_path": model_loader.weights_dir,
    "load_pretrained": True,
}
model = SaprotFoldseekMutationModel(**config)
tokenizer = model.tokenizer

device = "cpu"
model.eval()
model.to(device)

seq = "MdEvVpQpLrVyQdYaKv"

# Predict the effect of mutating the 3rd amino acid to A
mut_info = "V3A"
mut_value = model.predict_mut(seq, mut_info)
print(mut_value)

# Predict all effects of mutations at 3rd position
mut_pos = 3
mut_dict = model.predict_pos_mut(seq, mut_pos)
print(mut_dict)

# Predict probabilities of all amino acids at 3rd position
mut_pos = 3
mut_dict = model.predict_pos_prob(seq, mut_pos)
print(mut_dict)
