from SaProt.utils.foldseek_util import get_struc_seq,FoldSeekSetup
pdb_path = "example/8ac8.cif"

# Extract the "A" chain from the pdb file and encode it into a struc_seq
# pLDDT is used to mask low-confidence regions if "plddt_mask" is True. Please set it to True when
# use AF2 structures for best performance.

foldseek=FoldSeekSetup(bin_dir='./foldseek/bin',base_url='https://github.com/steineggerlab/foldseek/releases/download/9-427df8a/').foldseek

print(foldseek)
parsed_seqs = get_struc_seq(foldseek, pdb_path, ["A"], plddt_mask=False)["A"]
seq, foldseek_seq, combined_seq = parsed_seqs

print(f"seq: {seq}")
print(f"foldseek_seq: {foldseek_seq}")
print(f"combined_seq: {combined_seq}")