import os
import re
import time
import json
import numpy as np
import sys
from typing import Literal
import tarfile
import platform
import subprocess
from dataclasses import dataclass


import pooch
import platformdirs

@dataclass
class FoldSeekSetup:
    """
    A configuration class for the FoldSeek tool, used to determine the 
    download and setup of the FoldSeek binary based on the operating system 
    and CPU architecture.

    Attributes:
        bin_dir (str): Directory where the FoldSeek binary will be stored.
        bin_dir: The directory where the FoldSeek binary is located.
        base_url (str, optional): Base URL from which to download FoldSeek binaries. Defaults to 'https://mmseqs.com/foldseek/'.
        os_build (str, optional): Operating system identifier. Defaults to 'linux'.
        arch (str): The machine's architecture as reported by the platform.uname() function.
        bin_path (str): Path to the FoldSeek binary, constructed by joining bin_dir and 'foldseek'.
    """
    bin_dir: str
    base_url: str = 'https://mmseqs.com/foldseek/'
    

    # Gather system information
    uname = platform.uname()
    cpu_info = subprocess.run(['cat', '/proc/cpuinfo'], stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode('utf8')

    arch: str = uname.machine

    os_build: str = uname.system

    


    def __post_init__(self):
        """
        Post-initialization method that sets up the FoldSeek binary.
        """

        if not os.path.exists(self.bin_dir):
            os.makedirs(self.bin_dir)
        

    @property
    def  bin_path(self) -> str: 
        return os.path.join(self.bin_dir, 'foldseek')

    def build_with_flag(self, flag: Literal['avx2', 'sse2']) -> bool:
        """
        Checks if the CPU supports a given instruction set.

        Args:
            flag (Literal['avx2', 'sse2']): Instruction set to check support for.

        Returns:
            bool: True if the CPU supports the instruction set, False otherwise.
        """
        return flag in self.cpu_info

    @property
    def retrieve_url(self) -> str:
        """
        Determines the download URL for the FoldSeek binary based on the OS and CPU architecture.

        Returns:
            str: The URL to download the appropriate FoldSeek binary.

        Raises:
            NotImplementedError: If the platform or architecture is unsupported.
        """
        if self.os_build == 'Darwin':
            return f'{self.base_url}/foldseek-osx-universal.tar.gz'
        if self.os_build == 'Linux':
            if self.arch == 'x86_64':
                if self.build_with_flag('sse2'):
                    return f'{self.base_url}/foldseek-linux-sse2.tar.gz'
                if self.build_with_flag('avx2'):
                    return f'{self.base_url}/foldseek-linux-avx2.tar.gz'
            if self.arch == 'aarch64':
                return f'{self.base_url}/foldseek-linux-arm64.tar.gz'
        raise NotImplementedError(f'Unsupported platform {self.os_build} or architecture {self.arch}')

    @property
    def foldseek(self) -> str:
        """
        Ensures the FoldSeek binary exists at the specified path.

        Returns:
            str: Path to the FoldSeek binary.

        Raises:
            RuntimeError: If the binary cannot be retrieved.
        """
        if not os.path.exists(self.bin_path):
            self.get_foldseek_binary()
        return self.bin_path

    def get_foldseek_binary(self) -> str:
        """
        Downloads and extracts the FoldSeek binary to the specified directory.

        Returns:
            str: Path to the downloaded and extracted binary.

        Raises:
            RuntimeError: If the binary cannot be retrieved after extraction.
        """
        compressed_file_path = pooch.retrieve(self.retrieve_url, known_hash=None, progressbar=True)
        with tarfile.open(compressed_file_path, 'r:gz') as tar:
            p=platformdirs.user_cache_dir('FoldSeek')
            tar.extractall(path=p)

            os.rename(os.path.join(p,'foldseek','bin', 'foldseek'), self.bin_path)
        
        os.remove(compressed_file_path)
        if not os.path.exists(self.bin_path):
            raise RuntimeError('Could not retrieve foldseek binary')
# Get structural seqs from pdb file
def get_struc_seq(foldseek,
                  path=None,
                  chains: list = None,
                  process_id: int = 0,
                  plddt_mask: bool = False,
                  plddt_threshold: float = 70.,
                  foldseek_verbose: bool = False) -> dict:
    """

    Args:
        foldseek: Binary executable file of foldseek

        path: Path to pdb file

        chains: Chains to be extracted from pdb file. If None, all chains will be extracted.

        process_id: Process ID for temporary files. This is used for parallel processing.

        plddt_mask: If True, mask regions with plddt < plddt_threshold. plddt scores are from the pdb file.

        plddt_threshold: Threshold for plddt. If plddt is lower than this value, the structure will be masked.

        foldseek_verbose: If True, foldseek will print verbose messages.

    Returns:
        seq_dict: A dict of structural seqs. The keys are chain IDs. The values are tuples of
        (seq, struc_seq, combined_seq).
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"PDB file not found: {path}")
    
    if foldseek is None or not os.path.exists(os.path.dirname(foldseek)):
        foldseek_dir= platformdirs.user_cache_dir(appname='SaProt')
    elif os.path.exists(os.path.dirname(foldseek)):
        foldseek_dir=os.path.dirname(foldseek)
        foldseek=FoldSeekSetup(foldseek_dir).foldseek

    
    tmp_save_path = f"get_struc_seq_{process_id}_{time.time()}.tsv"
    if foldseek_verbose:
        cmd = f"{foldseek} structureto3didescriptor --threads 1 --chain-name-mode 1 {path} {tmp_save_path}"
    else:
        cmd = f"{foldseek} structureto3didescriptor -v 0 --threads 1 --chain-name-mode 1 {path} {tmp_save_path}"
    os.system(cmd)
    
    seq_dict = {}
    name = os.path.basename(path)
    with open(tmp_save_path, "r") as r:
        for i, line in enumerate(r):
            desc, seq, struc_seq = line.split("\t")[:3]
            
            # Mask low plddt
            if plddt_mask:
                plddts = extract_plddt(path)
                assert len(plddts) == len(struc_seq), f"Length mismatch: {len(plddts)} != {len(struc_seq)}"
                
                # Mask regions with plddt < threshold
                indices = np.where(plddts < plddt_threshold)[0]
                np_seq = np.array(list(struc_seq))
                np_seq[indices] = "#"
                struc_seq = "".join(np_seq)
            
            name_chain = desc.split(" ")[0]
            chain = name_chain.replace(name, "").split("_")[-1]
            
            if chains is None or chain in chains:
                if chain not in seq_dict:
                    combined_seq = "".join([a + b.lower() for a, b in zip(seq, struc_seq)])
                    seq_dict[chain] = (seq, struc_seq, combined_seq)
    
    os.remove(tmp_save_path)
    os.remove(tmp_save_path + ".dbtype")
    return seq_dict


def extract_plddt(pdb_path: str) -> np.ndarray:
    """
    Extract plddt scores from pdb file.
    Args:
        pdb_path: Path to pdb file.

    Returns:
        plddts: plddt scores.
    """
    with open(pdb_path, "r") as r:
        plddt_dict = {}
        for line in r:
            line = re.sub(' +', ' ', line).strip()
            splits = line.split(" ")
            
            if splits[0] == "ATOM":
                # If position < 1000
                if len(splits[4]) == 1:
                    pos = int(splits[5])
                
                # If position >= 1000, the blank will be removed, e.g. "A 999" -> "A1000"
                # So the length of splits[4] is not 1
                else:
                    pos = int(splits[4][1:])
                
                plddt = float(splits[-2])
                
                if pos not in plddt_dict:
                    plddt_dict[pos] = [plddt]
                else:
                    plddt_dict[pos].append(plddt)
    
    plddts = np.array([np.mean(v) for v in plddt_dict.values()])
    return plddts
