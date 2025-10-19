"""
GPU similarity test harness
===========================

Runs small sample data through the GPU-based pairwise string similarity kernels:
- Jaro–Winkler
- Levenshtein
- Dice
- Jaccard

This helps verify that GPU kernels produce realistic, non-constant similarity values.
"""

import numpy as np
from numba import cuda
from numba_kernels import (
    calculate_jaro_winkler_pairwise_gpu,
    calculate_levenshtein_pairwise_gpu,
    calculate_dice_similarity_gpu_pairwise,
    calculate_jaccard_similarity_gpu_pairwise,
)

def build_bit_matrix(sets, vocab):
    """Converts a list of sets into a binary bit matrix."""
    n = len(sets)
    vocab_size = len(vocab)
    mat = np.zeros((n, vocab_size), dtype=np.uint8)
    token_to_idx = {tok: i for i, tok in enumerate(vocab)}
    for i, s in enumerate(sets):
        for tok in s:
            if tok in token_to_idx:
                mat[i, token_to_idx[tok]] = 1
    return mat

def pack_bits_uint32(mat):
    """
    Pack binary matrix (n, V) into (n, W) uint32 where W = ceil(V/32)
    """
    n, vocab_size = mat.shape
    num_words = (vocab_size + 31) // 32
    packed = np.zeros((n, num_words), dtype=np.uint32)
    for i in range(n):
        row = mat[i]
        # iterate bits; vectorization is possible but simple loop is fine for build step
        for bit_idx in range(vocab_size):
            if row[bit_idx]:
                word = bit_idx // 32
                bit = bit_idx % 32
                packed[i, word] |= (np.uint32(1) << np.uint32(bit))
    return packed

# Example: produce all-pairs similarities
def allpairs_example(samples_A, samples_B, q=2, batch_m=1024, threads=(16,16)):
    # prepare q-gram sets
    setsA, setsB = prepare_sets(samples_A, samples_B, q=q)
    vocab = sorted(set().union(*setsA, *setsB))
    matA = build_bit_matrix(setsA, vocab)      # (n, V) uint8
    matB = build_bit_matrix(setsB, vocab)
    # pack to uint32
    packedA = pack_bits_uint32(matA).astype(np.uint32, copy=False)
    packedB = pack_bits_uint32(matB).astype(np.uint32, copy=False)
    # transfer to GPU
    d_matA = cuda.to_device(packedA)
    d_matB = cuda.to_device(packedB)
    # compute all-pairs (may take memory/time depending on sizes)
    from numba_kernels import calculate_allpairs_dice_gpu, calculate_allpairs_jaccard_gpu
    dice_matrix = calculate_allpairs_dice_gpu(d_matA, d_matB, batch_m=batch_m, threads=threads)
    jacc_matrix = calculate_allpairs_jaccard_gpu(d_matA, d_matB, batch_m=batch_m, threads=threads)
    return dice_matrix, jacc_matrix

# Example usage inside main()
def main_allpairs_demo():
    samples_A = ["john", "michael", "kathryn", "steven", "lucy"]
    samples_B = ["jon", "micheal", "katherine", "stephen", "lucie"]
    dice_mat, jacc_mat = allpairs_example(samples_A, samples_B, q=2, batch_m=128, threads=(8,8))
    print("Dice all-pairs:\n", dice_mat)
    print("Jaccard all-pairs:\n", jacc_mat)

    # Example: top-1 match for each A based on Dice
    best_idx = np.argmax(dice_mat, axis=1)
    for i, bi in enumerate(best_idx):
        print(f"{samples_A[i]} -> {samples_B[bi]} (Dice={dice_mat[i, bi]:.3f}, Jaccard={jacc_mat[i,bi]:.3f})")

def _series_to_padded_uint8(py_list, max_len=64):
    """Converts a list of strings into a padded uint8 array for GPU kernels."""
    n = len(py_list)
    arr = np.zeros((n, max_len), dtype=np.uint8)
    lengths = np.zeros((n,), dtype=np.int32)
    for i, s in enumerate(py_list):
        if s is None:
            continue
        if isinstance(s, str):
            b = s.encode("utf8", errors="ignore")[:max_len]
        else:
            b = bytes(s)[:max_len]
        arr[i, : len(b)] = np.frombuffer(b, dtype=np.uint8)
        lengths[i] = len(b)
    return arr, lengths


def get_q_grams(s, q=2):
    """Simple q-gram extraction helper."""
    if s is None:
        return set()
    return {s[i : i + q] for i in range(len(s) - q + 1)}


def prepare_sets(listA, listB, q=2):
    """Converts lists of strings into lists of q-gram sets."""
    setsA = [get_q_grams(s, q) for s in listA]
    setsB = [get_q_grams(s, q) for s in listB]
    return setsA, setsB


def main():
    samples_A = ["john", "michael", "kathryn", "steven", "lucy"]
    samples_B = ["jon", "micheal", "katherine", "stephen", "lucie"]

    # --- Jaro-Winkler and Levenshtein (byte-padded) ---
    arrA, lenA = _series_to_padded_uint8(samples_A)
    arrB, lenB = _series_to_padded_uint8(samples_B)

    d_arrA = cuda.to_device(arrA)
    d_arrB = cuda.to_device(arrB)
    d_lenA = cuda.to_device(lenA)
    d_lenB = cuda.to_device(lenB)

    jw_scores = calculate_jaro_winkler_pairwise_gpu(d_arrA, d_lenA, d_arrB, d_lenB)
    lv_scores = calculate_levenshtein_pairwise_gpu(d_arrA, d_lenA, d_arrB, d_lenB)

    # --- Dice and Jaccard (bit-packed GPU) ---
    setsA, setsB = prepare_sets(samples_A, samples_B)
    vocab = sorted(set().union(*setsA, *setsB))

    matA = build_bit_matrix(setsA, vocab)
    matB = build_bit_matrix(setsB, vocab)

# Bit-pack to uint32 blocks
    packedA = pack_bits_uint32(matA)
    packedB = pack_bits_uint32(matB)
    packedA = packedA.astype(np.uint32, copy=False)
    packedB = packedB.astype(np.uint32, copy=False)

# Transfer to GPU
    d_matA = cuda.to_device(packedA)
    d_matB = cuda.to_device(packedB)

    dice_scores = calculate_dice_similarity_gpu_pairwise(d_matA, d_matB)
    jacc_scores = calculate_jaccard_similarity_gpu_pairwise(d_matA, d_matB)

    print("\nGPU kernel test results:")
    print("=========================")
    print(f"{'String A':12s}  {'String B':12s}  JW      LV      Dice    Jaccard")
    print("-" * 65)
    for i in range(len(samples_A)):
        print(
            f"{samples_A[i]:12s}  {samples_B[i]:12s}  "
            f"{jw_scores[i]:6.3f}  {lv_scores[i]:6.3f}  "
            f"{dice_scores[i]:6.3f}  {jacc_scores[i]:6.3f}"
        )


if __name__ == "__main__":
    main()

