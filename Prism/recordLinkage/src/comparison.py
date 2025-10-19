"""GPU-accelerated comparison utilities used throughout the linkage pipeline."""

import os
import numpy as np
import sys
from collections import Counter
import cudf
import cupy
from .numba_kernels import (
    calculate_jaccard_similarity_gpu_pairwise,
    calculate_dice_similarity_gpu_pairwise,
    calculate_jaro_winkler_pairwise_gpu,
    calculate_levenshtein_pairwise_gpu,
)

from . import config
from numba import cuda
from rapidfuzz.distance import Levenshtein

MAX_STRING_LEN = 256

def _series_to_padded_uint8(series, max_len=MAX_STRING_LEN):
    """Convert cuDF string Series to numpy uint8 padded array and lengths.
    Returns (arr, lengths) where arr.shape = (n, max_len), dtype=uint8 and
    lengths.shape = (n,), dtype=int32.
    """
    if len(series) == 0:
        return np.empty((0, max_len), dtype=np.uint8), np.empty((0,), dtype=np.int32)
    # Convert to Python list of bytes
    py_list = series.fillna('').to_arrow().to_pylist()
    n = len(py_list)
    arr = np.zeros((n, max_len), dtype=np.uint8)
    lengths = np.zeros((n,), dtype=np.int32)
    for i, s in enumerate(py_list):
        if s is None:
            continue
        if isinstance(s, str):
            b = s.encode('utf8', errors='ignore')[:max_len]
        else:
            # bytes-like
            b = bytes(s)[:max_len]
        arr[i, :len(b)] = np.frombuffer(b, dtype=np.uint8)
        lengths[i] = len(b)
    return arr, lengths


@cuda.jit
def gpu_jaro_winkler(s1, s2, out):
    """Numba CUDA kernel to compute Jaro-Winkler similarity for two series of strings."""
    i = cuda.grid(1)
    if i < s1.shape[0]:
        out[i] = jaro_winkler_similarity_kernel(s1[i], s2[i])

@cuda.jit
def gpu_levenshtein(s1, s2, out):
    """Numba CUDA kernel wrapper for Levenshtein similarity (minimal safe wrapper).
       Note: the full device-level Levenshtein kernel is defined as a device function
       above (levenshtein_similarity_kernel). This wrapper simply calls that device
       function for each row. This wrapper is kept to avoid silent failures if someone
       accidentally triggers the GPU path, but the production pipeline below chooses
       the CPU fallback for stability.
    """
    i = cuda.grid(1)
    if i < s1.shape[0]:
        out[i] = levenshtein_similarity_kernel(s1[i], s2[i])

def get_char_vocab_ascii_map(s1, s2):
    """
    Creates a mapping from ASCII value to an integer code for use in a Numba kernel.
    Returns a CuPy array where index=ASCII_code and value=integer_code.
    """
    # Combine series and get unique characters on GPU
    s = cudf.concat([s1, s2])
    unique_chars_series = s.str.character_tokenize().dropna().unique()
    
    # Pull unique characters to CPU to build map (small, one-time cost)
    unique_chars_cpu = unique_chars_series.to_pandas()
    
    # Create a map from ascii code -> integer code.
    # Using 256 to cover the extended ASCII range.
    # 0 is reserved for padding. Codes start from 1.
    ascii_map = np.zeros(256, dtype=np.int32)
    for i, char in enumerate(unique_chars_cpu):
        if len(char) == 1:
            ascii_val = ord(char)
            if ascii_val < 256:
                ascii_map[ascii_val] = i + 1
                
    return cupy.asarray(ascii_map)

@cuda.jit
def _fill_char_arrays_kernel(strings_chars, strings_offsets, ascii_to_code_map, output_array, max_len):
    """
    Numba kernel to convert a cuDF string column into a dense 2D array of integer codes.
    """
    i = cuda.grid(1)
    if i >= len(strings_offsets) - 1:
        return

    # Get start and end position of the string in the character buffer
    start = strings_offsets[i]
    end = strings_offsets[i+1]
    length = end - start

    # Loop through characters of the string
    for j in range(length):
        if j >= max_len:
            break
        
        # Get the byte value of the character
        char_byte = strings_chars[start + j]
        
        # Look up the integer code from the ASCII map
        # Note: Assumes character bytes are < 256
        if char_byte < 256:
            code = ascii_to_code_map[char_byte]
            output_array[i, j] = code
        # Characters outside the map will remain 0 (padding value)

def strings_to_char_arrays_gpu(s, ascii_to_code_map, max_len=256):
    """
    Wrapper function to convert a cuDF string Series to a 2D CuPy array
    of character codes using a Numba kernel.
    """
    num_strings = len(s)
    if num_strings == 0:
        return cupy.empty((0, max_len), dtype=cupy.int32)

    # Allocate output array on GPU, initialized to 0 (padding value)
    output_array = cupy.zeros((num_strings, max_len), dtype=cupy.int32)

    # Get the underlying character and offset arrays from the string column
    str_col = s._column

    # Handle edge case where a chunk contains only empty strings for an attribute
    if len(str_col.children) < 2:
        # This means there are no characters to process, so return the zero-filled array
        return output_array

    strings_chars = str_col.children[1].values
    strings_offsets = str_col.children[0].values

    # Configure and launch the kernel
    threads_per_block = 256
    blocks_per_grid = (num_strings + threads_per_block - 1) // threads_per_block
    
    _fill_char_arrays_kernel[blocks_per_grid, threads_per_block](
        strings_chars,
        strings_offsets,
        ascii_to_code_map,
        output_array,
        max_len
    )
    
    return output_array

@cuda.jit(device=True)
def jaro_winkler_similarity_kernel(s1, s2):
    """Numba CUDA kernel for Jaro-Winkler similarity."""
    len1 = len(s1)
    len2 = len(s2)

    if len1 == 0 or len2 == 0:
        return 0.0

    match_distance = (max(len1, len2) // 2) - 1

    s1_matches = cuda.local.array(MAX_STRING_LEN, dtype=np.bool_)
    s2_matches = cuda.local.array(MAX_STRING_LEN, dtype=np.bool_)

    for i in range(len1):
        s1_matches[i] = False
    for i in range(len2):
        s2_matches[i] = False

    matches = 0
    transpositions = 0

    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)

        for j in range(start, end):
            if not s2_matches[j] and s1[i] == s2[j]:
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if s1_matches[i]:
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1

    jaro_sim = (matches / len1 + matches / len2 + (matches - transpositions // 2) / matches) / 3.0

    # Winkler modification
    prefix = 0
    for i in range(min(len1, len2, 4)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return jaro_sim + prefix * 0.1 * (1.0 - jaro_sim)

@cuda.jit(device=True)
def levenshtein_similarity_kernel(s1, s2):
    """Numba CUDA kernel for Levenshtein similarity."""
    len1 = len(s1)
    len2 = len(s2)

    if len1 == 0 or len2 == 0:
        return 0.0

    if len1 < len2:
        s1, s2 = s2, s1
        len1, len2 = len2, len1

    prev_row = cuda.local.array(MAX_STRING_LEN, dtype=np.int32)
    curr_row = cuda.local.array(MAX_STRING_LEN, dtype=np.int32)

    for i in range(len2 + 1):
        prev_row[i] = i

    for i in range(1, len1 + 1):
        curr_row[0] = i
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr_row[j] = min(curr_row[j - 1] + 1,
                                prev_row[j] + 1,
                                prev_row[j - 1] + cost)
        for j in range(len2 + 1):
            prev_row[j] = curr_row[j]

    return 1.0 - (curr_row[len2] / len1)






""" Module with functionalities for comparison of attribute values as well as
    record pairs. The record pair comparison function will return a dictionary
    of the compared pairs to be used for classification.
"""

Q = 2    # Value length of q-grams for Jaccard and Dice comparison function

def get_q_grams(s, q):
    """Generate a set of q-grams (substrings of length q) from a string."""
    return {s[i:i+q] for i in range(len(s) - q + 1)}


def _digits_only(val):
    """Return only decimal digits from *val* while handling missing inputs."""
    if val is None:
        return ''
    if isinstance(val, bytes):
        try:
            val = val.decode('utf-8', errors='ignore')
        except Exception:
            return ''
    return ''.join(ch for ch in str(val) if '0' <= ch <= '9')


def _safe_int(val):
    """Return an integer extracted from *val* or ``None`` if conversion fails."""
    digits = _digits_only(val)
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def gender_comp(val1, val2):
    """Exact comparison on gender (case-insensitive first character match)."""
    if not val1 or not val2:
        return 0.0
    g1 = str(val1).strip().lower()
    g2 = str(val2).strip().lower()
    if not g1 or not g2:
        return 0.0
    return 1.0 if g1[0] == g2[0] else 0.0


def date_digits_comp(val1, val2):
    """Compare dates by their digit-only representation (e.g., 19800504)."""
    d1 = _digits_only(val1)
    d2 = _digits_only(val2)
    if len(d1) < 6 or len(d2) < 6:
        return 0.0
    return 1.0 if d1 == d2 else 0.0


def postcode_exact_comp(val1, val2):
    """Exact comparison for postcode after stripping whitespace."""
    p1 = _digits_only(val1)
    p2 = _digits_only(val2)
    if len(p1) < 3 or len(p2) < 3:
        return 0.0
    return 1.0 if p1 == p2 else 0.0


def phone_suffix_comp(val1, val2, min_digits=7):
    """Compare phone numbers using their digit-only suffix of length *min_digits*."""
    d1 = _digits_only(val1)
    d2 = _digits_only(val2)
    if len(d1) < min_digits or len(d2) < min_digits:
        return 0.0
    return 1.0 if d1[-min_digits:] == d2[-min_digits:] else 0.0


def age_similarity_comp(val1, val2, max_diff=12):
    """Compare ages by their absolute difference with a linear decay.

    Returns 1.0 for identical ages, decreases linearly until ``max_diff`` years
    apart where the score reaches 0.0. Missing or non-numeric values yield 0.0.
    """
    age1 = _safe_int(val1)
    age2 = _safe_int(val2)
    if age1 is None or age2 is None:
        return 0.0
    diff = abs(age1 - age2)
    if diff == 0:
        return 1.0
    if diff >= max_diff:
        return 0.0
    return max(0.0, 1.0 - (diff / float(max_diff)))

# =============================================================================
# First the basic functions to compare attribute values

def exact_comp(val1, val2):
    """Compare the two given attribute values exactly, return 1 if they are the
         same (but not both empty!) and 0 otherwise.
    """

    # If at least one of the values is empty return 0
    #
    if val1 is None or val2 is None or (len(val1) == 0) or (len(val2) == 0):
        return 0.0

    elif (val1 != val2):
        return 0.0
    else:    # The values are the same
        return 1.0

# -----------------------------------------------------------------------------


def jaccard_comp(val1, val2):
    """Calculate the Jaccard similarity between the two given attribute values
         by extracting sets of sub-strings (q-grams) of length q.

         Returns a value between 0.0 and 1.0.
    """

    # If at least one of the values is empty return 0
    #
    if val1 is None or val2 is None or (len(val1) == 0) or (len(val2) == 0):
        return 0.0

    # If both attribute values exactly match return 1
    #
    elif (val1 == val2):
        return 1.0

    # ********* Implement Jaccard similarity function here *********

    jacc_sim = 0.0    # Replace with your code

    q_grams_val1 = get_q_grams(val1, Q) 
    q_grams_val2 = get_q_grams(val2, Q)

    # Handle cases where q_grams might be empty 
    if not q_grams_val1 and not q_grams_val2:
                return 1.0 
    if not q_grams_val1 or not q_grams_val2:
                return 0.0 

    numerator = len(q_grams_val1.intersection(q_grams_val2))
    denominator = len(q_grams_val1.union(q_grams_val2))
    jacc_sim = float(numerator) / denominator

    # ************ End of your Jaccard code *************************************

    assert jacc_sim >= 0.0 and jacc_sim <= 1.0

    return jacc_sim

def jaccard_distance(val1, val2):
  """Calculate the Jaccard distance between the two given attribute values.
  """

  return 1.0 - jaccard_comp(val1, val2)

def jaccard_comp_gpu(vals1, vals2):
    """
    Calculate the Jaccard similarity between two cuDF Series of strings on the GPU.
    """
    
    vals1_list = vals1.to_arrow().to_pylist()
    vals2_list = vals2.to_arrow().to_pylist()

    sets1 = [get_q_grams_set(s) for s in vals1_list]
    sets2 = [get_q_grams_set(s) for s in vals2_list]
    
    sims = calculate_jaccard_similarity_gpu_pairwise(sets1, sets2)
    
    return cudf.Series(sims)

# -----------------------------------------------------------------------------


def dice_comp(val1, val2):
    """Calculate the Dice coefficient similarity between the two given attribute
         values by extracting sets of sub-strings (q-grams) of length q.

         Returns a value between 0.0 and 1.0.
    """

    # If at least one of the values is empty return 0
    #
    if val1 is None or val2 is None or (len(val1) == 0) or (len(val2) == 0):
        return 0.0

    # If both attribute values exactly match return 1
    #
    elif (val1 == val2):
        return 1.0

    # ********* Implement Dice similarity function here *********

    dice_sim = 0.0    # Replace with your code

    q_grams_val1 = get_q_grams(val1, Q)
    q_grams_val2 = get_q_grams(val2, Q)
    
    if not q_grams_val1 and not q_grams_val2:
        return 1.0
    if not q_grams_val1 or not q_grams_val2:
        return 0.0
       
    numerator = 2 * len(q_grams_val1.intersection(q_grams_val2))
    denominator = len(q_grams_val1) + len(q_grams_val2)
    dice_sim = float(numerator)/denominator 

    # ************ End of your Dice code ****************************************

    assert dice_sim >= 0.0 and dice_sim <= 1.0

    return dice_sim

# -----------------------------------------------------------------------------


JARO_MARKER_CHAR = chr(1)    # Special character used in the Jaro, Winkler comp.

def jaro_comp(val1, val2):
    """Calculate the similarity between the two given attribute values based on
        the Jaro comparison function.

         As described in 'An Application of the Fellegi-Sunter Model of Record
         Linkage to the 1990 U.S. Decennial Census' by William E. Winkler and Yves
         Thibaudeau.

         Returns a value between 0.0 and 1.0.
    """

    # If at least one of the values is empty return 0
    #
    if (val1 == '') or (val2 == ''):
        return 0.0

    # If both attribute values exactly match return 1
    #
    elif (val1 == val2):
        return 1.0

    len1 = len(val1)    # Number of characters in val1
    len2 = len(val2)    # Number of characters in val2

    halflen = int(max(len1, len2) / 2) - 1

    assingment1 = ''    # Characters assigned in val1
    assingment2 = ''    # Characters assigned in val2

    workstr1 = val1    # Copy of original value1
    workstr2 = val2    # Copy of original value2

    common1 = 0    # Number of common characters
    common2 = 0    # Number of common characters

    for i in range(len1):    # Analyse the first string
        start = max(0, i - halflen)
        end     = min(i + halflen + 1, len2)
        index = workstr2.find(val1[i], start, end)
        if (index > -1):        # Found common character, count and mark it as assigned
            common1 += 1
            assingment1 = assingment1 + val1[i]
            workstr2 = workstr2[:index] + JARO_MARKER_CHAR + workstr2[index+1:]

    for i in range(len2):    # Analyse the second string
        start = max(0, i - halflen)
        end     = min(i + halflen + 1, len1)
        index = workstr1.find(val2[i], start, end)
        if (index > -1):        # Found common character, count and mark it as assigned
            common2 += 1
            assingment2 = assingment2 + val2[i]
            workstr1 = workstr1[:index] + JARO_MARKER_CHAR + workstr1[index+1:]

    if (common1 != common2):
        common1 = float(common1 + common2) / 2.0

    if (common1 == 0):        # No common characters within half length of strings
        return 0.0

    transposition = 0    # Calculate number of transpositions

    for i in range(len(assingment1)):
        if (assingment1[i] != assingment2[i]):
            transposition += 1
    transposition = transposition / 2.0

    common1 = float(common1)

    jaro_sim = 1./3.*(common1 / float(len1) + common1 / float(len2) + \
                     (common1 - transposition) / common1)

    assert (jaro_sim >= 0.0) and (jaro_sim <= 1.0), \
                            'Similarity weight outside 0-1: %f' % (jaro_sim)

    return jaro_sim

# -----------------------------------------------------------------------------


def jaro_winkler_comp(val1, val2):
    """Calculate the similarity between the two given attribute values based on
         the Jaro-Winkler modifications.

         Applies the Winkler modification if the beginning of the two strings is
         the same.

         As described in 'An Application of the Fellegi-Sunter Model of Record
         Linkage to the 1990 U.S. Decennial Census' by William E. Winkler and Yves
         Thibaudeau.

         If the beginning of the two strings (up to first four characters) are the
         same, the similarity weight will be increased.

         Returns a value between 0.0 and 1.0.
    """

    # If at least one of the values is empty return 0
    #
    if (val1 == '') or (val2 == ''):
        return 0.0

    # If both attribute values exactly match return 1
    #
    elif (val1 == val2):
        return 1.0

    # First calculate the basic Jaro similarity
    #
    jaro_sim = jaro_comp(val1, val2)
    if (jaro_sim == 0):
        return 0.0    # No common characters

    # ********* Implement Winkler similarity function here *********
    
    strings = [val1, val2]
    p = min(len(os.path.commonprefix(strings)), 4) # Cap prefix length at 4
    jw_sim = jaro_sim + (1 - jaro_sim) * (p/10)

    # ************ End of your Winkler code *************************************

    assert (jw_sim >= jaro_sim), 'Winkler modification is negative'
    assert (jw_sim >= 0.0) and (jw_sim <= 1.0), \
                 'Similarity weight outside 0-1: %f' % (jw_sim)

    return jw_sim

# -----------------------------------------------------------------------------


def bag(s):
    count = Counter(s)
    return count

def bag_dist_sim_comp(val1, val2):
    """Calculate the bag distance similarity between the two given attribute
         values.

         Returns a value between 0.0 and 1.0.
    """

    # If at least one of the values is empty return 0
    #
    if val1 is None or val2 is None or (len(val1) == 0) or (len(val2) == 0):
        return 0.0

    # If both attribute values exactly match return 1
    #
    elif (val1 == val2):
        return 1.0

    # ********* Implement bag similarity function here *********
    # Extra task only

    bag_sim = 0.0    # Replace with your code
    s1 = bag(val1)
    s2 = bag(val2)
    difference_size = max(len(s1.keys() - s2.keys()), len(s2.keys() - s1.keys()))
    bag_sim = 1.0 - difference_size / max(len(val1), len(val2))
    
    # ************ End of your bag distance code ********************************

    assert bag_sim >= 0.0 and bag_sim <= 1.0

    return bag_sim

# -----------------------------------------------------------------------------


def edit_dist_sim_comp(val1, val2):
    if val1 is None or val2 is None or (len(val1) == 0) or (len(val2) == 0):
        return 0.0
    if val1 == val2:
        return 1.0
    # normalized Levenshtein
    dist = Levenshtein.distance(val1, val2)
    maxlen = max(len(val1), len(val2))
    return 1.0 - float(dist) / float(maxlen)


    return arr, lengths

def _series_to_bitpacked_qgrams_gpu(series, q=Q):
    """Convert cuDF string Series to CuPy array of bit-packed q-grams."""
    if len(series) == 0:
        return cupy.empty((0, 1), dtype=cupy.uint32) # Return empty array with correct dtype

    # Convert cuDF Series to Python list of strings
    py_list = series.fillna('').to_arrow().to_pylist()

    # Generate q-grams for each string
    qgrams_list = [get_q_grams(s, q) for s in py_list]

    # Create a vocabulary of all unique q-grams
    all_qgrams = set().union(*qgrams_list)
    qgram_to_id = {qg: i for i, qg in enumerate(sorted(list(all_qgrams)))}

    # Determine the number of uint32 words needed for bit-packing
    num_qgrams = len(qgram_to_id)
    num_words = (num_qgrams + 31) // 32 # Each uint32 can hold 32 bits

    # Create a CuPy array for bit-packed q-grams
    bitpacked_qgrams_cupy = cupy.zeros((len(py_list), num_words), dtype=cupy.uint32)

    # Populate the bit-packed array
    for i, qgrams_set in enumerate(qgrams_list):
        for qg in qgrams_set:
            qg_id = qgram_to_id[qg]
            word_idx = qg_id // 32
            bit_idx = qg_id % 32
            bitpacked_qgrams_cupy[i, word_idx] |= (1 << bit_idx)

    return bitpacked_qgrams_cupy

def prepare_sets(listA, listB, q=Q):
    setsA_bitpacked = _series_to_bitpacked_qgrams_gpu(listA, q)
    setsB_bitpacked = _series_to_bitpacked_qgrams_gpu(listB, q)
    return setsA_bitpacked, setsB_bitpacked


def jaro_winkler_comp_gpu(listA, listB):
    arrA, lenA = _series_to_padded_uint8(listA)
    arrB, lenB = _series_to_padded_uint8(listB)
    return calculate_jaro_winkler_pairwise_gpu(arrA, lenA, arrB, lenB)

def dice_comp_gpu(listA, listB):
    setsA, setsB = prepare_sets(listA, listB, q=Q)
    return calculate_dice_similarity_gpu_pairwise(setsA, setsB)

def jaccard_comp_gpu(listA, listB):
    setsA, setsB = prepare_sets(listA, listB, q=Q)
    return calculate_jaccard_similarity_gpu_pairwise(setsA, setsB)

def levenshtein_comp_gpu(listA, listB):
    arrA, lenA = _series_to_padded_uint8(listA)
    arrB, lenB = _series_to_padded_uint8(listB)
    return calculate_levenshtein_pairwise_gpu(arrA, lenA, arrB, lenB)

def run_exact_comp_gpu(listA, listB):
    """Convenience wrapper used by tests to run the exact comparison on GPU data."""
    colA = listA.fillna('')
    colB = listB.fillna('')
    return (colA == colB).astype('float32')

def run_jaccard_gpu(listA, listB, q=Q):
    """Helper for tests: compute Jaccard similarity using the GPU kernels."""
    colA = listA.fillna('')
    colB = listB.fillna('')
    setsA, setsB = prepare_sets(colA, colB, q=q)
    sims = calculate_jaccard_similarity_gpu_pairwise(setsA, setsB)
    return cudf.Series(sims)

def run_dice_gpu(listA, listB, q=Q):
    """Helper for tests: compute Dice similarity using the GPU kernels."""
    colA = listA.fillna('')
    colB = listB.fillna('')
    setsA, setsB = prepare_sets(colA, colB, q=q)
    sims = calculate_dice_similarity_gpu_pairwise(setsA, setsB)
    return cudf.Series(sims)

# -----------------------------------------------------------------------------


# =============================================================================
# Function to compare a block

def _process_chunk(pairs_chunk, recA_gdf_renamed, recB_gdf_renamed, attr_comp_list, chunk_num):
    """Helper function to process a single chunk of candidate pairs."""

    pairs_gdf = cudf.DataFrame(pairs_chunk, columns=['rec_id_A', 'rec_id_B'])

    # 1. Get unique record IDs from the current chunk
    unique_ids_A = pairs_gdf['rec_id_A'].unique()
    unique_ids_B = pairs_gdf['rec_id_B'].unique()

    # 2. Select only the necessary records from the main DataFrames
    chunk_recA_gdf = recA_gdf_renamed.loc[unique_ids_A]
    chunk_recB_gdf = recB_gdf_renamed.loc[unique_ids_B]

    # 3. Now, merge with these much smaller, filtered DataFrames
    merged_gdf = pairs_gdf.merge(chunk_recA_gdf, left_on='rec_id_A', right_index=True, how='left')
    merged_gdf = merged_gdf.merge(chunk_recB_gdf, left_on='rec_id_B', right_index=True, how='left')

    # 4. Apply comparisons
    print(f'  Comparing attribute values for candidate pairs chunk {chunk_num} (using native cudf and custom kernels where possible)...')
    sys.stdout.flush()
    
    gpu_attrs_char = set()
    gpu_attrs_qgram = set()
    # Map both CPU and GPU function variants to the same buffer-preparation logic.
    char_gpu_funcs = {
        jaro_winkler_comp,
        edit_dist_sim_comp,
        jaro_winkler_comp_gpu,
        levenshtein_comp_gpu,
    }
    qgram_gpu_funcs = {
        jaccard_comp,
        dice_comp,
        jaccard_comp_gpu,
        dice_comp_gpu,
    }
    for comp_funct, aA, aB in attr_comp_list:
        if comp_funct in char_gpu_funcs:
            gpu_attrs_char.add((aA, aB))
        if comp_funct in qgram_gpu_funcs:
            gpu_attrs_qgram.add((aA, aB))

    char_gpu_buffers = {}
    qgram_gpu_buffers = {}

    if config.USE_GPU_COMPARISON:
        for aA, aB in gpu_attrs_char:
            colA = merged_gdf[aA + '_A'].fillna('')
            colB = merged_gdf[aB + '_B'].fillna('')
            arrA, lenA = _series_to_padded_uint8(colA, max_len=MAX_STRING_LEN)
            arrB, lenB = _series_to_padded_uint8(colB, max_len=MAX_STRING_LEN)
            # Transfer to device
            d_arrA = cuda.to_device(arrA)
            d_lenA = cuda.to_device(lenA)
            d_arrB = cuda.to_device(arrB)
            d_lenB = cuda.to_device(lenB)
            char_gpu_buffers[(aA, aB)] = (d_arrA, d_lenA, d_arrB, d_lenB)

        for aA, aB in gpu_attrs_qgram:
            colA = merged_gdf[aA + '_A'].fillna('')
            colB = merged_gdf[aB + '_B'].fillna('')
            d_matA, d_matB = prepare_sets(colA, colB, q=Q)
            qgram_gpu_buffers[(aA, aB)] = (d_matA, d_matB)

    sim_vectors_list = []

    for comp_funct, attr_nameA, attr_nameB in attr_comp_list:
        col_A = merged_gdf[attr_nameA + '_A'].fillna('')
        col_B = merged_gdf[attr_nameB + '_B'].fillna('')

        if comp_funct == exact_comp:
            sim_col = (col_A == col_B).astype('float32')

        elif config.USE_GPU_COMPARISON and comp_funct in (jaccard_comp, jaccard_comp_gpu):
            try:
                d_matA, d_matB = qgram_gpu_buffers[(attr_nameA, attr_nameB)]
                sims = calculate_jaccard_similarity_gpu_pairwise(d_matA, d_matB)
                sim_col = cudf.Series(sims)
            except Exception:
                print('    NOTE: GPU path failed for Jaccard; using CPU fallback.')
                sys.stdout.flush()
                s_A = col_A.to_pandas()
                s_B = col_B.to_pandas()
                sim_list = [jaccard_comp(v1, v2) for v1, v2 in zip(s_A, s_B)]
                sim_col = cudf.Series(sim_list, nan_as_null=False)

        elif config.USE_GPU_COMPARISON and comp_funct in (dice_comp, dice_comp_gpu):
            try:
                d_matA, d_matB = qgram_gpu_buffers[(attr_nameA, attr_nameB)]
                sims = calculate_dice_similarity_gpu_pairwise(d_matA, d_matB)
                sim_col = cudf.Series(sims)
            except Exception:
                print('    NOTE: GPU path failed for Dice; using CPU fallback.')
                sys.stdout.flush()
                s_A = col_A.to_pandas()
                s_B = col_B.to_pandas()
                sim_list = [dice_comp(v1, v2) for v1, v2 in zip(s_A, s_B)]
                sim_col = cudf.Series(sim_list, nan_as_null=False)

        elif config.USE_GPU_COMPARISON and comp_funct in (jaro_winkler_comp, jaro_winkler_comp_gpu):
            try:
                d_arrA, d_lenA, d_arrB, d_lenB = char_gpu_buffers[(attr_nameA, attr_nameB)]
                sims = calculate_jaro_winkler_pairwise_gpu(d_arrA, d_lenA, d_arrB, d_lenB)
                sim_col = cudf.Series(sims)
            except Exception:
                print('    NOTE: GPU path failed for Jaro-Winkler; using CPU fallback.')
                sys.stdout.flush()
                s_A = col_A.to_pandas()
                s_B = col_B.to_pandas()
                sim_list = [jaro_winkler_comp(v1, v2) for v1, v2 in zip(s_A, s_B)]
                sim_col = cudf.Series(sim_list, nan_as_null=False)

        elif config.USE_GPU_COMPARISON and comp_funct in (edit_dist_sim_comp, levenshtein_comp_gpu):
            try:
                d_arrA, d_lenA, d_arrB, d_lenB = char_gpu_buffers[(attr_nameA, attr_nameB)]
                sims = calculate_levenshtein_pairwise_gpu(d_arrA, d_lenA, d_arrB, d_lenB)
                sim_col = cudf.Series(sims)
            except Exception:
                print('    NOTE: GPU path failed for Levenshtein; using CPU fallback.')
                sys.stdout.flush()
                s_A = col_A.to_pandas()
                s_B = col_B.to_pandas()
                sim_list = [edit_dist_sim_comp(v1, v2) for v1, v2 in zip(s_A, s_B)]
                sim_col = cudf.Series(sim_list, nan_as_null=False)

        else: # Fallback for CPU or if GPU is disabled
            print(f"    Processing '{comp_funct.__name__}' on CPU.")
            sys.stdout.flush()
            s_A = col_A.to_pandas()
            s_B = col_B.to_pandas()
            sim_list = [comp_funct(v1, v2) for v1, v2 in zip(s_A, s_B)]
            sim_col = cudf.Series(sim_list, nan_as_null=False)

        try:
            sample_vals = sim_col.head(5).to_pandas().tolist()
            print(f'    sample sim ({attr_nameA}->{attr_nameB}): {sample_vals}')
            sys.stdout.flush()
        except Exception:
            pass

        sim_vectors_list.append(sim_col.astype('float32'))

    # 4. Assemble the final DataFrame for the chunk
    sim_vectors_gdf = cudf.concat(sim_vectors_list, axis=1)
    sim_column_names = []
    for func_idx, (_, attr_nameA, attr_nameB) in enumerate(attr_comp_list):
        if attr_nameA == attr_nameB:
            sim_column_names.append(f'sim_{attr_nameA}')
        else:
            sim_column_names.append(f'sim_{attr_nameA}_{attr_nameB}')

    sim_vectors_gdf.columns = sim_column_names
    sim_vectors_gdf['rec_id_A'] = merged_gdf['rec_id_A']
    sim_vectors_gdf['rec_id_B'] = merged_gdf['rec_id_B']

    # Clean up memory
    del pairs_gdf, merged_gdf
    import gc
    gc.collect()

    return sim_vectors_gdf

def compareBlocks(blockA_dict, blockB_dict, recA_gdf, recB_gdf, attr_comp_list):
    """Build a similarity dictionary with pair of records from the two given
     block dictionaries using a vectorized GPU approach with CPU fallback for
     unsupported functions.
    """

    print(f'Vectorizing {len(blockA_dict)} blocks from dataset A with {len(blockB_dict)} blocks from dataset B')
    sys.stdout.flush()

    all_sim_vectors_gdf = []
    chunk_size = 5_000_000  # Process 5,000,000 pairs at a time to manage memory
    pair_buffer = []
    chunk_num = 1

    recA_gdf_renamed = recA_gdf.add_suffix('_A')
    recB_gdf_renamed = recB_gdf.add_suffix('_B')

    for block_bkv, rec_idA_list in blockA_dict.items():
        if block_bkv in blockB_dict:
            rec_idB_list = blockB_dict[block_bkv]
            for rec_idA in rec_idA_list:
                for rec_idB in rec_idB_list:
                    pair_buffer.append((rec_idA, rec_idB))

                    if len(pair_buffer) >= chunk_size:
                        sim_vectors_chunk = _process_chunk(pair_buffer, recA_gdf_renamed, recB_gdf_renamed, attr_comp_list, chunk_num)
                        all_sim_vectors_gdf.append(sim_vectors_chunk)
                        pair_buffer = []
                        chunk_num += 1

    # Process any remaining pairs in the buffer
    if pair_buffer:
        sim_vectors_chunk = _process_chunk(pair_buffer, recA_gdf_renamed, recB_gdf_renamed, attr_comp_list, chunk_num)
        all_sim_vectors_gdf.append(sim_vectors_chunk)

    if not all_sim_vectors_gdf:
        print('  No candidate pairs found after blocking.')
        return cudf.DataFrame()

    final_sim_vectors_gdf = cudf.concat(all_sim_vectors_gdf, ignore_index=True)

    print(f'  Compared {len(final_sim_vectors_gdf)} record pairs')
    print('')
    sys.stdout.flush()

    # Clean up the list of chunked dataframes
    del all_sim_vectors_gdf
    import gc
    gc.collect()

    return final_sim_vectors_gdf

def compare_pairs(pairs_gdf, recA_gdf, recB_gdf, attr_comp_list):
    """
    Build a similarity dictionary for a given DataFrame of candidate pairs
    using a vectorized GPU approach.
    """
    print(f'Comparing {len(pairs_gdf)} candidate pairs...')
    sys.stdout.flush()

    if pairs_gdf.empty:
        return cudf.DataFrame()

    all_sim_vectors_gdf = []
    BATCH_SIZE = 1_000_000  # Process 1,000,000 pairs at a time
    chunk_num = 1

    recA_gdf_renamed = recA_gdf.add_suffix('_A')
    recB_gdf_renamed = recB_gdf.add_suffix('_B')

    for i in range(0, len(pairs_gdf), BATCH_SIZE):
        # Chunk the pair list to limit the size of the intermediate cudf frame.
        pairs_chunk_gdf = pairs_gdf.iloc[i:i + BATCH_SIZE]
        
        # Convert just the chunk to a list of tuples for _process_chunk
        pair_buffer_chunk = [tuple(x) for x in pairs_chunk_gdf.to_records(index=False)]
        
        sim_vectors_chunk = _process_chunk(pair_buffer_chunk, recA_gdf_renamed, recB_gdf_renamed, attr_comp_list, chunk_num)
        all_sim_vectors_gdf.append(sim_vectors_chunk)
        chunk_num += 1

    if not all_sim_vectors_gdf:
        print('  No similarity vectors generated.')
        return cudf.DataFrame()

    final_sim_vectors_gdf = cudf.concat(all_sim_vectors_gdf, ignore_index=True)

    # Clean up the list of chunked dataframes
    del all_sim_vectors_gdf
    import gc
    gc.collect()

    print(f'  Finished comparing {len(final_sim_vectors_gdf)} record pairs')
    print('')
    sys.stdout.flush()

    return final_sim_vectors_gdf
# -----------------------------------------------------------------------------

# End of program.

