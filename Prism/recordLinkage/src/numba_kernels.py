import numpy as np
from numba import cuda

@cuda.jit
def _jaro_winkler_kernel(arr1, len1, arr2, len2, out, max_len):
    """Compute Jaro-Winkler similarity for each string pair in arr1/arr2."""
    i = cuda.grid(1)
    if i >= arr1.shape[0]:
        return
    l1 = len1[i]
    l2 = len2[i]
    if l1 == 0 or l2 == 0:
        out[i] = 0.0
        return
    match_distance = max(l1, l2) // 2 - 1
    matches = 0
    trans = 0
    for a in range(l1):
        start = 0 if a - match_distance < 0 else a - match_distance
        end = l2 if a + match_distance + 1 > l2 else a + match_distance + 1
        for b in range(start, end):
            if arr1[i, a] == arr2[i, b]:
                matches += 1
                break
    if matches == 0:
        out[i] = 0.0
        return
    minl = l1 if l1 < l2 else l2
    mismatch = 0
    for k in range(minl):
        if arr1[i, k] != arr2[i, k]:
            mismatch += 1
    trans = mismatch
    jaro = (matches / l1 + matches / l2 + (matches - trans/2) / matches) / 3.0
    prefix = 0
    for k in range(min(4, minl)):
        if arr1[i, k] == arr2[i, k]:
            prefix += 1
        else:
            break
    out[i] = jaro + prefix * 0.1 * (1.0 - jaro)


@cuda.jit
def _levenshtein_kernel(arr1, len1, arr2, len2, out, max_len):
    """Compute normalized Levenshtein similarity for each string pair."""
    i = cuda.grid(1)
    if i >= arr1.shape[0]:
        return
    l1 = len1[i]
    l2 = len2[i]
    if l1 == 0 and l2 == 0:
        out[i] = 1.0
        return
    if l1 == 0 or l2 == 0:
        out[i] = 0.0
        return
    prev = cuda.local.array(256, dtype=np.int32)
    curr = cuda.local.array(256, dtype=np.int32)
    for j in range(l2 + 1):
        prev[j] = j
    for a in range(1, l1 + 1):
        curr[0] = a
        ca = arr1[i, a - 1]
        for b in range(1, l2 + 1):
            cost = 0 if ca == arr2[i, b - 1] else 1
            insert = curr[b - 1] + 1
            delete = prev[b] + 1
            replace = prev[b - 1] + cost
            m = insert
            if delete < m:
                m = delete
            if replace < m:
                m = replace
            curr[b] = m
        for j in range(l2 + 1):
            prev[j] = curr[j]
    dist = curr[l2]
    out[i] = 1.0 - float(dist) / float(max(l1, l2))


def calculate_jaro_winkler_pairwise_gpu(d_arrA, d_lenA, d_arrB, d_lenB):
    """Launch `_jaro_winkler_kernel` and return host similarities."""
    n = d_arrA.shape[0]
    out = cuda.device_array(n, dtype=np.float32)
    threadsperblock = 128
    blockspergrid = (n + (threadsperblock - 1)) // threadsperblock
    _jaro_winkler_kernel[blockspergrid, threadsperblock](
        d_arrA, d_lenA, d_arrB, d_lenB, out, d_arrA.shape[1]
    )
    return out.copy_to_host()


def calculate_levenshtein_pairwise_gpu(d_arrA, d_lenA, d_arrB, d_lenB):
    """Launch `_levenshtein_kernel` and return host similarities."""
    n = d_arrA.shape[0]
    out = cuda.device_array(n, dtype=np.float32)
    threadsperblock = 128
    blockspergrid = (n + (threadsperblock - 1)) // threadsperblock
    _levenshtein_kernel[blockspergrid, threadsperblock](
        d_arrA, d_lenA, d_arrB, d_lenB, out, d_arrA.shape[1]
    )
    return out.copy_to_host()

@cuda.jit(device=True)
def _popcount32(x):
    """Return number of set bits in a 32-bit integer."""
    count = 0
    while x:
        x &= x - 1
        count += 1
    return count

@cuda.jit
def _dice_kernel_bitpacked(matA, matB, out):
    """Compute Dice similarity for each row pair of bit-packed matrices."""
    i = cuda.grid(1)
    if i >= matA.shape[0]:
        return
    inter = 0
    countA = 0
    countB = 0
    num_words = matA.shape[1]
    for j in range(num_words):
        # load as unsigned 32-bit
        a = matA[i, j]
        b = matB[i, j]
        # ensure we treat them as uint32 for bit ops — cuda.popc expects integer operand
        # a & b will be typed as uint32 if matA/matB are uint32 device arrays
        inter += cuda.popc(a & b)
        countA += cuda.popc(a)
        countB += cuda.popc(b)
    denom = countA + countB
    if denom == 0:
        out[i] = 0.0
    else:
        out[i] = 2.0 * inter / denom


@cuda.jit
def _jaccard_kernel_bitpacked(matA, matB, out):
    """Compute Jaccard similarity for each row pair of bit-packed matrices."""
    i = cuda.grid(1)
    if i >= matA.shape[0]:
        return
    inter = 0
    union = 0
    num_words = matA.shape[1]
    for j in range(num_words):
        a = matA[i, j]
        b = matB[i, j]
        inter += cuda.popc(a & b)
        union += cuda.popc(a | b)
    if union == 0:
        out[i] = 0.0
    else:
        out[i] = inter / union


def calculate_dice_similarity_gpu_pairwise(d_matA, d_matB):
    """Run `_dice_kernel_bitpacked` over one-to-one row pairs."""
    n = d_matA.shape[0]
    out = cuda.device_array(n, dtype=np.float32)
    threadsperblock = 128
    threadsperblock = min(threadsperblock, max(1, d_matA.shape[0]))
    blockspergrid = (d_matA.shape[0] + threadsperblock - 1) // threadsperblock
    _dice_kernel_bitpacked[blockspergrid, threadsperblock](d_matA, d_matB, out)
    return out.copy_to_host()


def calculate_jaccard_similarity_gpu_pairwise(d_matA, d_matB):
    """Run `_jaccard_kernel_bitpacked` over one-to-one row pairs."""
    n = d_matA.shape[0]
    out = cuda.device_array(n, dtype=np.float32)
    threadsperblock = 128
    threadsperblock = min(threadsperblock, max(1, d_matA.shape[0]))
    blockspergrid = (d_matA.shape[0] + threadsperblock - 1) // threadsperblock
    _jaccard_kernel_bitpacked[blockspergrid, threadsperblock](d_matA, d_matB, out)
    return out.copy_to_host()

@cuda.jit
def _allpairs_dice_bitpacked(matA, matB, out):
    """
    Populate `out` with Dice similarity for every combination of rows.

    matA: (n, W) uint32
    matB: (m, W) uint32
    out:  (n, m) float32  (device array slice)
    Each thread computes one (i,j) pair.
    """
    i, j = cuda.grid(2)
    n = matA.shape[0]
    m = matB.shape[0]
    if i >= n or j >= m:
        return
    num_words = matA.shape[1]
    inter = 0
    countA = 0
    countB = 0
    # iterate over packed words
    for w in range(num_words):
        a = matA[i, w]
        b = matB[j, w]
        # cuda.popc returns int popcount
        inter += cuda.popc(a & b)
        countA += cuda.popc(a)
        countB += cuda.popc(b)
    denom = countA + countB
    if denom == 0:
        out[i, j] = 0.0
    else:
        out[i, j] = 2.0 * inter / denom


@cuda.jit
def _allpairs_jaccard_bitpacked(matA, matB, out):
    """Populate `out` with Jaccard similarity for every combination of rows."""
    i, j = cuda.grid(2)
    n = matA.shape[0]
    m = matB.shape[0]
    if i >= n or j >= m:
        return
    num_words = matA.shape[1]
    inter = 0
    union = 0
    for w in range(num_words):
        a = matA[i, w]
        b = matB[j, w]
        inter += cuda.popc(a & b)
        union += cuda.popc(a | b)
    if union == 0:
        out[i, j] = 0.0
    else:
        out[i, j] = inter / union


def calculate_allpairs_dice_gpu(d_matA, d_matB, batch_m=1024, threads=(16, 16)):
    """Compute full Dice matrix via batched `_allpairs_dice_bitpacked` launches."""
    n = int(d_matA.shape[0])
    m = int(d_matB.shape[0])
    out_full = np.zeros((n, m), dtype=np.float32)

    threads_x, threads_y = threads
    for start in range(0, m, batch_m):
        end = min(m, start + batch_m)
        # slice B on device (it's safe: numba supports device slicing)
        d_B_batch = d_matB[start:end]
        batch_size = end - start
        # allocate device out for this batch
        d_out = cuda.device_array((n, batch_size), dtype=np.float32)
        # compute grid
        blocks_x = (n + threads_x - 1) // threads_x
        blocks_y = (batch_size + threads_y - 1) // threads_y
        _allpairs_dice_bitpacked[(blocks_x, blocks_y), (threads_x, threads_y)](d_matA, d_B_batch, d_out)
        out_full[:, start:end] = d_out.copy_to_host()
    return out_full


def calculate_allpairs_jaccard_gpu(d_matA, d_matB, batch_m=1024, threads=(16, 16)):
    """Compute full Jaccard matrix via batched `_allpairs_jaccard_bitpacked`."""
    n = int(d_matA.shape[0])
    m = int(d_matB.shape[0])
    out_full = np.zeros((n, m), dtype=np.float32)

    threads_x, threads_y = threads
    for start in range(0, m, batch_m):
        end = min(m, start + batch_m)
        d_B_batch = d_matB[start:end]
        batch_size = end - start
        d_out = cuda.device_array((n, batch_size), dtype=np.float32)
        blocks_x = (n + threads_x - 1) // threads_x
        blocks_y = (batch_size + threads_y - 1) // threads_y
        _allpairs_jaccard_bitpacked[(blocks_x, blocks_y), (threads_x, threads_y)](d_matA, d_B_batch, d_out)
        out_full[:, start:end] = d_out.copy_to_host()
    return out_full
