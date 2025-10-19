""" Module with functionalities for blocking based on a dictionary of records,
    where a blocking function must return a dictionary with block identifiers
    as keys and values being sets or lists of record identifiers in that block.
"""

import random
import sys
import numpy as np
import cupy
import cudf
import faiss
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.preprocessing import normalize

# =============================================================================

def noBlocking(rec_dict):
  """A function which does no blocking but simply puts all records from the
     given dictionary into one block.

     Parameter Description:
       rec_dict : Dictionary that holds the record identifiers as keys and
                  corresponding list of record values
  """

  print("Run 'no' blocking:")
  print('  Number of records to be blocked: '+str(len(rec_dict)))
  

  rec_id_list = list(rec_dict.keys())

  block_dict = {'all_rec':rec_id_list}

  return block_dict

# -----------------------------------------------------------------------------

def simpleBlocking(gdf, blk_attr_list):
    """Build the blocking index data structure (dictionary) to store blocking
     key values (BKV) as keys and the corresponding list of record identifiers.

     A blocking is implemented that simply concatenates attribute values.

     Parameter Description:
       gdf (cudf.DataFrame): A cuDF DataFrame containing the records.
       blk_attr_list (list): List of blocking key attributes to use.

     This method returns a dictionary with blocking key values as its keys and
     list of record identifiers as its values (one list for each block).
    """

    block_dict = {}

    print('Run simple blocking:')
    print(f'  List of blocking key attributes: {blk_attr_list}')
    print(f'  Number of records to be blocked: {len(gdf)}')

    gdf = gdf.reset_index()
    gdf = gdf.rename(columns={'index': 'rec_id'})

    # Create the blocking key value by concatenating specified attribute columns
    gdf['bkv'] = ''
    for col in blk_attr_list:
        gdf['bkv'] = gdf['bkv'] + gdf[col].astype(str)

    grouped = gdf.groupby('bkv')
    for name, group in grouped:
        block_dict[name] = group['rec_id'].to_arrow().to_pylist()

    return block_dict

# -----------------------------------------------------------------------------

def phoneticBlocking(rec_dict, blk_attr_list):
  """Build the blocking index data structure (dictionary) to store blocking
     key values (BKV) as keys and the corresponding list of record identifiers.

     A blocking is implemented that concatenates Soundex encoded values of
     attribute values.

     Parameter Description:
       rec_dict      : Dictionary that holds the record identifiers as keys
                       and corresponding list of record values
       blk_attr_list : List of blocking key attributes to use

     This method returns a dictionary with blocking key values as its keys and
     list of record identifiers as its values (one list for each block).
  """

  block_dict = {}  # The dictionary with blocks to be generated and returned

  print('Run phonetic blocking:')
  print('  List of blocking key attributes: '+str(blk_attr_list))
  print('  Number of records to be blocked: '+str(len(rec_dict)))
  
  for (rec_id, rec_values) in rec_dict.items():
    #print(f'record dictionary: {rec_id} and {rec_values}')

    rec_bkv = ''  # Initialise the blocking key value for this record

    # Process selected blocking attributes
    # 
    for attr in blk_attr_list:
      attr_val = rec_values[attr]

      # *********** Implement Soundex function here *********

      # Add your code here
      if attr_val == '' or attr_val is None:
        rec_bkv += 'z000'  # Often used as Soundex code for empty values
      else: 
        attr_val = attr_val.lower()
        sndx_val = attr_val[0] # keep first letter

        for char in attr_val[1:]:
            if char in 'aeiouyhw':
                pass
            elif char in 'bfpv':
                if sndx_val[-1] != '1': # dont add duplicates of digits
                    sndx_val += '1'
            elif char in 'cgjkqsxz':
                if sndx_val[-1] != '2':  # Don't add duplicates of digits
                    sndx_val += '2'
            elif char in 'dt':
                if sndx_val[-1] != '3':  # Don't add duplicates of digits
                    sndx_val += '3'
            elif char in 'l':
                if sndx_val[-1] != '4':  # Don't add duplicates of digits
                    sndx_val += '4'
            elif char in 'mn':
                if sndx_val[-1] != '5':  # Don't add duplicates of digits
                    sndx_val += '5'
            elif char in 'r':
                if sndx_val[-1] != '6':  # Don't add duplicates of digits
                    sndx_val += '6'
        if len(sndx_val) < 4:
                    sndx_val += '000'
        # set max lenth to four 
        sndx_val = sndx_val[:4]
        rec_bkv += sndx_val

      # ************ End of your Soundex code *********************************

    # Insert the blocking key value and record into blocking dictionary
    # 
    if (rec_bkv in block_dict): # Block key value in block index

      # Only need to add the record
      # 
      rec_id_list = block_dict[rec_bkv]
      rec_id_list.append(rec_id)

    else: # Block key value not in block index

      # Create a new block and add the record identifier
      # 
      rec_id_list = [rec_id]

    block_dict[rec_bkv] = rec_id_list  # Store the new block

  return block_dict

# -----------------------------------------------------------------------------

def slkBlocking(rec_dict, fam_name_attr_ind, giv_name_attr_ind, 
                dob_attr_ind, gender_attr_ind):
  """Build the blocking index data structure (dictionary) to store blocking
     key values (BKV) as keys and the corresponding list of record identifiers.

     This function should implement the statistical linkage key (SLK-581)
     blocking approach as used in real-world linkage applications:

     http://www.aihw.gov.au/WorkArea/DownloadAsset.aspx?id=60129551915

     A SLK-581 blocking key is the based on the concatenation of:
     - 3 letters of family name
     - 2 letters of given name
     - Date of birth
     - Sex

     Parameter Description:
       rec_dict          : Dictionary that holds the record identifiers as
                           keys and corresponding list of record values
       fam_name_attr_ind : The number (index) of the attribute that contains
                           family name (last name) 
       giv_name_attr_ind : The number (index) of the attribute that contains
                           given name (first name)
       dob_attr_ind      : The number (index) of the attribute that contains
                           date of birth
       gender_attr_ind   : The number (index) of the attribute that contains
                           gender (sex)

     This method returns a dictionary with blocking key values as its keys and
     list of record identifiers as its values (one list for each block).
  """

  block_dict = {}  # The dictionary with blocks to be generated and returned

  print('Run SLK-581 blocking:')
  print('  Number of records to be blocked: '+str(len(rec_dict)))
  
  for (rec_id, rec_values) in rec_dict.items():

    rec_bkv = ''  # Initialise the blocking key value for this record
 
    # *********** Implement SLK-581 function here ***********

    # Family Name (2nd, 3rd, 5th letters)
    fam_name = rec_values[fam_name_attr_ind]
    slk_fam_name_part = ""
    if not fam_name:
        slk_fam_name_part = "222" # 3 chars for family name
    else:
        alpha_chars_fam = [char for char in fam_name.lower() if char.isalpha()]
        
        # 2nd letter (index 1)
        if 1 < len(alpha_chars_fam):
            slk_fam_name_part += alpha_chars_fam[1]
        else:
            slk_fam_name_part += '2'
            
        # 3rd letter (index 2)
        if 2 < len(alpha_chars_fam):
            slk_fam_name_part += alpha_chars_fam[2]
        else:
            slk_fam_name_part += '2'
            
        # 5th letter (index 4)
        if 4 < len(alpha_chars_fam):
            slk_fam_name_part += alpha_chars_fam[4]
        else:
            slk_fam_name_part += '2'
            
    rec_bkv += slk_fam_name_part.upper()

    # Given Name (2nd, 3rd letters)
    giv_name = rec_values[giv_name_attr_ind]
    slk_giv_name_part = ""
    if not giv_name:
        slk_giv_name_part = "22" # 2 chars for given name
    else:
        alpha_chars_giv = [char for char in giv_name.lower() if char.isalpha()]
        
        # 2nd letter (index 1)
        if 1 < len(alpha_chars_giv):
            slk_giv_name_part += alpha_chars_giv[1]
        else:
            slk_giv_name_part += '2'
            
        # 3rd letter (index 2)
        if 2 < len(alpha_chars_giv):
            slk_giv_name_part += alpha_chars_giv[2]
        else:
            slk_giv_name_part += '2'
            
    rec_bkv += slk_giv_name_part.upper()

    # DoB structure we use: dd/mm/yyyy

    # Get date of birth
    # 
    dob = rec_values[dob_attr_ind]
    if not dob:
        dob = '01/01/1900'

    dob_list = dob.split('/')

    # Add some checks
    # 
    if len(dob_list[0]) < 2:
        dob_list[0] = '0' + dob_list[0]  # Add leading zero for days < 10
    if len(dob_list[1]) < 2:
        dob_list[1] = '0' + dob_list[1]  # Add leading zero for months < 10

    dob = ''.join(dob_list)  # Create: dd/mm/yyyy

    assert len(dob) == 8, dob

    rec_bkv += dob

    # Get gender
    # 
    gender_val = rec_values[gender_attr_ind]
    gender = gender_val.lower() if gender_val else ''

    if gender == 'm':
        rec_bkv += '1'
    elif gender == 'f':
        rec_bkv += '2'
    else:
        rec_bkv += '9'

    # ************ End of your SLK-581 code ***********************************

    # Insert the blocking key value and record into blocking dictionary
    # 
    if (rec_bkv in block_dict): # Block key value in block index

      # Only need to add the record
      # 
      rec_id_list = block_dict[rec_bkv]
      rec_id_list.append(rec_id)

    else: # Block key value not in block index

      # Create a new block and add the record identifier
      # 
      rec_id_list = [rec_id]

    block_dict[rec_bkv] = rec_id_list  # Store the new block

  return block_dict

# -----------------------------------------------------------------------------

# Extra task if you have time:
# - Implement canopy clustering based blocking as described in the lectures
#   and the Data Matching book

def _vectorize_for_faiss_tfidf(gdf, blk_attr_list):
    """
    Vectorizes string attributes of a DataFrame for Faiss using TF-IDF.
    Returns L2-normalized vectors.
    """
    # Combine attributes into a single string per record
    gdf['combined_attrs'] = ''
    for col in blk_attr_list:
        # Fill NA to handle missing values gracefully
        gdf['combined_attrs'] = gdf['combined_attrs'] + gdf[col].fillna('').astype(str).str.lower() + ' '

    # Use TfidfVectorizer to create vectors from q-grams
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 2))
    vectors = vectorizer.fit_transform(gdf['combined_attrs'])

    vectors = normalize(vectors, norm='l2', axis=1).astype('float32')

    return vectors

def canopy_clustering(gdf, blk_attr_list, T1, T2):
    """
    Implements an optimized canopy clustering for blocking records using Faiss ANN.

    This version uses TF-IDF to vectorize string attributes and cosine similarity
    (approximated with L2 distance in Faiss) as the distance measure.
    The Faiss search is performed on the GPU using a knn search to simulate
    a range search.

    Parameters:
        gdf (cudf.DataFrame): The DataFrame containing the records.
        blk_attr_list (list): List of attribute names to use for clustering.
        T1 (float): The loose distance threshold (Jaccard-like distance).
        T2 (float): The tight distance threshold (T2 < T1).

    Returns:
        dict: A dictionary with block identifiers as keys and values being lists of
              record identifiers in that block.
    """
    print("Running optimized canopy clustering with Faiss (GPU knn search)...")
    sys.stdout.flush()

    if gdf.empty:
        print("  Input DataFrame is empty. Returning empty block dictionary.")
        return {}

    block_dict = {}
    
    # Vectorize the records using TF-IDF
    vectors = _vectorize_for_faiss_tfidf(gdf, blk_attr_list)
    num_records, dim = vectors.shape
    
    print(f"  Vectorized {num_records} records into vectors of dimension {dim}.")
    sys.stdout.flush()

    # L2 normalize the vectors for cosine similarity and convert to NumPy
    vectors = normalize(vectors, norm='l2', axis=1).toarray().astype('float32')
    vectors_np = vectors.get() # Convert to NumPy array for Faiss CPU ops

    # Build Faiss index for L2 distance using IndexIVFFlat for GPU support
    nlist = int(np.sqrt(num_records))
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist)

    # Train the index
    print("  Training Faiss index...")
    sys.stdout.flush()
    index.train(vectors_np)

    # Move index to GPU
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index.add(vectors_np)
    gpu_index.nprobe = 10  # Number of cells to visit for search

    print(f"  Built and trained Faiss index on GPU with {nlist} cells.")
    sys.stdout.flush()

    # Convert Jaccard-like distance thresholds (T1, T2) to SQUARED L2 distance
    # thresholds for use with Faiss knn search, which returns squared L2 distances.
    # L2_dist^2 = 2 - 2 * cos_sim. Since T is a distance, sim = 1-T.
    # L2_dist^2 = 2 - 2 * (1-T) = 2T.
    l2_T1_squared = 2 * T1
    l2_T2_squared = 2 * T2

    rec_ids = gdf.index.to_arrow().to_pylist()
    unassigned_rec_indices = set(range(num_records))

    total_records = len(rec_ids)
    progress_step = 10
    next_progress = progress_step

    while unassigned_rec_indices:
        assigned_records = total_records - len(unassigned_rec_indices)
        progress = (assigned_records / total_records) * 100
        if progress >= next_progress:
            print(f"  Canopy clustering progress: {int(progress)}%")
            sys.stdout.flush()
            next_progress += progress_step

        center_index = random.choice(list(unassigned_rec_indices))
        center_vector = np.array([vectors_np[center_index, :]], dtype='float32')

        # Use knn search and filter by radius to simulate range search
        # Set k to a reasonably large number. Cap at 2048.
        k = min(num_records, 2048) #try out different numbers
        D, I = gpu_index.search(center_vector, k)

        # Filter results by the T1 radius (using squared distances)
        canopy_mask = D[0] <= l2_T1_squared
        
        canopy_indices = I[0][canopy_mask]
        canopy_distances_squared = D[0][canopy_mask]

        if len(canopy_indices) > 0:
            # Get the indices of records within the tight threshold T2
            close_indices_mask = canopy_distances_squared <= l2_T2_squared
            close_indices = canopy_indices[close_indices_mask]

            # Remove the close indices from the unassigned set
            # The search result includes the query point itself, so it will be removed.
            unassigned_rec_indices.difference_update(close_indices)

            # Create the block key value from the center attributes
            center_attrs = [gdf[attr].iloc[center_index] for attr in blk_attr_list]
            canopy_bkv = "".join([str(attr) for attr in center_attrs])
            
            # Get the record identifiers for the canopy
            canopy_rec_ids = [rec_ids[i] for i in canopy_indices]

            if canopy_bkv in block_dict:
                block_dict[canopy_bkv].extend(canopy_rec_ids)
            else:
                block_dict[canopy_bkv] = canopy_rec_ids

        else: # No points in canopy, remove center to avoid infinite loop
            unassigned_rec_indices.remove(center_index)


    print(f"  Generated {len(block_dict)} blocks based on canopy clustering.")
    sys.stdout.flush()

    return block_dict

def ann_candidate_generation(recA_gdf, recB_gdf, k, blk_attr_list, sim_threshold=0.5):
    """
    Generates candidate pairs using Approximate Nearest Neighbor (ANN) search with Faiss.

    This approach vectorizes records from two dataframes, indexes one, and searches
    it for the k-nearest neighbors of each record from the other dataframe.

    Parameters:
        recA_gdf (cudf.DataFrame): The first dataframe of records.
        recB_gdf (cudf.DataFrame): The second dataframe of records.
        k (int): The number of nearest neighbors to find for each record in recA_gdf.
        blk_attr_list (list): List of attribute names to use for vectorization.
        sim_threshold (float): Cosine similarity threshold to filter candidate pairs.

    Returns:
        cudf.DataFrame: A DataFrame of candidate pairs ('rec_id_A', 'rec_id_B').
    """
    print(f"Running ANN candidate generation for {len(recA_gdf)} x {len(recB_gdf)} records...")
    sys.stdout.flush()

    if recA_gdf.empty or recB_gdf.empty:
        return cudf.DataFrame({'rec_id_A': [], 'rec_id_B': []})

    # Step 1: Vectorize both datasets using a shared vocabulary
    # Combine attributes into a single string per record for both GDFs
    recA_gdf['combined_attrs'] = ''
    recB_gdf['combined_attrs'] = ''
    for col in blk_attr_list:
        recA_gdf['combined_attrs'] = recA_gdf['combined_attrs'] + recA_gdf[col].fillna('').astype(str).str.lower() + ' '
        recB_gdf['combined_attrs'] = recB_gdf['combined_attrs'] + recB_gdf[col].fillna('').astype(str).str.lower() + ' '

    # Fit the vectorizer on the combined text from both datasets to create a shared vocabulary
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 2))
    vectorizer.fit(cudf.concat([recA_gdf['combined_attrs'], recB_gdf['combined_attrs']]))

    # Transform each dataset separately using the shared vocabulary
    vectors_A = vectorizer.transform(recA_gdf['combined_attrs'])
    vectors_B = vectorizer.transform(recB_gdf['combined_attrs'])

    # Normalize the vectors
    vectors_A = normalize(vectors_A, norm='l2', axis=1).astype('float32')
    vectors_B = normalize(vectors_B, norm='l2', axis=1).astype('float32')

    dim = vectors_A.shape[1]

    vectors_A_np = vectors_A.toarray().get() # to NumPy for Faiss
    vectors_B_np = vectors_B.toarray().get()

    # Step 2: Build and train Faiss index for dataset B
    n_samples_B = len(recB_gdf)
    res = faiss.StandardGpuResources()
    quantizer = None
    index = None
    flat_index = None

    use_ivf = n_samples_B >= 1500
    if use_ivf:
        nlist = max(1, int(np.sqrt(n_samples_B)))
        nlist = min(nlist, max(1, n_samples_B // 2))
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index.train(vectors_B_np)
        gpu_index.add(vectors_B_np)
        gpu_index.nprobe = min(20, max(1, nlist))
    else:
        flat_index = faiss.IndexFlatL2(dim)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, flat_index)
        gpu_index.add(vectors_B_np)

    # Step 3: Search for k-nearest neighbors
    # Squared L2 distance threshold: L2_dist^2 = 2 - 2 * cos_sim
    l2_dist_sq_threshold = 2 - (2 * sim_threshold)
    
    distances, indices = gpu_index.search(vectors_A_np, k)

    # Step 4: Process results to create candidate pairs
    rec_ids_A_series = recA_gdf.index.to_series()
    rec_ids_B_series = recB_gdf.index.to_series()

    pairs_A = cupy.repeat(cupy.arange(len(recA_gdf)), k)
    pairs_B_indices = indices.flatten()
    
    # Filter out invalid indices (-1) from Faiss search
    valid_mask = pairs_B_indices != -1
    pairs_A = pairs_A[valid_mask]
    pairs_B_indices = pairs_B_indices[valid_mask]
    
    # Filter by distance threshold
    dist_mask = distances.flatten()[valid_mask] <= l2_dist_sq_threshold
    pairs_A = pairs_A[dist_mask]
    pairs_B_indices = pairs_B_indices[dist_mask]

    # Convert the numpy array of indices to a cupy array before passing to take()
    pairs_B_indices_gpu = cupy.asarray(pairs_B_indices)

    # Create the two columns for the new DataFrame
    col_A = rec_ids_A_series.take(cupy.ascontiguousarray(pairs_A))
    col_B = rec_ids_B_series.take(pairs_B_indices_gpu)

    # Create the DataFrame, resetting the indices of the source columns to
    # avoid the "Cannot align indices with non-unique values" error.
    candidate_pairs_gdf = cudf.DataFrame({
        'rec_id_A': col_A.reset_index(drop=True),
        'rec_id_B': col_B.reset_index(drop=True)
    })
    
    print(f"  Generated {len(candidate_pairs_gdf)} candidate pairs from ANN search.")
    sys.stdout.flush()

    # free up GPU memory
    del vectors_A, vectors_B, vectors_A_np, vectors_B_np
    del distances, indices
    del pairs_A, pairs_B_indices, pairs_B_indices_gpu
    del col_A, col_B
    if use_ivf:
        del quantizer, index
    else:
        del flat_index
    del gpu_index, res
    import gc
    gc.collect()
    
    return candidate_pairs_gdf

# -----------------------------------------------------------------------------

def merge_block_dicts(dict1, dict2):
    """Merges two block dictionaries."""
    merged_dict = dict1.copy()
    for key, value in dict2.items():
        if key in merged_dict:
            # Use a set to handle duplicates efficiently
            merged_dict[key] = list(set(merged_dict[key] + value))
        else:
            merged_dict[key] = value
    return merged_dict
# -----------------------------------------------------------------------------

def printBlockStatistics(blockA_dict, blockB_dict):
  """Calculate and print some basic statistics about the generated blocks
  """

  print('Statistics of the generated blocks:')

  numA_blocks = len(blockA_dict)
  numB_blocks = len(blockB_dict)

  block_sizeA_list = []
  for rec_id_list in blockA_dict.values():  # Loop over all blocks
    block_sizeA_list.append(len(rec_id_list))

  block_sizeB_list = []
  for rec_id_list in blockB_dict.values():  # Loop over all blocks
    block_sizeB_list.append(len(rec_id_list))

  print('Dataset A number of blocks generated: %d' % (numA_blocks))
  if numA_blocks > 0:
    print('    Minimum block size: %d' % (min(block_sizeA_list)))
    print('    Average block size: %.2f' % \
          (float(sum(block_sizeA_list)) / len(block_sizeA_list)))
    print('    Maximum block size: %d' % (max(block_sizeA_list)))
  else:
    print('    No blocks generated for Dataset A.')
  print('')

  print('Dataset B number of blocks generated: %d' % (numB_blocks))
  if numB_blocks > 0:
    print('    Minimum block size: %d' % (min(block_sizeB_list)))
    print('    Average block size: %.2f' % \
          (float(sum(block_sizeB_list)) / len(block_sizeB_list)))
    print('    Maximum block size: %d' % (max(block_sizeB_list)))
  else:
    print('    No blocks generated for Dataset B.')
  print('')

# -----------------------------------------------------------------------------

# End of program.
