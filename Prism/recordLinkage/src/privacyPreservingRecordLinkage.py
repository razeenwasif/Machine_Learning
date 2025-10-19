# ============================================================================
# Privacy-preserving Record linkage module for the COMP3430/COMP8430
# Data Wrangling course, 2025.
# Version 1.0
#
# Copyright (C) 2025 the Australian National University and
# others. All Rights Reserved.
#
# =============================================================================

"""Module for linking records from two files in a privacy preserving manner.

   This module uses the Bloom filter based privacy-preserving technique to 
   perform the record linkage. This module calls the necessary modules to 
   perform the functionalities of the steps in the linkage process.
"""

# =============================================================================
# Import necessary modules (Python standard modules first, then other modules)

import time
import hashlib
import random 
import bitarray # This module is used to generate Bloom filters
from . import loadDataset
from . import evaluation

# =============================================================================
# Variables

# ******** Uncomment to select a pair of datasets **************

datasetA_name = 'datasets/clean-A-1000.csv'
datasetB_name = 'datasets/clean-B-1000.csv'

#datasetA_name = 'datasets/little-dirty-A-10000.csv.gz'
#datasetB_name = 'datasets/little-dirty-B-10000.csv.gz'

headerA_line   = True  # Dataset A header line available - True or Flase
headerB_line   = True  # Dataset B header line available - True or Flase

# Name of the corresponding file with true matching record pair

# ***** Uncomment a file name corresponding to your selected datasets *******

truthfile_name = 'datasets/clean-true-matches-1000.csv'

#truthfile_name = 'datasets/little-dirty-true-matches-10000.csv.gz'

# The two attribute numbers that contain the record identifiers
#
rec_idA_col = 0
rec_idB_col = 0

# The list of attributes to be used either for blocking or linking
#
# For the example data sets used in COMP8430 data wrangling:
# 
#  0: rec_id
#  1: first_name
#  2: middle_name
#  3: last_name
#  4: gender
#  5: current_age
#  6: birth_date
#  7: street_address
#  8: suburb
#  9: postcode
# 10: state
# 11: phone
# 12: email

attrA_list    = [1,2,3,4,6,7,8,9,10,11]
attrB_list    = [1,2,3,4,6,7,8,9,10,11]

BF_HASH_FUNCT1 = hashlib.sha1 # Hash function 1 used for Bloom filter encoding
BF_HASH_FUNCT2 = hashlib.md5  # Hash function 2 used for Bloom filter encoding

# =============================================================================
# Function implementations

# -----------------------------------------------------------------------------
# Function for generating q-gram sets for datasets
# -----------------------------------------------------------------------------
def gen_q_gram_dict(rec_attr_val_dict, q):
  """Generate a set of q-grams for each of the given attribute value lists 
     and return a dictionary with q-gram sets as values.

     Parameter Description:
       - rec_attr_val_dict  A dictionary where keys are record ids and values
                            are lists of attribute values for records.
       - q                  The length of a q-gram.

     This method returns an output,
       - rec_q_gram_dict    The dictionary of q-gram sets.
  """

  rec_q_gram_dict = {}  # The dictionary to store the q-gram sets.

  qm1 = q - 1

  # Iterate through each attribute value list in the dictionary
  for rec_id in rec_attr_val_dict:

    # Get the attribute value list of the record
    rec_val_list = rec_attr_val_dict[rec_id]

    q_gram_set = set()

    # Iterate through each attribute value in the attribute value list
    for attr_val in rec_val_list:

      # Generate set of q-grams for the attribute value
      #
      attr_val_len = len(attr_val)
      attr_q_gram_set = set([attr_val[i:i+q] 
                             for i in range(attr_val_len - qm1)]) 

      # Add the attribute q-gram set into the record q-gram set
      #
      q_gram_set = q_gram_set.union(attr_q_gram_set)

    # Add the generated q-gram set into the dictionary
    #
    rec_q_gram_dict[rec_id] = q_gram_set

  print('Generated %d q-gram sets' % (len(rec_q_gram_dict)))
  print('')

  return rec_q_gram_dict


# -----------------------------------------------------------------------------
# Function for generating Bloom filters 
# -----------------------------------------------------------------------------

def gen_bf_dict(rec_q_gram_dict, bf_len, num_hash_funct, hash_type='dh'):
  """Encode each of the given q-gram set into one Bloom filter of the 
     given length based on the given encoding method and the given 
     number of hash functions and return a dictionary with 
     Bloom filters as values.

     Parameter Description:
       - rec_q_gram_dict  A dictionary of q-gram sets where each record_id 
                          in the dictionary has a set of q-gram set as value.
       - bf_len           The Bloom filter length.
       - num_hash_funct   The number of hash functios to be used for encoding 
                          each q-gram into the Bloom filter.
       - hash_type        The method to be used to encode attribute values
                          into Bloom filters. Encoding method can be either,
                            dh - Double hashing (default value) 
                            rh - Random hashing

     This method returns an output,
       - rec_bf_dict      The dictionary of Bloom filters where keys are record 
                          ids and Bloom filters are values.
  """

  print('Generate BF bit-patterns for %d records' % (len(rec_q_gram_dict)))
  print('  Attribute Bloom filter length: %d' % bf_len)
  print('  Number of hash functions used: %d' % num_hash_funct)
  print('  Hashing type used: %s            ' % \
        {'dh':'Double hashing', 'rh':'Random hashing'}[hash_type])

  rec_bf_dict= {}  # One Bloom filter per record

  bf_len_m1 = bf_len-1

  # Iterate through each q-gram set in the dictionary
  for rec_id in rec_q_gram_dict:

    # Get the q-gram set for the record
    rec_q_gram_set = rec_q_gram_dict[rec_id]

    # Generate a Bloom filter and set all bits to 0       
    rec_bf = bitarray.bitarray(bf_len)
    rec_bf.setall(0)

    # Hash all q-grams into bits in the Bloom filter
    #
    for q_gram in rec_q_gram_set:
      q_gram = q_gram.encode('utf-8')
      
      # Applying the double hashing for each q-gram
      if (hash_type == 'dh'):
        hex_str1 = BF_HASH_FUNCT1(q_gram).hexdigest()
        int1 =     int(hex_str1, 16)

        hex_str2 = BF_HASH_FUNCT2(q_gram).hexdigest()
        int2 =     int(hex_str2, 16)

        for i in range(num_hash_funct):
          gi = int1 + i*int2
          gi = int(gi % bf_len)

          if (rec_bf[gi] == 0):  # Not yet set
            rec_bf[gi] = 1

      # Applying the random hashing for each q-gram
      elif (hash_type == 'rh'):   
        hex_str = BF_HASH_FUNCT1(q_gram).hexdigest()
        random_seed = random.seed(int(hex_str,16))

        for i in range(num_hash_funct):
          gi = random.randint(0, bf_len_m1)

          if (rec_bf[gi] == 0):  # Not yet set
            rec_bf[gi] = 1

      else:  # Should not happend
        raise Exception("Wrong hash type: %s" %hash_type)

      # Concatenating the attribute level bfs together to 
      rec_bf_dict[rec_id] = rec_bf


  assert len(rec_bf_dict) == len(rec_q_gram_dict)
  print('Generated %d Bloom filters' % (len(rec_bf_dict)))
  print('')

  return rec_bf_dict


# -----------------------------------------------------------------------------
# Functions for computing similarities between two Bloom filters 
# -----------------------------------------------------------------------------

def dice_bf_sim(bf1, bf2):
  """Calculate the Dice Similarity between the two given Bloom filters. 

     Dice similarity is calculated between two Bloom filters A and B as 

        Dice similarity (A,B) = 2 x number of common 1 bit positions of A and B
                                -----------------------------------------------
                                number of 1 bit positions of A + 
                                                 number of 1 bit positions of B

     Parameter Description:
       - bf1  The first Bloom filter.
       - bf2  The second Bloom filter.

     This method returns an output,
       - bf_sim  The dice similarity between the Bloom filter pair.
  """

  assert len(bf1) == len(bf2), (len(bf1), len(bf2))

  num_ones_bf1 = bf1.count(1) # Count the number of bit positions set to 1 in BF1 
  num_ones_bf2 = bf2.count(1) # Count the number of bit positions set to 1 in BF2 
  
  # Perform the AND operation between two BFs
  # to get the common bit positions set to 1
  bf_common = bf1 & bf2 

  num_common_ones = bf_common.count(1) # Count the number of 1 bit positions

  bf_sim = (2 * num_common_ones) / (float(num_ones_bf1) + \
                                    float(num_ones_bf2))

  return bf_sim

# -----------------------------------------------------------------------------

def jaccard_bf_sim(bf1, bf2):
  """Calculate the Jaccard Similarity between the two given Bloom filters. 

     Jaccard similarity is calculated between two Bloom filters A and B as 

        Jaccard similarity (A,B) =  number of common 1 bit positions of A and B
                                   --------------------------------------------- 
                                   number of all 1 bit positions in both A and B

     Parameter Description:
       - bf1  The first Bloom filter.
       - bf2  The second Bloom filter.

     This method returns an output,
       - bf_sim  The jaccard similarity between the Bloom filter pair.
  """

  assert len(bf1) == len(bf2)

  # Perform the AND operation between two BFs
  # to get the common bit positions set to 1
  bf_common = bf1 & bf2

  num_common_ones = bf_common.count(1) # Count the number of 1 bit positions

  # Perform the OR operation between two BFs
  # to get all bit positions set to 1
  bf_all = bf1 | bf2
  num_all_ones = bf_all.count(1)

  bf_sim = num_common_ones / float(num_all_ones)

  return bf_sim

# -----------------------------------------------------------------------------

def hamming_bf_sim(bf1, bf2):
  """Calculate the Hamming distance between the two given Bloom filters. 

     Hamming similarity is calculated between two Bloom filters A and B as

        Hamming similarity (A,B) = 1 - Hamming distance between A and B

        Hamming distance is calculated as number of bit positions different
        between A and B. This is equal to performing XOR operation between
        A and B and count the number of 1 bits in the resulting Bloom filter.        

     Parameter Description:
       - bf1  The first Bloom filter.
       - bf2  The second Bloom filter.

     This method returns an output,
       - bf_sim  The hamming similarity between the Bloom filter pair.
  """

  assert len(bf1) == len(bf2)

  # Perform the XOR operation between two BFs
  # to get all bit positions that are different
  xor_bf = bf1 ^ bf2
  num_set_ones = xor_bf.count(1) # Count the number of 1 bit positions

  hamming_distance = num_set_ones / float(len(bf1))

  bf_sim = 1 - hamming_distance

  return bf_sim


# -----------------------------------------------------------------------------
# Functions for performing blocking based on the Bloom filters
# -----------------------------------------------------------------------------

def noBlocking(rec_bf_dict):
  """A function which does no blocking but simply puts all records from the
     given dictionary into one block.

     Parameter Description:
       rec_dict : Dictionary that holds the record identifiers as keys and
                  corresponding list of record values
  """

  print("Run 'no' blocking:")
  print('  Number of records to be blocked: '+str(len(rec_bf_dict)))
  print('')

  rec_id_list = list(rec_bf_dict.keys())

  block_dict = {'all_rec':rec_id_list}

  return block_dict


# -----------------------------------------------------------------------------

def segment_blocking(rec_bf_dict, num_seg):
  """Split the given dictionary of records into smaller blocks based on Bloom
     filter sets of bit position of length 'num_bit_pos_key' they have in
     common and return a blocking dictionary.

     Parameter Description:
       - rec_bf_dict     A dictionary of records with their entity identifiers
                         as keys and corresponding Bloom filters as values.
       - num_seg         The number of segments of Bloom filter bit positions
                         to be used to generate blocking keys.

     This method returns an output,
       - block_dict  A dictionary with blocking keys as keys and sets of record
                     identifiers as values.
  """

  assert num_seg >= 1, num_seg

  block_dict = {}

  bf_len = None

  # Loop over all records and extract all Bloom filter bit position sub arrays
  # of length 'seg_len' and insert the record into these corresponding
  # blocks
  #
  for rec_id in rec_bf_dict:

    rec_bf = rec_bf_dict[rec_id]

    # First time calculate the indices to use for splitting a Bloom filter
    #
    if (bf_len == None):
      bf_len = len(rec_bf)

      seg_len = int(float(bf_len) / num_seg)

      bf_split_index_list = []
      start_pos = 0
      end_pos =   seg_len

      # Generate the required bit position sub arrays
      #
      while (end_pos <= bf_len):
        bf_split_index_list.append((start_pos, end_pos))
        start_pos = end_pos   # Starting bit position
        end_pos +=  seg_len   # Ending bit position

      # Depending upon the Bloom filter length and 'seg_len' to use
      # the last segement might contain less than 'seg_len' number of 
      # positions.

    # Extract the bit position arrays for these segments
    #
    for (start_pos, end_pos) in bf_split_index_list:
      bf_seg = rec_bf[start_pos:end_pos]

      block_key =  bf_seg.to01()  # Make it a string
      block_rec_id_set = block_dict.get(block_key, set())
      block_rec_id_set.add(rec_id)
      block_dict[block_key] = block_rec_id_set

  return block_dict

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
  print('    Minimum block size: %d' % (min(block_sizeA_list)))
  print('    Average block size: %.2f' % \
        (float(sum(block_sizeA_list)) / len(block_sizeA_list)))
  print('    Maximum block size: %d' % (max(block_sizeA_list)))
  print('')

  print('Dataset B number of blocks generated: %d' % (numB_blocks))
  print('    Minimum block size: %d' % (min(block_sizeB_list)))
  print('    Average block size: %.2f' % \
        (float(sum(block_sizeB_list)) / len(block_sizeB_list)))
  print('    Maximum block size: %d' % (max(block_sizeB_list)))
  print('')


# -----------------------------------------------------------------------------
# Function for performing comparison based on the Bloom filters
# -----------------------------------------------------------------------------

def compareBlocks(blockA_dict, blockB_dict, recA_bf_dict, recB_bf_dict, \
                  comp_funct):
  """Build a similarity dictionary with pair of records from the two given
     block dictionaries. Candidate pairs are generated by pairing each record
     in a given block from data set A with all the records in the same block
     from dataset B.

     For each candidate pair a similarity is computed by comparing
     corresponding Bloom filters with the specified comparison method.

     Parameter Description:
       blockA_dict    : Dictionary of blocks from dataset A
       blockB_dict    : Dictionary of blocks from dataset B
       recA_bf_dict   : Dictionary of Bloom filters from dataset A
       recB_bf_dict   : Dictionary of Bloom filters from dataset B
       comp_funct     : Comparison method for comparing two 
                        Bloom filters. This function should returns a
                        similarity value.

     This method returns an output,
       - sim_bf_dict  : A similarity dictionary with one similarity value per
                        compared Bloom filter pair.

     Example: sim_bf_dict = {(recA1,recB1) = 1.0,
                             (recA1,recB5) = 0.9,
                             ...
                            }
  """

  print('Compare %d blocks from dataset A with %d blocks from dataset B' % \
        (len(blockA_dict), len(blockB_dict)))

  sim_bf_dict = {}  # A dictionary where keys are record pairs and values
                    # are similarity values between corresponding Bloom filters

  # Iterate through each block in block dictionary from dataset A
  #
  for (block_bkv, rec_idA_list) in blockA_dict.items():

    # Check if the same blocking key occurs also for dataset B
    #
    if (block_bkv in blockB_dict):

      # If so get the record identifier list from dataset B
      #
      rec_idB_list = blockB_dict[block_bkv]

      # Compare each record in rec_id_listA with each record from rec_id_listB
      #
      for rec_idA in rec_idA_list:

        bf_A = recA_bf_dict[rec_idA]  # Get the Bloom filter of record A

        for rec_idB in rec_idB_list:

          bf_B = recB_bf_dict[rec_idB] # Get the Bloom filter of record B

          # Compute the similarity between two Bloom filters
          #
          sim = comp_funct(bf_A, bf_B)

          # Add the similarity vector of the compared pair to the similarity
          # vector dictionary
          #
          sim_bf_dict[(rec_idA, rec_idB)] = sim

  print('  Compared %d Bloom filter pairs' % (len(sim_bf_dict)))
  print('')

  return sim_bf_dict

# -----------------------------------------------------------------------------
# Function for performing classification based on the Bloom filters
# -----------------------------------------------------------------------------

def thresholdClassify(sim_bf_dict, sim_thres):
  """Method to classify the given similarity dictionary with regard to
     a given similarity threshold (in the range 0.0 to 1.0), where Bloom filter
     pairs with a similarity of at least this threshold are classified as
     matches and all others as non-matches.

     Parameter Description:
       sim_bf_dict : Dictionary of record pairs with their identifiers as
                     as keys and their corresponding similarity value as
                     values.
       sim_thres   : The classification similarity threshold.
  """

  assert sim_thres >= 0.0 and sim_thres <= 1.0, sim_thres

  print('Similarity threshold based classification of %d Bloom filter pairs' % \
        (len(sim_bf_dict)))
  print('  Classification similarity threshold: %.3f' % (sim_thres))

  class_match_set    = set()
  class_nonmatch_set = set()

  # Iterate over all record pairs
  #
  for (rec_id_tuple, sim) in sim_bf_dict.items():

    if sim >= sim_thres:  # Similarity is high enough
      class_match_set.add(rec_id_tuple)
    else:
      class_nonmatch_set.add(rec_id_tuple)

  print('  Classified %d Bloom filter pairs as matches and %d as non-matches' % \
        (len(class_match_set), len(class_nonmatch_set)))
  print('')

  return class_match_set, class_nonmatch_set

# =============================================================================
# Main Program
# =============================================================================
#
# Step 1: Load the two datasets from CSV files

start_time = time.time()

recA_dict = loadDataset.load_data_set(datasetA_name, rec_idA_col, \
                                      attrA_list, headerA_line)
recB_dict = loadDataset.load_data_set(datasetB_name, rec_idB_col, \
                                      attrB_list, headerB_line)

# Load data set of true matching pairs
#
true_match_set = loadDataset.load_truth_data(truthfile_name)

loading_time = time.time() - start_time

# -----------------------------------------------------------------------------
# Step 2: Generate the q-gram sets of the datasets
start_time = time.time()

q = 2   # Q-gram length

recA_q_gram_dict = gen_q_gram_dict(recA_dict, q)
recB_q_gram_dict = gen_q_gram_dict(recB_dict, q)

qgram_gen_time = time.time() - start_time

# -----------------------------------------------------------------------------
# Step 3: Generate Bloom filters of the datasets
start_time = time.time()

bf_len         = 1000 # Bloom filter length
num_hash_funct = 15   # Number of hash functions
hash_type      = 'dh' # Hashing method (dh: Double hashing, rh: Random hashing)

recA_bf_dict = gen_bf_dict(recA_q_gram_dict, bf_len, num_hash_funct, hash_type)
recB_bf_dict = gen_bf_dict(recB_q_gram_dict, bf_len, num_hash_funct, hash_type)

bf_gen_time = time.time() - start_time

# -----------------------------------------------------------------------------
# Step 4: Block the datasets

start_time = time.time()

# Select one blocking technique

# No blocking (all records in one block)
#
#blockA_dict = noBlocking(recA_bf_dict)
#blockB_dict = noBlocking(recB_bf_dict)

# Simple segmentation-based blocking
#
num_seg = 100
assert bf_len > num_seg

blockA_dict = segment_blocking(recA_bf_dict, num_seg)
blockB_dict = segment_blocking(recB_bf_dict, num_seg)

blocking_time = time.time() - start_time

# Print blocking statistics
#
printBlockStatistics(blockA_dict, blockB_dict)

# -----------------------------------------------------------------------------
# Step 5: Compare the candidate Bloom filter pairs

start_time = time.time()

# Similarity functions that could be used for comparing 
# of a Bloom filter pair
#   1. dice_bf_sim    - # Dice similarity function
#   2. jaccard_bf_sim - # Jaccard similarity function
#   3. hamming_bf_sim - # Hamming similarity function

comp_funct = dice_bf_sim  

sim_bf_dict = compareBlocks(blockA_dict, blockB_dict, \
                            recA_bf_dict, recB_bf_dict, \
                            comp_funct)

comparison_time = time.time() - start_time

# -----------------------------------------------------------------------------
# Step 6: Classify the candidate pairs

start_time = time.time()

# Similarity threshold based classification
#
sim_threshold = 0.9   # Similarity threshold
class_match_set, class_nonmatch_set = thresholdClassify(sim_bf_dict, \
                                                        sim_threshold)

classification_time = time.time() - start_time

# -----------------------------------------------------------------------------
# Step 7: Evaluate the classification

# Get the number of record pairs compared
#
num_comparisons = len(sim_bf_dict)

# Get the number of total record pairs to compared if no blocking used
#
all_comparisons = len(recA_dict) * len(recB_dict)

# Get the list of identifiers of the compared record pairs
#
cand_rec_id_pair_list = sim_bf_dict.keys()

# Blocking evaluation
#
rr = evaluation.reduction_ratio(num_comparisons, all_comparisons)
pc = evaluation.pairs_completeness(cand_rec_id_pair_list, true_match_set)
pq = evaluation.pairs_quality(cand_rec_id_pair_list, true_match_set)

print('Blocking evaluation:')
print('  Reduction ratio:    %.3f' % (rr))
print('  Pairs completeness: %.3f' % (pc))
print('  Pairs quality:      %.3f' % (pq))
print('')

# Linkage evaluation
#
linkage_result = evaluation.confusion_matrix(class_match_set,
                                             class_nonmatch_set,
                                             true_match_set,
                                             all_comparisons)

accuracy =    evaluation.accuracy(linkage_result)
precision =   evaluation.precision(linkage_result)
recall    =   evaluation.recall(linkage_result)
fmeasure  =   evaluation.fmeasure(linkage_result)

print('Linkage evaluation:')
print('  Accuracy:    %.3f' % (accuracy))
print('  Precision:   %.3f' % (precision))
print('  Recall:      %.3f' % (recall))
print('  F-measure:   %.3f' % (fmeasure))
print('')

linkage_time = loading_time + qgram_gen_time + bf_gen_time + \
               blocking_time + comparison_time + classification_time

print('Total runtime required for linkage: %.3f sec' % (linkage_time))

# -----------------------------------------------------------------------------

# End of program.

