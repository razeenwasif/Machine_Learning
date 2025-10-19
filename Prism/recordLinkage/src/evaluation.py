""" Module with functionalities to evaluate the results of a record linkage
    excercise, both with reagrd to linkage quality as well as complexity.
"""
import cudf

# =============================================================================

def confusion_matrix(class_match_set, class_nonmatch_set, true_match_set,
                     all_comparisons):
  """Compute the confusion (error) matrix which has the following form:

     +-----------------+-----------------------+----------------------+
     |                 |  Predicted Matches    | Predicted NonMatches |
     +=================+=======================+======================+
     | True  Matches   | True Positives (TP)   | False Negatives (FN) |
     +-----------------+-----------------------+----------------------+
     | True NonMatches | False Positives (FP)  | True Negatives (TN)  |
     +-----------------+-----------------------+----------------------+

     The four values calculated in the confusion matrix (TP, FP, TN, and FN)
     are then the basis of linkag equality measures such as precision and
     recall.

     Parameter Description:
       class_match_set    : Set of classified matches (record identifier
                            pairs)
       class_nonmatch_set : Set of classified non-matches (record identifier
                            pairs)
       true_match_set     : Set of true matches (record identifier pairs)
       all_comparisons    : The total number of comparisons between all record
                            pairs

     This function returns a list with four values representing TP, FP, FN,
     and TN.
  """

  print('Calculating confusion matrix using %d classified matches, %d ' % \
        (len(class_match_set), len(class_nonmatch_set)) + 'classified ' + \
        'non-matches, and %d true matches' % (len(true_match_set)))

  num_tp = 0  # number of true positives
  num_fp = 0  # number of false positives
  num_tn = 0  # number of true negatives
  num_fn = 0  # number of false negatives

  # Iterate through the classified matches to check if they are true matches or
  # not
  #
  for rec_id_tuple in class_match_set:
    if (rec_id_tuple in true_match_set):
      num_tp += 1
    else:
      num_fp += 1

  # Iterate through the classified non-matches to check of they are true
  # non-matches or not
  #
  for rec_id_tuple in class_nonmatch_set:

    # Check a record tuple is only counted once
    #
    assert rec_id_tuple not in class_match_set, rec_id_tuple

    if (rec_id_tuple in true_match_set):
      num_fn += 1
    else:
      num_tn += 1

  # Finally count all missed true matches to the false negatives
  #
  for rec_id_tuple in true_match_set:
    if ((rec_id_tuple not in class_match_set) and \
        (rec_id_tuple not in class_nonmatch_set)):
      num_fn += 1

  num_tn = all_comparisons - num_tp - num_fp - num_fn

  print('  TP=%s, FP=%d, FN=%d, TN=%d' % (num_tp, num_fp, num_fn, num_tn))
  print('')

  return [num_tp, num_fp, num_fn, num_tn]

# =============================================================================
# Different linkage quality measures

def accuracy(confusion_matrix):
  """Compute accuracy using the given confusion matrix.

     Accuracy is calculated as (TP + TN) / (TP + FP + FN + TN).

     Parameter Description:
       confusion_matrix : The matrix with TP, FP, FN, TN values.

     The method returns a float value.
  """

  num_tp = confusion_matrix[0]
  num_fp = confusion_matrix[1]
  num_fn = confusion_matrix[2]
  num_tn = confusion_matrix[3]

  accuracy = float(num_tp + num_tn) / (num_tp + num_fp + num_fn + num_tn)

  return accuracy

# -----------------------------------------------------------------------------

def precision(confusion_matrix):
  """Compute precision using the given confusion matrix.

     Precision is calculated as TP / (TP + FP).

     Parameter Description:
       confusion_matrix : The matrix with TP, FP, FN, TN values.

     The method returns a float value.
  """

  num_tp = confusion_matrix[0]
  num_fp = confusion_matrix[1]

  if (num_tp + num_fp) == 0:
    precision = 0.0
  else:
    precision = float(num_tp) / (num_tp + num_fp)

  return precision

# -----------------------------------------------------------------------------

def recall(confusion_matrix):
  """Compute recall using the given confusion matrix.

     Recall is calculated as TP / (TP + FN).

      Parameter Description:
        confusion_matrix : The matrix with TP, FP, FN, TN values.

      The method returns a float value.
  """

  num_tp = confusion_matrix[0]
  num_fn = confusion_matrix[2]

  if (num_tp + num_fn) == 0:
    recall = 0.0
  else:
    recall = float(num_tp) / (num_tp + num_fn)

  return recall

# -----------------------------------------------------------------------------

def fmeasure(confusion_matrix):
  """Compute the f-measure of the linkage.

     The f-measure is calculated as:

              2 * (precision * recall) / (precision + recall).

     Parameter Description:
       confusion_matrix : The matrix with TP, FP, FN, TN values.

     The method returns a float value.
  """
  prec = precision(confusion_matrix)
  rec = recall(confusion_matrix)

  if (prec + rec) == 0:
    f_measure = 0.0
  else:
    f_measure = 2.0 * (prec * rec) / (prec + rec)

  return f_measure

# =============================================================================
# Different linkage complexity measures

def reduction_ratio(num_comparisons, all_comparisons):
  """Compute the reduction ratio using the given confusion matrix.

     Reduction ratio is calculated as 1 - num_comparison / (TP + FP + FN+ TN).

     Parameter Description:
       num_comparisons : The number of candidate record pairs
       all_comparisons : The total number of comparisons between all record
                         pairs

     The method returns a float value.
  """

  if (num_comparisons == 0):
    return 1.0

  rr = 1.0 - float(num_comparisons) / all_comparisons

  return rr

# -----------------------------------------------------------------------------

def pairs_completeness(sim_vectors_gdf, true_match_set):
  """Pairs completeness measures the effectiveness of a blocking technique in
     the record linkage process.

     Pairs completeness is calculated as the number of true matches included in
     the candidate record pairs divided by the number of all true matches.

     Parameter Description:
       sim_vectors_gdf : A cuDF DataFrame with candidate record pairs.
       true_match_set  : Set of true matches (record identifier pairs)

     The method returns a float value.
  """

  cand_pairs_gdf = sim_vectors_gdf[['rec_id_A', 'rec_id_B']]
  true_matches_gdf = cudf.DataFrame(list(true_match_set), columns=['rec_id_A', 'rec_id_B'])

  merged = cand_pairs_gdf.merge(true_matches_gdf, on=['rec_id_A', 'rec_id_B'], how='inner')
  num_true_matches_in_cand = len(merged)
  num_true_matches = len(true_match_set)

  if num_true_matches == 0:
    pc = 0.0
  else:
    pc = float(num_true_matches_in_cand) / num_true_matches

  return pc

# -----------------------------------------------------------------------------

def pairs_quality(sim_vectors_gdf, true_match_set):
  """Pairs quality measures the efficiency of a blocking technique.

     Pairs quality is calculated as the number of true matches included in the
     candidate record pairs divided by the number of candidate record pairs
     generated by blocking.

     Parameter Description:
       sim_vectors_gdf : A cuDF DataFrame with candidate record pairs.
       true_match_set  : Set of true matches (record identifier pairs)

     The method returns a float value.
  """

  cand_pairs_gdf = sim_vectors_gdf[['rec_id_A', 'rec_id_B']]
  true_matches_gdf = cudf.DataFrame(list(true_match_set), columns=['rec_id_A', 'rec_id_B'])

  merged = cand_pairs_gdf.merge(true_matches_gdf, on=['rec_id_A', 'rec_id_B'], how='inner')
  num_true_matches_in_cand = len(merged)
  num_cand_pairs = len(sim_vectors_gdf)

  if num_cand_pairs == 0:
    pq = 0.0
  else:
    pq = float(num_true_matches_in_cand) / num_cand_pairs

  return pq

# -----------------------------------------------------------------------------


def evaluate_linkage(class_match_set, class_nonmatch_set, true_match_set, all_comparisons):
  """Evaluate the linkage process and print the results.

     Parameter Description:
       class_match_set    : Set of classified matches (record identifier
                            pairs)
       class_nonmatch_set : Set of classified non-matches (record identifier
                            pairs)
       true_match_set     : Set of true matches (record identifier pairs)
       all_comparisons    : The total number of comparisons between all record
                            pairs
  """

  print('Linkage evaluation:')
  print('===================')

  conf_matrix = confusion_matrix(class_match_set, class_nonmatch_set, true_match_set, all_comparisons)

  acc = accuracy(conf_matrix)
  print(f'Accuracy: {acc:.3f}')

  prec = precision(conf_matrix)
  print(f'Precision: {prec:.3f}')

  rec = recall(conf_matrix)
  print(f'Recall: {rec:.3f}')

  f_measure = fmeasure(conf_matrix)
  print(f'F-measure: {f_measure:.3f}')

# -----------------------------------------------------------------------------

def evaluate_blocking(sim_vectors_gdf, true_match_set, num_comparisons, all_comparisons):
  """Evaluate the blocking process and print the results.

     Parameter Description:
       sim_vectors_gdf : A cuDF DataFrame with candidate record pairs.
       true_match_set  : A set of true match record ID pairs.
       num_comparisons : The number of comparisons made (the number of
                         candidate pairs).
       all_comparisons : The total number of possible comparisons.
  """

  print('Blocking evaluation:')
  print('====================')

  rr = reduction_ratio(num_comparisons, all_comparisons)
  print(f'Reduction ratio: {rr:.3f}')

  pc = pairs_completeness(sim_vectors_gdf, true_match_set)
  print(f'Pairs completeness: {pc:.3f}')

  pq = pairs_quality(sim_vectors_gdf, true_match_set)
  print(f'Pairs quality: {pq:.3f}')

# -----------------------------------------------------------------------------

# End of program.

