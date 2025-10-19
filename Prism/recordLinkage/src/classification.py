""" Module with functionalities for classifying a dictionary of record pairs
    and their similarities.

    Each function in this module returns two sets, one with record pairs
    classified as matches and the other with record pairs classified as
    non-matches.
"""

# =============================================================================

import sys
import cudf
import cupy
import numpy as np
from cuml.ensemble import RandomForestClassifier
from cuml.model_selection import train_test_split


def _ensure_cupy_array(data):
    """Return *data* as a CuPy array without triggering an unnecessary host copy.

    Parameters
    ----------
    data : cudf.Series, cupy.ndarray, numpy.ndarray, or array-like
        The object produced by cuDF/cuML, potentially already on device.

    Returns
    -------
    cupy.ndarray
        A GPU array view of the supplied data.
    """
    if hasattr(data, "to_cupy"):
        return data.to_cupy()
    return cupy.asarray(data)


def _build_training_sample(
    X_matches,
    y_matches,
    X,
    y,
    non_match_indices,
    n_matches,
    ratio,
    random_state=42,
):
    """Construct a balanced training sample for the classifier.

    Parameters
    ----------
    X_matches, y_matches : cudf.DataFrame, cudf.Series
        Feature matrix and labels for known matches.
    X, y : cudf.DataFrame, cudf.Series
        Full feature matrix and corresponding labels.
    non_match_indices : cudf.Series
        Index of rows labelled as non-matches inside *y*.
    n_matches : int
        Total number of known matches.
    ratio : int
        Desired sampling ratio of non-matches per match (e.g., 2 => 1:2).
    random_state : int, optional
        Seed for reproducible sampling.

    Returns
    -------
    tuple (cudf.DataFrame, cudf.Series, int)
        Sampled feature matrix, sampled labels, and number of non-matches drawn.
    """
    if n_matches == 0 or ratio <= 0:
        return X_matches, y_matches, 0

    n_non_matches_available = len(non_match_indices)
    n_non_match_sample = min(n_non_matches_available, n_matches * ratio)
    if n_non_match_sample <= 0:
        return X_matches, y_matches, 0

    sampled_non_match_indices = non_match_indices.sample(
        n=n_non_match_sample, replace=False, random_state=random_state
    )
    X_non_match_sample = X.take(sampled_non_match_indices)
    y_non_match_sample = y.take(sampled_non_match_indices)

    X_sampled = cudf.concat([X_matches, X_non_match_sample])
    y_sampled = cudf.concat([y_matches, y_non_match_sample])
    return X_sampled, y_sampled, n_non_match_sample


PRECISION_FOCUSED_BETA = 0.25  # Favour precision while maintaining recall
GLOBAL_MIN_PRECISION = 0.55    # Enforce minimum precision when choosing global thresholds
GLOBAL_MIN_RECALL = 0.05       # Avoid degenerate thresholds with zero recall


def _select_threshold_for_fbeta(probas_np, labels_np, default_threshold, beta=1.0, min_precision=0.0):
    """Select the probability threshold that maximises the F-beta score.

    Parameters
    ----------
    probas_np : numpy.ndarray
        Predicted probabilities for the positive class.
    labels_np : numpy.ndarray
        Ground-truth binary labels aligned with *probas_np*.
    default_threshold : float
        Fallback threshold to use when metrics cannot be computed.
    beta : float, optional
        Weighting factor for the F-beta score (`beta < 1` favours precision,
        `beta > 1` favours recall).
    min_precision : float, optional
        Minimum precision a candidate threshold must achieve to be considered.

    Returns
    -------
    tuple (float, float, float, float)
        Best threshold, precision, recall, and F-beta score achieved.
    """
    if probas_np.size == 0:
        return default_threshold, 0.0, 0.0, 0.0

    # Ensure labels are binary 0/1
    labels_np = labels_np.astype(np.int32, copy=False)
    positives = labels_np == 1
    if positives.sum() == 0:
        return default_threshold, 0.0, 0.0, 0.0

    candidate_thresholds = np.unique(
        np.clip(
            np.concatenate(
                [
                    np.linspace(0.05, 0.95, 19),
                    np.percentile(probas_np, [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]),
                    np.array([default_threshold, 0.5]),
                ]
            ),
            0.0,
            1.0,
        )
    )

    beta_sq = beta * beta
    best_threshold = None
    best_precision = 0.0
    best_recall = 0.0
    best_fbeta = 0.0

    fallback_threshold = default_threshold
    fallback_precision = 0.0
    fallback_recall = 0.0
    fallback_fbeta = 0.0

    positive_count = positives.sum()

    for thr in candidate_thresholds:
        preds = probas_np >= thr
        tp = np.count_nonzero(preds & positives)
        fp = np.count_nonzero(preds) - tp
        fn = positive_count - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        if precision < min_precision:
            # Still track fallback statistics for precision ordering
            recall_tmp = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if precision > fallback_precision or (
                np.isclose(precision, fallback_precision) and thr > fallback_threshold
            ):
                fallback_threshold = thr
                fallback_precision = precision
                fallback_recall = recall_tmp
                fallback_fbeta = (
                    0.0
                    if (precision == 0.0 and recall_tmp == 0.0)
                    else (1 + beta_sq) * precision * recall_tmp / (beta_sq * precision + recall_tmp)
                )
            continue
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision == 0.0 and recall == 0.0:
            fbeta = 0.0
        else:
            fbeta = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)

        if fbeta > best_fbeta or (np.isclose(fbeta, best_fbeta) and thr < best_threshold):
            best_threshold = thr
            best_precision = precision
            best_recall = recall
            best_fbeta = fbeta

        # Update fallback to capture the most precise option seen so far
        if precision > fallback_precision or (
            np.isclose(precision, fallback_precision) and thr > fallback_threshold
        ):
            fallback_threshold = thr
            fallback_precision = precision
            fallback_recall = recall
            fallback_fbeta = fbeta

    if best_threshold is None:
        return fallback_threshold, fallback_precision, fallback_recall, fallback_fbeta

    return best_threshold, best_precision, best_recall, best_fbeta

def exactClassify(sim_vec_dict):
  """Method to classify the given similarity vector dictionary assuming only
     exact matches (having all similarities of 1.0) are matches.

     Parameter Description:
       sim_vec_dict : Dictionary of record pairs with their identifiers as
                      as keys and their corresponding similarity vectors as
                      values.

     The classification is based on the exact matching of attribute values,
     that is the similarity vector for a given record pair must contain 1.0
     for all attribute values.

     Example:
       (recA1, recB1) = [1.0, 1.0, 1.0, 1.0] => match
       (recA2, recB5) = [0.0, 1.0, 0.0, 1.0] = non-match
  """

  print('Exact classification of %d record pairs' % (len(sim_vec_dict)))

  class_match_set    = set()
  class_nonmatch_set = set()

  # Iterate over all record pairs
  #
  for (rec_id_tuple, sim_vec) in sim_vec_dict.items():

    sim_sum = sum(sim_vec)  # Sum all attribute similarities

    if sim_sum == len(sim_vec):  # All similarities were 1.0
      class_match_set.add(rec_id_tuple)
    else:
      class_nonmatch_set.add(rec_id_tuple)

  print('  Classified %d record pairs as matches and %d as non-matches' % \
        (len(class_match_set), len(class_nonmatch_set)))
  print('')

  return class_match_set, class_nonmatch_set

# -----------------------------------------------------------------------------

def thresholdClassify(sim_vec_dict, sim_thres):
  """Method to classify the given similarity vector dictionary with regard to
     a given similarity threshold (in the range 0.0 to 1.0), where record pairs
     with an average similarity of at least this threshold are classified as
     matches and all others as non-matches.

     Parameter Description:
       sim_vec_dict : Dictionary of record pairs with their identifiers as
                      as keys and their corresponding similarity vectors as
                      values.
       sim_thres    : The classification similarity threshold.
  """

  assert sim_thres >= 0.0 and sim_thres <= 1.0, sim_thres

  print('Similarity threshold based classification of %d record pairs' % \
        (len(sim_vec_dict)))
  print('  Classification similarity threshold: %.3f' % (sim_thres))

  class_match_set    = set()
  class_nonmatch_set = set()

  # Iterate over all record pairs
  #
  for (rec_id_tuple, sim_vec) in sim_vec_dict.items():

    sim_sum = float(sum(sim_vec))  # Sum all attribute similarities
    avr_sim = sim_sum / len(sim_vec)

    if avr_sim >= sim_thres:  # Average similarity is high enough
      class_match_set.add(rec_id_tuple)
    else:
      class_nonmatch_set.add(rec_id_tuple)

  print('  Classified %d record pairs as matches and %d as non-matches' % \
        (len(class_match_set), len(class_nonmatch_set)))
  print('')

  return class_match_set, class_nonmatch_set

# -----------------------------------------------------------------------------

def minThresholdClassify(sim_vec_dict, sim_thres):
  """Method to classify the given similarity vector dictionary with regard to
     a given similarity threshold (in the range 0.0 to 1.0), where record pairs
     that have all their similarities (of all attributes compared) with at
     least this threshold are classified as matches and all others as
     non-matches.

     Parameter Description:
       sim_vec_dict : Dictionary of record pairs with their identifiers as
                      as keys and their corresponding similarity vectors as
                      values.
       sim_thres    : The classification minimum similarity threshold.
  """

  assert sim_thres >= 0.0 and sim_thres <= 1.0, sim_thres

  print('Minimum similarity threshold based classification of ' + \
        '%d record pairs' % (len(sim_vec_dict)))
  print('  Classification similarity threshold: %.3f' % (sim_thres))

  class_match_set    = set()
  class_nonmatch_set = set()

  # Iterate over all record pairs
  #
  for (rec_id_tuple, sim_vec) in sim_vec_dict.items():

    # Flag to check is all attribute similarities are high enough or not
    #
    record_pair_match = True

    # check for all the compared attributes
    #
    for sim in sim_vec:
      if sim < sim_thres:  # Similarity is not enough
        record_pair_match = False
        break  # No need to compare more similarities, speed-up the process

    if (record_pair_match):  # All similaries are high enough
      class_match_set.add(rec_id_tuple)
    else:
      class_nonmatch_set.add(rec_id_tuple)

  print('  Classified %d record pairs as matches and %d as non-matches' % \
        (len(class_match_set), len(class_nonmatch_set)))
  print('')

  return class_match_set, class_nonmatch_set

# -----------------------------------------------------------------------------

def weightedSimilarityClassify(sim_vec_dict, weight_vec, sim_thres):
  """Method to classify the given similarity vector dictionary with regard to
     a given weight vector and a given similarity threshold (in the range 0.0
     to 1.0), where an overall similarity is calculated based on the weights
     for each attribute, and where record pairs with the similarity of at least
     the given threshold are classified as matches and all others as
     non-matches.

     Parameter Description:
       sim_vec_dict : Dictionary of record pairs with their identifiers as
                      as keys and their corresponding similarity vectors as
                      values.
       weight_vec   : A vector with weights, one weight for each attribute.
       sim_thres    : The classification similarity threshold.
  """

  assert sim_thres >= 0.0 and sim_thres <= 1.0, sim_thres

  # Check weights are available for all attributes
  #
  first_sim_vec = list(sim_vec_dict.values())[0]
  assert len(weight_vec) == len(first_sim_vec), len(weight_vec)

  print('Weighted similarity based classification of %d record pairs' % \
        (len(sim_vec_dict)))
  print('  Weight vector: %s'   % (str(weight_vec)))
  print('  Classification similarity threshold: %.3f' % (sim_thres))

  class_match_set    = set()
  class_nonmatch_set = set()

  weight_sum = sum(weight_vec)  # Sum of all attribute weights

  # Iterate over all record pairs
  #
  for (rec_id_tuple, sim_vec) in sim_vec_dict.items():

    sim_sum = 0.0

    # Compute weighted sim for each attribute
    #
    for sim, weight in zip(sim_vec, weight_vec):
      sim_sum += sim * weight

    avr_sim = sim_sum / weight_sum  # Compute noramlised average similarity

    if avr_sim >= sim_thres:  # Average similarity is high enough
      class_match_set.add(rec_id_tuple)
    else:
      class_nonmatch_set.add(rec_id_tuple)

  print('  Classified %d record pairs as matches and %d as non-matches' % \
        (len(class_match_set), len(class_nonmatch_set)))
  print('')

  return class_match_set, class_nonmatch_set

# -----------------------------------------------------------------------------

def supervisedMLClassify(
    sim_vectors_gdf,
    true_match_set,
    n_estimators=10,
    threshold=0.4,
    threshold_offset=0.0,
    min_precision=GLOBAL_MIN_PRECISION,
    min_recall=GLOBAL_MIN_RECALL,
    precision_beta=PRECISION_FOCUSED_BETA,
):
    """Classify candidate pairs using a GPU-backed random forest classifier.

    The training data is constructed from the supplied similarity vectors,
    sampling non-matches at multiple ratios and evaluating several `max_depth`
    settings. The model/threshold combination that yields the highest
    validation F1 score is selected, after which the decision threshold is
    tightened using a precision-focused F-beta search before being applied to
    the full candidate set.

    Parameters
    ----------
    sim_vectors_gdf : cudf.DataFrame
        Similarity vectors with `rec_id_A` and `rec_id_B` columns.
    true_match_set : set[tuple]
        Ground-truth record identifier pairs flagged as matches.
    n_estimators : int, optional
        Number of trees in the random forest ensemble.
    threshold : float, optional
        Baseline probability threshold used during the hyper-parameter sweep.
    threshold_offset : float, optional
        Value added to the selected decision threshold prior to classification.
        Negative offsets relax the threshold (potentially boosting recall).

    Returns
    -------
    tuple (set, set)
        `class_match_set` and `class_nonmatch_set` containing record-ID pairs.
    """

    class_match_set =    set()
    class_nonmatch_set = set()

    print('Supervised random forest classification of %d record pairs' % \
        (len(sim_vectors_gdf)))
    sys.stdout.flush()

    rec_pairs = sim_vectors_gdf[['rec_id_A', 'rec_id_B']]
    X = sim_vectors_gdf.drop(columns=['rec_id_A', 'rec_id_B'])
    X = X.fillna(0.0)
    
    # Vectorized label creation using isin for efficient lookup
    #
    true_match_df = cudf.DataFrame(list(true_match_set), columns=['rec_id_A', 'rec_id_B'])
    true_match_df['label'] = 1

    labeled_pairs = rec_pairs.merge(
        true_match_df, on=['rec_id_A', 'rec_id_B'], how='left', sort=False
    )
    y = labeled_pairs['label'].fillna(0).astype('int32')

    # --- Create a training sample with improved class balance ---

    match_mask = (y == 1)
    X_matches = X[match_mask]
    y_matches = y[match_mask]
    n_matches = len(X_matches)

    print(f'  Total true matches in dataset: {n_matches}')
    print(f'  Total non-matches in dataset: {len(y) - n_matches}')

    non_match_indices_gpu = y[~match_mask].index.to_series()

    ratio_candidates = [1, 2, 3, 4]
    depth_candidates = [12, 15, 18]

    print(
        '  Sweeping non-match ratios %s and max_depth values %s'
        % (ratio_candidates, [d if d is not None else 'None' for d in depth_candidates])
    )
    sys.stdout.flush()

    global_min_precision = max(0.0, min(1.0, float(min_precision)))
    global_min_recall = max(0.0, min(1.0, float(min_recall)))
    precision_focus_beta = max(0.01, float(precision_beta))

    best_search = {
        'f1': -1.0,
        'ratio': 1,
        'depth': 15,
        'threshold': threshold,
        'precision': 0.0,
        'recall': 0.0,
        'accuracy': 0.0,
    }

    for ratio in ratio_candidates:
        X_sampled_tmp, y_sampled_tmp, n_non_match_sample = _build_training_sample(
            X_matches, y_matches, X, y, non_match_indices_gpu, n_matches, ratio, random_state=42
        )
        if len(X_sampled_tmp) == 0:
            continue

        print(
            f'  Evaluating ratio 1:{ratio} (non-matches sampled: {n_non_match_sample})'
        )
        sys.stdout.flush()

        X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp = train_test_split(
            X_sampled_tmp, y_sampled_tmp, test_size=0.33, random_state=42
        )

        for depth in depth_candidates:
            print(f'    Trying RandomForest max_depth={depth if depth is not None else "None"}...')
            sys.stdout.flush()

            clf_tmp = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=42,
                max_depth=depth,
                min_samples_split=20,
                min_samples_leaf=10,
            )
            clf_tmp.fit(X_train_tmp, y_train_tmp)
            accuracy_tmp = clf_tmp.score(X_test_tmp, y_test_tmp)

            test_probas_tmp = clf_tmp.predict_proba(X_test_tmp)
            test_probas_gpu_tmp = _ensure_cupy_array(test_probas_tmp)[:, 1]
            y_test_gpu_tmp = _ensure_cupy_array(y_test_tmp)

            test_probas_np_tmp = cupy.asnumpy(test_probas_gpu_tmp)
            y_test_np_tmp = cupy.asnumpy(y_test_gpu_tmp)
            thr_tmp, prec_tmp, rec_tmp, f1_tmp = _select_threshold_for_fbeta(
                test_probas_np_tmp, y_test_np_tmp, threshold, beta=1.0
            )

            print(
                '      Validation accuracy: %.3f, precision: %.3f, recall: %.3f, F1: %.3f (threshold %.3f)'
                % (accuracy_tmp, prec_tmp, rec_tmp, f1_tmp, thr_tmp)
            )
            sys.stdout.flush()

            if f1_tmp > best_search['f1']:
                best_search = {
                    'f1': f1_tmp,
                    'ratio': ratio,
                    'depth': depth,
                    'threshold': thr_tmp,
                    'precision': prec_tmp,
                    'recall': rec_tmp,
                    'accuracy': accuracy_tmp,
                }

            del clf_tmp, test_probas_tmp, test_probas_gpu_tmp, y_test_gpu_tmp

        del X_sampled_tmp, y_sampled_tmp, X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp
        import gc
        gc.collect()

    best_ratio = best_search['ratio']
    best_depth = best_search['depth']
    print(
        '  Selected configuration -> ratio 1:%d, max_depth=%s (validation F1=%.3f)'
        % (best_ratio, best_depth if best_depth is not None else 'None', best_search['f1'])
    )
    sys.stdout.flush()

    X_sampled, y_sampled, n_non_match_sample = _build_training_sample(
        X_matches, y_matches, X, y, non_match_indices_gpu, n_matches, best_ratio, random_state=42
    )
    print(
        f'  Building final training sample with ratio 1:{best_ratio} (non-matches: {n_non_match_sample})'
    )
    sys.stdout.flush()

    if len(X_sampled) == 0:
        X_sampled = X_matches
        y_sampled = y_matches
        print('  WARNING: No non-match samples available; training on matches only.')
        sys.stdout.flush()

    X_train, X_test, y_train, y_test = train_test_split(
        X_sampled, y_sampled, test_size=0.33, random_state=42
    )

    print('  Number of training records: %d' % len(X_train))
    print('  Number of testing records: %d' % len(X_test))
    print('')
    sys.stdout.flush()

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        max_depth=best_depth,
        min_samples_split=20,
        min_samples_leaf=10,
    )
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    print('  Classifier accuracy on sampled test set: %.3f' % accuracy)
    print('')
    sys.stdout.flush()

    test_probas = clf.predict_proba(X_test)
    test_probas_gpu = _ensure_cupy_array(test_probas)[:, 1]
    y_test_gpu = _ensure_cupy_array(y_test)
    test_probas_np = cupy.asnumpy(test_probas_gpu)
    y_test_np = cupy.asnumpy(y_test_gpu)
    best_threshold, best_precision, best_recall, best_f1 = _select_threshold_for_fbeta(
        test_probas_np, y_test_np, best_search['threshold'], beta=1.0
    )
    print(
        '  Validation threshold search: thr=%.3f -> precision=%.3f, recall=%.3f, F1=%.3f'
        % (best_threshold, best_precision, best_recall, best_f1)
    )
    sys.stdout.flush()

    precision_val_threshold, precision_val_prec, precision_val_rec, precision_val_fbeta = _select_threshold_for_fbeta(
        test_probas_np, y_test_np, best_threshold, beta=precision_focus_beta, min_precision=global_min_precision
    )
    print(
        '  Precision-focused validation threshold: thr=%.3f -> precision=%.3f, recall=%.3f, F-beta(%.2f)=%.3f'
        % (
            precision_val_threshold,
            precision_val_prec,
            precision_val_rec,
            precision_focus_beta,
            precision_val_fbeta,
        )
    )
    sys.stdout.flush()

    # First pass: collect probability predictions to analyze distribution
    chunk_size = 1_000_000
    probas_list = []
    print(f'  Predicting probabilities on {len(X)} pairs in {((len(X)-1)//chunk_size)+1} chunks of size {chunk_size}...')
    sys.stdout.flush()
    
    for i in range(0, len(X), chunk_size):
        chunk = X.iloc[i:i + chunk_size]
        chunk_probas = clf.predict_proba(chunk)
        if hasattr(chunk_probas, "to_cupy"):
            chunk_probas_gpu = chunk_probas.to_cupy()
        else:
            chunk_probas_gpu = cupy.asarray(chunk_probas)
        # Store only the positive class probability
        probas_list.append(chunk_probas_gpu[:, 1])

    probas_all = cupy.concatenate(probas_list)
    
    # Analyze probability distribution for logging
    probas_np = cupy.asnumpy(probas_all)
    percentiles = np.percentile(probas_np, [50, 75, 90, 95, 99])
    print(
        f'  Probability distribution - 50th: {percentiles[0]:.3f}, 75th: {percentiles[1]:.3f}, '
        f'90th: {percentiles[2]:.3f}, 95th: {percentiles[3]:.3f}, 99th: {percentiles[4]:.3f}'
    )

    y_all_gpu = _ensure_cupy_array(y)
    y_all_np = cupy.asnumpy(y_all_gpu)
    global_threshold, global_precision, global_recall, global_f1 = _select_threshold_for_fbeta(
        probas_np,
        y_all_np,
        precision_val_threshold,
        beta=precision_focus_beta,
        min_precision=global_min_precision,
    )
    if (
        (global_precision < global_min_precision or global_recall < global_min_recall)
        and precision_val_prec >= global_min_precision
    ):
        # Fall back to the validation threshold if global search cannot meet the precision floor
        global_threshold = precision_val_threshold
        global_precision = precision_val_prec
        global_recall = precision_val_rec
        global_f1 = (
            0.0
            if (global_precision == 0.0 and global_recall == 0.0)
            else 2.0 * global_precision * global_recall / (global_precision + global_recall)
        )
        print(
            '  Global search failed to meet precision/recall floor; using validation threshold instead.'
        )

    decision_threshold = max(0.05, min(0.99, global_threshold + threshold_offset))
    if not np.isclose(decision_threshold, global_threshold):
        print(
            '  Applied threshold offset %.3f; adjusted decision threshold to %.3f (was %.3f).'
            % (threshold_offset, decision_threshold, global_threshold)
        )

        decision_preds = probas_np >= decision_threshold
        positives_total = np.count_nonzero(y_all_np == 1)
        tp_decision = np.count_nonzero(decision_preds & (y_all_np == 1))
        fp_decision = np.count_nonzero(decision_preds) - tp_decision
        fn_decision = positives_total - tp_decision
        decision_precision = tp_decision / (tp_decision + fp_decision) if (tp_decision + fp_decision) > 0 else 0.0
        decision_recall = tp_decision / (tp_decision + fn_decision) if (tp_decision + fn_decision) > 0 else 0.0
        if decision_precision == 0.0 and decision_recall == 0.0:
            decision_f1 = 0.0
        else:
            decision_f1 = 2.0 * decision_precision * decision_recall / (decision_precision + decision_recall)
    else:
        decision_precision = global_precision
        decision_recall = global_recall
        decision_f1 = global_f1

    print(
        '  Final threshold summary (min precision %.2f): thr=%.3f -> precision=%.3f, recall=%.3f, F1=%.3f'
        % (global_min_precision, global_threshold, global_precision, global_recall, global_f1)
    )
    print(
        '  Using decision threshold: %.3f (validation precision=%.3f, recall=%.3f, F1=%.3f; '
        'precision-focused validation precision=%.3f, recall=%.3f, F-beta=%.3f; '
        'final precision=%.3f, recall=%.3f, F1=%.3f; provided threshold was %.3f)'
        % (
            decision_threshold,
            best_precision,
            best_recall,
            best_f1,
            precision_val_prec,
            precision_val_rec,
            precision_val_fbeta,
            decision_precision,
            decision_recall,
            decision_f1,
            threshold,
        )
    )
    predicted_mask = probas_np >= decision_threshold
    predicted_positive = int(predicted_mask.sum())
    true_positive = int(np.count_nonzero(predicted_mask & (y_all_np == 1)))
    est_precision = (true_positive / predicted_positive) if predicted_positive else 0.0
    print(
        f'  Threshold diagnostic: {predicted_positive} predicted matches, {true_positive} true positives (est. precision {est_precision:.3f})'
    )
    print('')
    sys.stdout.flush()

    # Second pass: apply threshold to get predictions
    predictions = (probas_all >= decision_threshold).astype('int32')

    # --- Memory Cleanup ---
    print('  Cleaning up memory before final result collection...')
    sys.stdout.flush()
    del probas_list
    del X
    del clf
    del X_sampled, y_sampled, X_train, y_train, X_test, y_test
    del X_matches, y_matches
    del y
    import gc
    gc.collect()
    
    # Vectorized result collection in chunks to avoid OOM
    predictions_series = cudf.Series(predictions)

    chunk_size = 1_000_000
    class_match_set = set()
    class_nonmatch_set = set()

    print(f'  Collecting results in {((len(rec_pairs)-1)//chunk_size)+1} chunks of size {chunk_size}...')
    sys.stdout.flush()

    for i in range(0, len(rec_pairs), chunk_size):
        rec_pairs_chunk = rec_pairs.iloc[i:i + chunk_size]
        predictions_chunk = predictions_series.iloc[i:i + chunk_size]
        mask_chunk = (predictions_chunk == 1)

        match_pairs_chunk = rec_pairs_chunk[mask_chunk]
        if not match_pairs_chunk.empty:
            class_match_set.update(map(tuple, match_pairs_chunk.to_pandas().to_records(index=False)))

        non_match_pairs_chunk = rec_pairs_chunk[~mask_chunk]
        if not non_match_pairs_chunk.empty:
            class_nonmatch_set.update(map(tuple, non_match_pairs_chunk.to_pandas().to_records(index=False)))

    print('  Classified %d record pairs as matches and %d as non-matches' % \
        (len(class_match_set), len(class_nonmatch_set)))
    print('')
    sys.stdout.flush()

    return class_match_set, class_nonmatch_set

# -----------------------------------------------------------------------------

# End of program.


