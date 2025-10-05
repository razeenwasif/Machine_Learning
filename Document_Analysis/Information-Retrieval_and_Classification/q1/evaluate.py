import os
from trectools import TrecQrel, TrecRun


qrels_file = os.path.join('gov', 'qrels', 'gov.qrels')
qrels = TrecQrel(qrels_file)

run = TrecRun('retrieved_documents.txt')
res = run.evaluate_run(qrels, per_query=False)

# See the "Major measures" in
# https://www-nlpir.nist.gov/projects/trecvid/trecvid.tools/trec_eval_video/A.README
metrics = ['map', 'Rprec', 'recip_rank', 'P_5', 'P_10', 'P_20']
for metric in metrics:
    print(f'{metric}: {res.get_results_for_metric(metric)["all"]:.4f}')

