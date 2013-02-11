# -*- coding: utf-8 -*-
# configuration file for cluster.py

# target directory
target_directory = 'testout'

# use value between 0 and 1 (0 for unsmoothed variant)
decay_factor = 0.5

# language model order
ngram_order = 1

# minimal size of cluster: for LMs trained on low amounts of training data, there are possible (but undesirable) side effects:
# the algorithm is biased towards small LMs because unknown words are either ignored (without -unk) or given a high probability (with -unk)
# By not assigning any sentence pairs to small clusters, some clusters may end up empty, but the other clusters turn out better.
min_cluster_size = 100

# path to SRILM's ngram / ngram-count
ngram_count_cmd = 'ngram-count'
ngram_cmd = 'ngram'

# number of clusters
clusters = 2

# texts for bilingual clustering:
# need to be sentence-aligned.
# No normalization is required, but since a LM is used for clustering, it might work better with tokenized/lowercased texts.
textfile_s = 'test/1000.de'
textfile_t = 'test/1000.en'

# monolingual texts; sentences will be assigned to closest cluster after bitext clustering: use empty string to skip
textfile_s_mono = 'test/2000.de'
textfile_t_mono = 'test/2000.en'
