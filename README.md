bitext_clusterer
================

A project of the Computational Linguistics Group at the University of Zurich (http://www.cl.uzh.ch).

Project Homepage: http://github.com/rsennrich/bitext_clusterer

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation


ABOUT
-----

This program performs sentence-level k-means clustering for parallel texts based on language model similarity.

It follows the description in:
    Hirofumi Yamamoto and Eiichiro Sumita. 2007. Bilingual cluster based models for statistical machine translation. In Proceedings of the 2007 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning, pages 514–523.
with optional exponential smoothing as described in:
    Rico Sennrich. 2012. Mixture-modeling with unsupervised clusters for domain adaptation in statistical machine translation. In 16th Annual Conference of the European Association for Machine Translation (EAMT 2012).


REQUIREMENTS
------------

The program requires Python (2.6 or greater), and SRILM to train and apply language models. Set the paths to SRILM in `config.py`.


USAGE
-----

A number of options have to be set in `config.py`:

    - parallel input files
    - monolingual input files (optional)
    - output directory
    - number of target clusters

    - decay factor
    - language model n-gram order

After that, simply execute the program:

    python cluster.py

The clustered text files will be stored in the target directory (`.s` for source side, `.t` for target side):
    0.*, 1.*, ...: parallel text clusters
    0.lmtrain.*, 1.lmtrain.*, ...: additional monolingual text for each cluster (target side)
    lm*: temporary language model files


CONTACT
-------

For questions and feeback, please contact sennrich@cl.uzh.ch or use the GitHub repository.
