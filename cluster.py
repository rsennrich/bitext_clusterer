#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright: University of Zurich
# Author: Rico Sennrich

"""sentence clustering for parallel texts"""


from __future__ import division
import sys
import os
import random
import math
from subprocess import Popen, PIPE
from collections import defaultdict
from config import decay_factor, ngram_cmd, ngram_count_cmd, min_cluster_size, ngram_order


class Cluster():
    '''offers two methods that can/should be called: cluster() and cluster_monolingual().
       monolingual_clustering() requires that cluster() has been executed before,
       but not necessarily in the same Python session (the cluster files need to exist)'''

    def __init__(self, clusters, textfile_s, textfile_t, tempfolder):
        self.clusters = clusters
        self.textfile_s = textfile_s
        self.textfile_t = textfile_t

        self.n = 0
        f = open(textfile_s, 'r')
        while f.readline():
            self.n += 1

        ## can be used for evaluation purposes: give a number/label to each sentence,
        ## which identifies the data set it originally comes from
        #self.goldclasses = ['A']*500+['B']*500
        self.goldclasses = False
        self.number = defaultdict(int)

        self.tempfolder = tempfolder
        self.current = {}

    def cluster(self):
        '''initialize clusters, then iteratively calculate means and assign
        sentence pairs until convergence'''

        self.initialize_clusters()

        lm_entropy = float('inf')
        old_entropy = float('inf')
        iteration = 0
        while lm_entropy == float('inf') or old_entropy - lm_entropy > 1:
            old_entropy = lm_entropy
            self.train_lms()
            clusters = dict([(i, {'s':self.textfile_s, 't':self.textfile_t}) for i in range(self.clusters)])
            scores = self.score_lms(clusters)
            lm_entropy, class_entropy = self.create_clusters(scores)
            iteration += 1
            if self.goldclasses:
                class_info = ' - class entropy: {0}'.format(class_entropy)
            else:
                class_info = ''
            sys.stderr.write('iteration {0}: LM entropy: {1} {2}\n'.format(iteration, lm_entropy, class_info))


    def initialize_clusters(self):
        '''distribute sentence pairs randomly over all clusters'''

        files_s = []
        files_t = []
        for i in range(self.clusters):
            files_s.append(open(os.path.join(self.tempfolder, '{0}.s'.format(i)), 'w'))
            files_t.append(open(os.path.join(self.tempfolder, '{0}.t'.format(i)), 'w'))

        order = range(self.n)
        random.shuffle(order)

        j = 0
        k = (len(order)//self.clusters)+1

        # assign each sentence number to a cluster
        for i in range(self.clusters):
            for x in order[j:k]:
                self.current[x] = i
            j = k
            k += len(order)//self.clusters

        for x in order[j:k]: # add remaining sentences
            self.current[x] = i

        text_s = open(self.textfile_s, 'r')
        text_t = open(self.textfile_t, 'r')

        # write sentences to corresponding file
        for i in range(self.n):
            cluster = self.current[i]
            self.number[cluster] += 1

            line_s = text_s.readline()
            line_t = text_t.readline()

            files_s[cluster].write(line_s)
            files_t[cluster].write(line_t)


    def train_lms(self, train_global_model=False):
        '''train a language model for each cluster/language with SRILM'''

        cmd = [ngram_count_cmd, '-order', str(ngram_order), '-text', '', '-lm', '', '-unk']

        sys.stderr.write('training LM')
        if train_global_model:
            todo = ['all']
        else:
            todo = range(self.clusters)
        for i in todo:
            for lang in ['s', 't']:
                #sys.stderr.write('Training LM{0}.{1}\n'.format(i, lang))
                sys.stderr.write('.')
                if train_global_model:
                    if lang == 's':
                        cmd[4] = self.textfile_s
                    elif lang == 't':
                        cmd[4] = config.textfile_t_mono
                else:
                    cmd[4] = os.path.join(self.tempfolder, '{0}.{1}'.format(i, lang))
                cmd[6] = os.path.join(self.tempfolder, 'lm{0}.{1}'.format(i, lang))

                lm = Popen(cmd, stderr=open('/dev/null', 'w'))
                lm.wait()
        sys.stderr.write('\n')


    def smooth_and_assign(self, i, dist, scores):
        '''perform exponential smoothing and mark it if cluster 'i' is currently closest
           to a data point. Must be called for each cluster.'''

        cache = 0
        dist_smoothed = defaultdict(float)

        if decay_factor:
            for j in range(len(dist)-1, -1, -1):

                cache *= decay_factor
                dist_smoothed[j] += cache
                cache += dist[j]

        cache = 0
        for j in range(len(dist)):

            cache *= decay_factor
            dist_smoothed[j] += dist[j] + cache
            cache += dist[j]

            if dist_smoothed[j] < scores[j][0]:
                scores[j] = (dist_smoothed[j], i)

        return scores


    def score_lms(self, clusters):
        '''assignment step: score each sentence pair with each cluster-specific LM,
        and return best cluster for each sentence pair.'''

        scores = defaultdict(lambda: (float("inf"), 0))
        sys.stderr.write('scoring LMs')

        for i, texts in clusters.items():

            if i != 'all' and self.number[i] < min_cluster_size:
                sys.stderr.write('Skipping cluster ' + str(i) + ' (too small)\n')
                continue

            dist = defaultdict(float)

            for lang in texts:

                cmd = [ngram_cmd,
                    '-order', str(ngram_order),
                    '-lm', os.path.join(self.tempfolder, 'lm{0}.{1}'.format(i, lang)),
                    '-ppl', '-',
                    '-debug', '1',
                    '-unk']

                textin = open(texts[lang])
                scorer = Popen(cmd, stdin=textin, stdout=PIPE, stderr=open('/dev/null', 'w'))
                sys.stderr.write('.')

                # read sentence length and log-likelihood from SRILM output
                for k, line in enumerate(scorer.stdout):
                    if k % 4 == 0 and line.startswith('file -:'):
                        break
                    elif k % 4 == 1:
                        length = int(line.split()[2])
                    elif k % 4 == 2:
                        j = k // 4
                        dist[j] -= (float(line.split()[3]))/length

            scores = self.smooth_and_assign(i, dist, scores)

        sys.stderr.write('\n')

        return scores


    def create_clusters(self, scores_lm):
        '''given a list with the best cluster for each sentence pair,
        write new files and calculate LM entropy'''

        files_s = []
        files_t = []
        self.current = {}
        for i in range(self.clusters):
            files_s.append(open(os.path.join(self.tempfolder, '{0}.s'.format(i)), 'w'))
            files_t.append(open(os.path.join(self.tempfolder, '{0}.t'.format(i)), 'w'))

        entropy_lm = 0
        entropy_class = 0
        self.number = defaultdict(int)
        labels = defaultdict(lambda:defaultdict(int))

        text_s = open(self.textfile_s, 'r')
        text_t = open(self.textfile_t, 'r')

        for i in range(self.n):

            line_s = text_s.readline()
            line_t = text_t.readline()

            score, best_cluster = scores_lm[i]
            entropy_lm += score

            self.number[best_cluster] += 1
            self.current[i] = best_cluster

            files_s[best_cluster].write(line_s)
            files_t[best_cluster].write(line_t)

            if self.goldclasses:
                labels[best_cluster][self.goldclasses[i]] += 1

        if self.goldclasses:
            entropy_class = self.calc_class_entropy(labels)

        for i in range(self.clusters):
            print "cluster {0}: {1} sentences".format(i, self.number[i])

        return entropy_lm, entropy_class


    def calc_class_entropy(self, clusters):
        '''Given a label for each sentence pair (e.g. its domain),
        calculate the entropy of the resulting clusters'''

        entropy = 0
        total = sum(sum(clusters[c].values()) for c in clusters)
        for i in clusters:
            entropy_cluster = 0
            total_cluster = sum(clusters[i].values())
            for cl in clusters[i]:
                prob = clusters[i][cl]/total_cluster
                entropy_cluster += -prob*math.log(prob, 2)

            entropy += entropy_cluster * total_cluster/total

        return entropy


    def monolingual_clustering(self, lang, text):
        '''monolingual data can be clustered as well. Uses behavior described by Yamamato and Sumita (2008)
        to only assign sentences that are closer to at least one of the bilingual clusters than to a general LM.
        Requires finished bilingual clustering (which one gets from executing cluster() ).'''

        clusters = dict([(i, {lang:text}) for i in range(c.clusters) + ['all']])

        self.train_lms(train_global_model=True)
        scores = self.score_lms(clusters)

        files = []
        for i in range(self.clusters):
            files.append(open(os.path.join(self.tempfolder, '{0}.lmtrain.{1}'.format(i, lang)), 'w'))

        for i, line in enumerate(open(text, 'r')):

            score, best_cluster = scores[i]
            if best_cluster == 'all':
                continue
            else:
                files[best_cluster].write(line)


if __name__ == '__main__':
    import config

    if os.path.exists(config.target_directory):
        sys.stderr.write('ERROR: target directory exists. Please choose different target directory, or delete it\n')
        sys.exit(1)
    else:
        os.mkdir(config.target_directory)

    if not 0 <= config.decay_factor < 1:
        sys.stderr.write('ERROR: decay factor may not be negative, and must be smaller than 1\n')

    c = Cluster(config.clusters, config.textfile_s, config.textfile_t, config.target_directory)
    c.cluster()

    if config.textfile_s_mono:
        sys.stderr.write('Clustering additional monolingual data (source side)\n')
        c.monolingual_clustering('s', config.textfile_s_mono)
    if config.textfile_t_mono:
        sys.stderr.write('Clustering additional monolingual data (target side)\n')
        c.monolingual_clustering('t', config.textfile_t_mono)
