#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import os
import os.path


def cartesian_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def summary(configuration):
    kvs = sorted([(k, v) for k, v in configuration.items()], key=lambda e: e[0])
    return '_'.join([('%s=%s' % (k, v)) for (k, v) in kvs])


def to_cmd(c):
    prefix = ''
    command = '{}\n' \
              'python3 /home/ucl/eisuc296/workspace/neural-walker/bin/adv-cli.py' \
              ' --train /home/ucl/eisuc296/workspace/neural-walker/data/wn18/wordnet-mlj12-train.txt' \
              ' --valid /home/ucl/eisuc296/workspace/neural-walker/data/wn18/wordnet-mlj12-valid.txt' \
              ' --test /home/ucl/eisuc296/workspace/neural-walker/data/wn18/wordnet-mlj12-test.txt' \
              ' --nb-epochs {}' \
              ' --lr {}' \
              ' --nb-batches {}' \
              ' --model {}' \
              ' --similarity {}' \
              ' --margin {}' \
              ' --entity-embedding-size {}' \
              ' --clauses /home/ucl/eisuc296/workspace/neural-walker/data/wn18/clauses/clauses_0.9.pl' \
              ' --adv-lr {} --adv-nb-epochs {} --adv-weight {} ' \
              ' --adv-restart'.format(prefix, c['epochs'], c['lr'], c['batches'], c['model'], c['similarity'],
                                      c['margin'], c['embedding_size'], c['adv_lr'], c['adv_nb_epochs'], c['adv_weight'])
    return command


def to_logfile(c, path):
    outfile = "%s/wn18_adv_v2.%s.log" % (path, summary(c))
    return outfile


hyperparameters_space = dict(
    epochs=[100],
    optimizer=['adagrad'],
    lr=[.1],
    batches=[10],
    model=['ComplEx'],
    similarity=['dot'],
    margin=[1, 2, 5, 10],
    embedding_size=[10, 20, 50, 100, 150, 200, 300],

    adv_lr=[.01, .1, 1],
    adv_nb_epochs=[1, 5, 10, 25, 50, 100],
    adv_weight=[0, 1, 10, 100, 1000, 10000]
)

configurations = cartesian_product(hyperparameters_space)

path = '/home/ucl/eisuc296/workspace/neural-walker/logs/wn18_adv_v2/'

for job_id, cfg in enumerate(configurations):
    logfile = to_logfile(cfg, path)

    completed = False
    if os.path.isfile(logfile):
        with open(logfile, 'r') as f:
            content = f.read()
            completed = '### MICRO (test filtered)' in content

    if not completed:
        file_name = 'wn18_adv_v2_{}.job'.format(job_id)

        line = '{} >> {} 2>&1'.format(to_cmd(cfg), logfile)

        alias = """
alias python3="LD_LIBRARY_PATH='${HOME}/utils/libc6_2.17/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}' '${HOME}/utils/libc6_2.17/lib/x86_64-linux-gnu/ld-2.17.so' $(command -v python3)"
        """

        job_script = '#BSUB -W 2:00\n' \
                     '{}\n' \
                     'nvidia-smi > {}\n' \
                     'export CUDA_VISIBLE_DEVICES=`~/bin/lugpu.sh`\n' \
                     'export TMP_CUDA_VISIBLE_DEVICES={}\n' \
                     '{}\n'.format(alias, logfile, job_id % 4, line)

        with open(file_name, 'w') as f:
            f.write(job_script)
