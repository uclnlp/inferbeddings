#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import os
import os.path

import sys
import argparse
import logging


def cartesian_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def summary(configuration):
    kvs = sorted([(k, v) for k, v in configuration.items()], key=lambda e: e[0])
    return '_'.join([('%s=%s' % (k, v)) for (k, v) in kvs])


def to_cmd(c, _path=None):
    if _path is None:
        _path = '/home/pminervi/workspace/inferbeddings/'
    command = 'python3 {}/bin/adv-cli.py' \
              ' --train {}/data/music_mte10_5k/music_mte10-train.tsv' \
              ' --valid {}/data/music_mte10_5k/music_mte10-valid.tsv' \
              ' --test {}/data/music_mte10_5k/music_mte10-test.tsv' \
              ' --clauses {}/data/music_mte10_5k/clauses/clauses_B=100_C=0.8.pl' \
              ' --nb-epochs {}' \
              ' --lr {}' \
              ' --nb-batches {}' \
              ' --model {}' \
              ' --similarity {}' \
              ' --margin {}' \
              ' --embedding-size {}' \
              ' --head-subsample-size {}' \
              ' --loss {}' \
              ' --adv-lr {} --adv-init-ground --adversary-epochs {}' \
              ' --discriminator-epochs {} --adv-weight {} --adv-batch-size {} --adv-pooling {}' \
              ''.format(_path, _path, _path, _path, _path,
                        c['epochs'], c['lr'], c['batches'],
                        c['model'], c['similarity'],
                        c['margin'], c['embedding_size'],
                        c['subsample_size'],
                        c['loss'],
                        c['adv_lr'], c['adv_epochs'],
                        c['disc_epochs'], c['adv_weight'], c['adv_batch_size'], c['adv_pooling'])
    return command


def to_logfile(c, path):
    outfile = "%s/ucl_music_adv_xheadshot_v1.%s.log" % (path, summary(c))
    return outfile


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Generating experiments for the UCL cluster', formatter_class=formatter)
    argparser.add_argument('--debug', '-D', action='store_true', help='Debug flag')
    argparser.add_argument('--path', '-p', action='store', type=str, default=None, help='Path')

    args = argparser.parse_args(argv)

    hyperparameters_space_transe = dict(
        epochs=[100],
        optimizer=['adagrad'],
        lr=[.1],
        batches=[10],
        model=['TransE'],
        similarity=['l1', 'l2', 'dot'],
        margin=[1],  # margin=[1, 2, 5, 10],
        embedding_size=[20, 50, 100, 150, 200],
        loss=['hinge'],
        subsample_size=[.1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
        adv_lr=[.1],
        adv_epochs=[0, 10],
        disc_epochs=[10],
        adv_weight=[0, 1, 100, 10000, 1000000],
        adv_batch_size=[1, 10, 100],
        adv_pooling=['sum', 'mean', 'max', 'logsumexp']
    )

    hyperparameters_space_distmult_complex = dict(
        epochs=[100],
        optimizer=['adagrad'],
        lr=[.1],
        batches=[10],
        model=['DistMult', 'ComplEx'],
        similarity=['dot'],
        margin=[1],  # margin=[1, 2, 5, 10],
        embedding_size=[20, 50, 100, 150, 200],
        loss=['hinge'],
        subsample_size=[.1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
        adv_lr=[.1],
        adv_epochs=[0, 10],
        disc_epochs=[10],
        adv_weight=[0, 1, 100, 10000, 1000000],
        adv_batch_size=[1, 10, 100],
        adv_pooling=['sum', 'mean', 'max', 'logsumexp']
    )

    configurations_transe = cartesian_product(hyperparameters_space_transe)
    configurations_distmult_complex = cartesian_product(hyperparameters_space_distmult_complex)

    path = '/home/pminervi/workspace/inferbeddings/logs/ucl_music_adv_xheadshot_v1/'
    if not os.path.exists(path):
        os.makedirs(path)

    configurations = list(configurations_distmult_complex)  # + list(configurations_transe)

    for job_id, cfg in enumerate(configurations):
        logfile = to_logfile(cfg, path)

        completed = False
        if os.path.isfile(logfile):
            with open(logfile, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                completed = '### MICRO (test filtered)' in content

        if not completed:
            line = '{} >> {} 2>&1'.format(to_cmd(cfg, _path=args.path), logfile)

            if args.debug:
                print(line)
            else:
                file_name = 'ucl_music_adv_xheadshot_v1_{}.job'.format(job_id)
                alias = ''
                job_script = '#$ -S /bin/bash\n' \
                             '#$ -wd /tmp/\n' \
                             '#$ -l h_vmem=8G,tmem=8G\n' \
                             '#$ -l h_rt=24:00:00\n' \
                             'export LANG="en_GB.UTF-8"\n' \
                             '{}\n{}\n'.format(alias, line)

                with open(file_name, 'w') as f:
                    f.write(job_script)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
