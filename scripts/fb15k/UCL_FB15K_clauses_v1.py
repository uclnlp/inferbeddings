#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
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
              ' --train {}/data/fb15k/freebase_mtr100_mte100-train.txt' \
              ' --valid {}/data/fb15k/freebase_mtr100_mte100-valid.txt' \
              ' --test {}/data/fb15k/freebase_mtr100_mte100-test.txt' \
              ' --clauses {}/data/fb15k/clauses/{}' \
              ' --nb-epochs {}' \
              ' --lr {}' \
              ' --nb-batches {}' \
              ' --model {}' \
              ' --similarity {}' \
              ' --margin {}' \
              ' --embedding-size {}' \
              ' --adv-lr {} --adv-init-ground --adversary-epochs {}' \
              ' --discriminator-epochs {} --adv-weight {} --adv-batch-size {}' \
              ''.format(_path, _path, _path, _path, _path,
                        c['clausefile'],
                        c['epochs'], c['lr'], c['batches'],
                        c['model'], c['similarity'],
                        c['margin'], c['embedding_size'],
                        c['adv_lr'], c['adv_epochs'],
                        c['disc_epochs'], c['adv_weight'], c['adv_batch_size'])
    return command


def to_logfile(c, path):
    outfile = "%s/ucl_fb15k_clauses_v1.%s.log" % (path, summary(c))
    return outfile


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Generating experiments for the UCL cluster', formatter_class=formatter)
    argparser.add_argument('--debug', '-D', action='store_true', help='Debug flag')
    argparser.add_argument('--path', '-p', action='store', type=str, default=None, help='Path')

    args = argparser.parse_args(argv)

    hyperparameters_space_transe = dict(
        clausefile=['clauses_highconf_highsupp.pl', 'clauses_highconf_lowsupp.pl', 'clauses_lowconf_highsupp.pl', 'clauses_lowconf_lowsupp.pl'],
        epochs=[100],
        optimizer=['adagrad'],
        lr=[.1],
        batches=[10],
        model=['TransE'],
        similarity=['l1'],
        margin=[1],
        embedding_size=[100],
        adv_lr=[.1],
        adv_epochs=[0, 1, 10],
        disc_epochs=[1, 10],
        adv_weight=[0, 1, 10, 100, 1000, 10000],
        adv_batch_size=[10]
    )

    hyperparameters_space_distmult_complex = dict(
        clausefile=['clauses_highconf_highsupp.pl', 'clauses_highconf_lowsupp.pl', 'clauses_lowconf_highsupp.pl', 'clauses_lowconf_lowsupp.pl'],
        epochs=[100],
        optimizer=['adagrad'],
        lr=[.1],
        batches=[10],
        model=['ComplEx', 'DistMult'],
        similarity=['dot'],
        margin=[1],
        embedding_size=[100],
        adv_lr=[.1],
        adv_epochs=[0, 1, 10],
        disc_epochs=[1, 10],
        adv_weight=[0, 1, 10, 100, 1000, 10000],
        adv_batch_size=[10]
    )

    configurations_transe = cartesian_product(hyperparameters_space_transe)
    configurations_distmult_complex = cartesian_product(hyperparameters_space_distmult_complex)

    path = '/home/pminervi/workspace/inferbeddings/logs/ucl_fb15k_clauses_v1/'
    if not os.path.exists(path):
        os.makedirs(path)

    configurations = list(configurations_transe) + list(configurations_distmult_complex)

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
                file_name = 'ucl_fb15k_clauses_v1_{}.job'.format(job_id)
                alias = ''
                job_script = '#$ -S /bin/bash\n' \
                             '#$ -wd /tmp/\n' \
                             '#$ -l h_vmem=6G,tmem=6G\n' \
                             '#$ -l h_rt=24:00:00\n' \
                             '{}\n{}\n'.format(alias, line)

                with open(file_name, 'w') as f:
                    f.write(job_script)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
