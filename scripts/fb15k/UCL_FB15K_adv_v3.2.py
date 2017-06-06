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
    command = 'python3 {}/bin/kbp-cli.py' \
              ' --train {}/data/fb15k/freebase_mtr100_mte100-train.txt' \
              ' --valid {}/data/fb15k/freebase_mtr100_mte100-valid.txt' \
              ' --test {}/data/fb15k/freebase_mtr100_mte100-test.txt' \
              ' --clauses {}/data/fb15k/clauses/clauses_0.9_mins=1000_minis=1000.pl' \
              ' --nb-epochs {}' \
              ' --lr {}' \
              ' --nb-batches {}' \
              ' --model {}' \
              ' --similarity {}' \
              ' --margin {}' \
              ' --embedding-size {}' \
              ' --adv-lr {} --adv-init-ground --adversary-epochs {}' \
              ' --discriminator-epochs {} --adv-weight {} --adv-batch-size {}' \
              ' --predicate-norm 1'.format(_path, _path, _path, _path, _path,
                                           c['epochs'], c['lr'], c['batches'],
                                           c['model'], c['similarity'],
                                           c['margin'], c['embedding_size'],
                                           c['adv_lr'], c['adv_epochs'],
                                           c['disc_epochs'], c['adv_weight'], c['adv_batch_size'])
    return command


def to_logfile(c, path):
    outfile = "%s/ucl_fb15k_adv_v3.2.%s.log" % (path, summary(c))
    return outfile


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Generating experiments for the UCL cluster', formatter_class=formatter)
    argparser.add_argument('--debug', '-D', action='store_true', help='Debug flag')
    argparser.add_argument('--path', '-p', action='store', type=str, default=None, help='Path')

    args = argparser.parse_args(argv)

    hyperparameters_space = dict(
        epochs=[100],
        optimizer=['adagrad'],
        lr=[.1],
        batches=[10],
        model=['ComplEx'],
        similarity=['dot'],
        margin=[2, 5, 10],
        embedding_size=[20, 50, 100, 150, 200],
        adv_lr=[.1],
        adv_epochs=[0, 1, 10],
        disc_epochs=[1, 10],
        adv_weight=[0, 1, 10, 100, 1000, 10000],
        adv_batch_size=[1, 10, 100]
    )

    configurations = cartesian_product(hyperparameters_space)

    path = '/home/pminervi/workspace/inferbeddings/logs/ucl_fb15k_adv_v3.2/'
    if not os.path.exists(path):
        os.makedirs(path)

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
                file_name = 'ucl_fb15k_adv_v3.2_{}.job'.format(job_id)
                alias = ''
                job_script = '#$ -S /bin/bash\n' \
                             '#$ -wd /tmp/\n' \
                             '#$ -l h_vmem=10G,tmem=10G\n' \
                             '#$ -l h_rt=96:00:00\n' \
                             '{}\n{}\n'.format(alias, line)

                with open(file_name, 'w') as f:
                    f.write(job_script)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
