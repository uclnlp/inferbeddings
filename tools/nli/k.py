#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob

# $ ls models/snli/dam_1/acl/ | grep batch_dsearch_reg_v | python3 k.py | grep -v mkdir  | parallel -j 8 --delay 6

out_str = ''

for s in sys.stdin:
    s = s.strip()
    a = s.split("_v")
    p = "_v{}".format(a[1])

    tmp = glob.glob('models/snli/dam_1/acl/{}/*_0.index'.format(s))

    if len(tmp) > 0:
        restore_path = tmp[0].split('_0.ind')[0]

        output_path = 'out_nli/k{}/'.format(p)
        out_str += """
mkdir -p {}
""".format(output_path)

        template = ''
        for (prefix, suffix) in [('train', ''), ('dev', 'd'), ('test', 't')]:
            template += """

python3 ./bin/nli-debug-cli.py --has-bos --has-unk --batch-size 128 --restore {}_0 -d data/snli/*{}* 2>&1 | tail -n 20 > {}/1{}.log
python3 ./bin/nli-debug-cli.py --has-bos --has-unk --batch-size 128 --restore {}_3 -d data/snli/*{}* 2>&1 | tail -n 20 > {}/2{}.log
python3 ./bin/nli-debug-cli.py --has-bos --has-unk --batch-size 128 --restore {}_6 -d data/snli/*{}* 2>&1 | tail -n 20 > {}/3{}.log
python3 ./bin/nli-debug-cli.py --has-bos --has-unk --batch-size 128 --restore {}_9 -d data/snli/*{}* 2>&1 | tail -n 20 > {}/4{}.log
python3 ./bin/nli-debug-cli.py --has-bos --has-unk --batch-size 128 --restore {}_12 -d data/snli/*{}* 2>&1 | tail -n 20 > {}/5{}.log
python3 ./bin/nli-debug-cli.py --has-bos --has-unk --batch-size 128 --restore {}_15 -d data/snli/*{}* 2>&1 | tail -n 20 > {}/6{}.log

""".format(*(['{}', prefix, '{}', suffix] * 6))

        out_str += template.format(*([restore_path, output_path] * (6 * 3)))

        out_str += """
mkdir -p {}/v1/
""".format(output_path)

        temp = ''
        for size in ['100', '500', '1000', '2000', '3000', '4000', '5000', 'full']:
            for model in ['dam_', 'esim_', '']:
                temp += """
python3 ./bin/nli-debug-cli.py --has-bos --has-unk --batch-size 128 --restore {}_0 -d data/snli/acl/v1/genadv/snli_genadv_{}{}_dev.jsonl.gz 2>&1 | tail -n 20 > {}/v1/1_{}_dev.log
python3 ./bin/nli-debug-cli.py --has-bos --has-unk --batch-size 128 --restore {}_3 -d data/snli/acl/v1/genadv/snli_genadv_{}{}_dev.jsonl.gz 2>&1 | tail -n 20 > {}/v1/2_{}_dev.log
python3 ./bin/nli-debug-cli.py --has-bos --has-unk --batch-size 128 --restore {}_6 -d data/snli/acl/v1/genadv/snli_genadv_{}{}_dev.jsonl.gz 2>&1 | tail -n 20 > {}/v1/3_{}_dev.log
python3 ./bin/nli-debug-cli.py --has-bos --has-unk --batch-size 128 --restore {}_9 -d data/snli/acl/v1/genadv/snli_genadv_{}{}_dev.jsonl.gz 2>&1 | tail -n 20 > {}/v1/4_{}_dev.log
python3 ./bin/nli-debug-cli.py --has-bos --has-unk --batch-size 128 --restore {}_12 -d data/snli/acl/v1/genadv/snli_genadv_{}{}_dev.jsonl.gz 2>&1 | tail -n 20 > {}/v1/5_{}_dev.log
python3 ./bin/nli-debug-cli.py --has-bos --has-unk --batch-size 128 --restore {}_15 -d data/snli/acl/v1/genadv/snli_genadv_{}{}_dev.jsonl.gz 2>&1 | tail -n 20 > {}/v1/6_{}_dev.log
        """.format(*(['{}', model, size, '{}', model + size] * 6))

        out_str += temp.format(*([restore_path, output_path] * (6 * 8 * 3)))

        temp = ''
        for size in ['100', '500', '1000', '2000', '3000', '4000', '5000', 'full']:
            for model in ['dam_', 'esim_', '']:
                temp += """
python3 ./bin/nli-debug-cli.py --has-bos --has-unk --batch-size 128 --restore {}_0 -d data/snli/acl/v1/genadv/snli_genadv_{}{}_test.jsonl.gz 2>&1 | tail -n 20 > {}/v1/1_{}_test.log
python3 ./bin/nli-debug-cli.py --has-bos --has-unk --batch-size 128 --restore {}_3 -d data/snli/acl/v1/genadv/snli_genadv_{}{}_test.jsonl.gz 2>&1 | tail -n 20 > {}/v1/2_{}_test.log
python3 ./bin/nli-debug-cli.py --has-bos --has-unk --batch-size 128 --restore {}_6 -d data/snli/acl/v1/genadv/snli_genadv_{}{}_test.jsonl.gz 2>&1 | tail -n 20 > {}/v1/3_{}_test.log
python3 ./bin/nli-debug-cli.py --has-bos --has-unk --batch-size 128 --restore {}_9 -d data/snli/acl/v1/genadv/snli_genadv_{}{}_test.jsonl.gz 2>&1 | tail -n 20 > {}/v1/4_{}_test.log
python3 ./bin/nli-debug-cli.py --has-bos --has-unk --batch-size 128 --restore {}_12 -d data/snli/acl/v1/genadv/snli_genadv_{}{}_test.jsonl.gz 2>&1 | tail -n 20 > {}/v1/5_{}_test.log
python3 ./bin/nli-debug-cli.py --has-bos --has-unk --batch-size 128 --restore {}_15 -d data/snli/acl/v1/genadv/snli_genadv_{}{}_test.jsonl.gz 2>&1 | tail -n 20 > {}/v1/6_{}_test.log
""".format(*(['{}', model, size, '{}', model + size] * 6))

        out_str += temp.format(*([restore_path, output_path] * (6 * 8 * 3)))

        out_str += """
mkdir -p {}/v1.1/
""".format(output_path)

        out_str += """
mkdir -p {}/v1.2/
""".format(output_path)

        t = """
python3 ./bin/nli-debug-cli.py --has-bos --has-unk --batch-size 128 --restore {}_0 -d data/snli/acl/v1/v1.1_edited.jsonl.gz 2>&1 | tail -n 20 > {}/v1.1/1.log
python3 ./bin/nli-debug-cli.py --has-bos --has-unk --batch-size 128 --restore {}_3 -d data/snli/acl/v1/v1.1_edited.jsonl.gz 2>&1 | tail -n 20 > {}/v1.1/2.log
python3 ./bin/nli-debug-cli.py --has-bos --has-unk --batch-size 128 --restore {}_6 -d data/snli/acl/v1/v1.1_edited.jsonl.gz 2>&1 | tail -n 20 > {}/v1.1/3.log
python3 ./bin/nli-debug-cli.py --has-bos --has-unk --batch-size 128 --restore {}_9 -d data/snli/acl/v1/v1.1_edited.jsonl.gz 2>&1 | tail -n 20 > {}/v1.1/4.log
python3 ./bin/nli-debug-cli.py --has-bos --has-unk --batch-size 128 --restore {}_12 -d data/snli/acl/v1/v1.1_edited.jsonl.gz 2>&1 | tail -n 20 > {}/v1.1/5.log
python3 ./bin/nli-debug-cli.py --has-bos --has-unk --batch-size 128 --restore {}_15 -d data/snli/acl/v1/v1.1_edited.jsonl.gz 2>&1 | tail -n 20 > {}/v1.1/6.log

python3 ./bin/nli-debug-cli.py --has-bos --has-unk --batch-size 128 --restore {}_0 -d data/snli/acl/v1/v1.2_edited.jsonl.gz 2>&1 | tail -n 20 > {}/v1.2/1.log
python3 ./bin/nli-debug-cli.py --has-bos --has-unk --batch-size 128 --restore {}_3 -d data/snli/acl/v1/v1.2_edited.jsonl.gz 2>&1 | tail -n 20 > {}/v1.2/2.log
python3 ./bin/nli-debug-cli.py --has-bos --has-unk --batch-size 128 --restore {}_6 -d data/snli/acl/v1/v1.2_edited.jsonl.gz 2>&1 | tail -n 20 > {}/v1.2/3.log
python3 ./bin/nli-debug-cli.py --has-bos --has-unk --batch-size 128 --restore {}_9 -d data/snli/acl/v1/v1.2_edited.jsonl.gz 2>&1 | tail -n 20 > {}/v1.2/4.log
python3 ./bin/nli-debug-cli.py --has-bos --has-unk --batch-size 128 --restore {}_12 -d data/snli/acl/v1/v1.2_edited.jsonl.gz 2>&1 | tail -n 20 > {}/v1.2/5.log
python3 ./bin/nli-debug-cli.py --has-bos --has-unk --batch-size 128 --restore {}_15 -d data/snli/acl/v1/v1.2_edited.jsonl.gz 2>&1 | tail -n 20 > {}/v1.2/6.log
        """.format(*(['{}', '{}'] * 12))

        out_str += t.format(*([restore_path, output_path] * (6 * 2)))

out_lst = out_str.split("\n")

for e in out_lst:
    e_split = e.split(" ")
    if len(e_split) > 0:
        path = e_split[-1]

        if 'log' in path:
            if os.path.isfile(path):
                with open(path, 'r') as f:
                    data = f.read()

                if '(True) AND NOT(S1' not in data:
                    print(e)
            else:
                print(e)

        if 'mkdir' in e:
            if not os.path.isdir(path):
                print(e)
