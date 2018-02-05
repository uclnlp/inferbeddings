import sys
import glob

# $ ls models/snli/dam_1/acl/ | grep batch_dsearch_reg_v | python3 k.py | grep -v mkdir  | parallel -j 8 --delay 6


for s in sys.stdin:
    s = s.strip()
    a = s.split("_v")
    p = "_v{}".format(a[1])

    tmp = glob.glob('models/snli/dam_1/acl/{}/*_0.index'.format(s))

    if len(tmp) > 0:
        restore_path = tmp[0].split('_0.ind')[0]

        output_path = 'out_nli/k{}/'.format(p)
        print('mkdir {}'.format(output_path))

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

        print(template.format(*([restore_path, output_path] * (6 * 3))))

        print('mkdir {}/v1/'.format(output_path))
        temp = ''
        for size in ['100', '500', '1000', '2000', '3000', '4000', '5000', 'full']:
            for model in ['_dam', '_esim', '']:
                temp += """
python3 ./bin/nli-debug-cli.py --has-bos --has-unk --batch-size 128 --restore {}_0 -d data/snli/acl/v1/genadv/snli_genadv_{}{}_test.jsonl.gz 2>&1 | tail -n 20 > {}/v1/1_{}.log
python3 ./bin/nli-debug-cli.py --has-bos --has-unk --batch-size 128 --restore {}_3 -d data/snli/acl/v1/genadv/snli_genadv_{}{}_test.jsonl.gz 2>&1 | tail -n 20 > {}/v1/2_{}.log
python3 ./bin/nli-debug-cli.py --has-bos --has-unk --batch-size 128 --restore {}_6 -d data/snli/acl/v1/genadv/snli_genadv_{}{}_test.jsonl.gz 2>&1 | tail -n 20 > {}/v1/3_{}.log
python3 ./bin/nli-debug-cli.py --has-bos --has-unk --batch-size 128 --restore {}_9 -d data/snli/acl/v1/genadv/snli_genadv_{}{}_test.jsonl.gz 2>&1 | tail -n 20 > {}/v1/4_{}.log
python3 ./bin/nli-debug-cli.py --has-bos --has-unk --batch-size 128 --restore {}_12 -d data/snli/acl/v1/genadv/snli_genadv_{}{}_test.jsonl.gz 2>&1 | tail -n 20 > {}/v1/5_{}.log
python3 ./bin/nli-debug-cli.py --has-bos --has-unk --batch-size 128 --restore {}_15 -d data/snli/acl/v1/genadv/snli_genadv_{}{}_test.jsonl.gz 2>&1 | tail -n 20 > {}/v1/6_{}.log
""".format(*(['{}', size, model, '{}', size + model] * 6))

        print(temp.format(*([restore_path, output_path] * (6 * 8 * 3))))

        print('mkdir {}/v1.1/'.format(output_path))
        print('mkdir {}/v1.2/'.format(output_path))

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

        print(t.format(*([restore_path, output_path] * (6 * 2))))