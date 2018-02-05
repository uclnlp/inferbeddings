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

        output_path = 'out_nli/{}/'.format(p)
        print('mkdir {}'.format(output_path))

        template = ""

        for (prefix, suffix) in [('train', ''), ('dev', 'd'), ('test', 't')]:
            template += """
python3 ./bin/nli-debug-cli.py --has-bos --has-unk --batch-size 128 --restore {}_0 -d data/snli/*{}* 2>&1 | tail -n 20 > {}/1{}.log
python3 ./bin/nli-debug-cli.py --has-bos --has-unk --batch-size 128 --restore {}_3 -d data/snli/*{}* 2>&1 | tail -n 20 > {}/2{}.log
python3 ./bin/nli-debug-cli.py --has-bos --has-unk --batch-size 128 --restore {}_6 -d data/snli/*{}* 2>&1 | tail -n 20 > {}/3{}.log
python3 ./bin/nli-debug-cli.py --has-bos --has-unk --batch-size 128 --restore {}_9 -d data/snli/*{}* 2>&1 | tail -n 20 > {}/4{}.log
python3 ./bin/nli-debug-cli.py --has-bos --has-unk --batch-size 128 --restore {}_12 -d data/snli/*{}* 2>&1 | tail -n 20 > {}/5{}.log
python3 ./bin/nli-debug-cli.py --has-bos --has-unk --batch-size 128 --restore {}_15 -d data/snli/*{}* 2>&1 | tail -n 20 > {}/6{}.log
""".format('{}', prefix, '{}', suffix)

        print(template.format(*([restore_path, output_path] * 18)))

