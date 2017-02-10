import logging

import sys


def main(argv):
    output_name = argv[-1]
    input_names = argv[:-1]
    import re

    folds = []

    for input_name in input_names:
        fold = []
        with open(input_name, 'r') as f:
            for line in f.readlines():
                if line != "\n":
                    split = re.split(",|\\(|\\)", line.strip())[:-1]
                    if len(split) == 2:
                        split = ["unary"] + split
                    rel, subj, obj = split[:3]
                    # print(split)
                    fold.append((subj, rel, obj))
        folds.append(fold)

    for fold_index, fold in enumerate(folds):
        with open("{}_fold_{}_test.tsv".format(output_name, fold_index), 'w') as csvout:
            for subj, rel, object in fold:
                if rel != "advisedBy":
                    csvout.write("{}\t{}\t{}\n".format(subj, rel, object))
        with open("{}_fold_{}_train.tsv".format(output_name, fold_index), 'w') as csvout:
            for other_fold_index in range(0, len(folds)):
                if other_fold_index != fold_index:
                    for subj, rel, object in fold:
                        csvout.write("{}\t{}\t{}\n".format(subj, rel, object))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
