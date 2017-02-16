import logging

import sys


def main(argv):
    output_name = argv[-1]
    input_names = argv[:-1]
    import re

    folds = []
    persons = set()

    for input_name in input_names:
        fold = []
        with open(input_name, 'r') as f:
            for line in f.readlines():
                if line != "\n":
                    split = [s.strip() for s in re.split(",|\\(|\\)", line.strip())[:-1]]
                    if len(split) == 2:
                        split = ["unary"] + split
                    rel, subj, obj = split[:3]
                    # if not rel.startswith("same"):
                    # print(split)
                    fold.append((subj, rel, obj))
                    if subj.startswith("Person"):
                        persons.add(subj)
                    if obj.startswith("Person"):
                        persons.add(obj)
        folds.append(fold)

    sorted_persons = sorted(persons)

    for fold_index, fold in enumerate(folds):
        with open("{}_fold_{}_test.tsv".format(output_name, fold_index), 'w') as test_file, \
                open("{}_fold_{}_train.tsv".format(output_name, fold_index), 'w') as train_file:
            true_advised_by = set()
            fold_persons = set()
            for subj, rel, obj in fold:
                if subj.startswith("Person"):
                    fold_persons.add(subj)
                if obj.startswith("Person"):
                    fold_persons.add(obj)
                if rel == "advisedBy":
                    test_file.write("{}\t{}\t{}\t1\n".format(subj, rel, obj))
                    true_advised_by.add((subj, obj))
                else:
                    train_file.write("{}\t{}\t{}\n".format(subj, rel, obj))
            for other_fold_index in range(0, len(folds)):
                if other_fold_index != fold_index:
                    for subj, rel, obj in folds[other_fold_index]:
                        train_file.write("{}\t{}\t{}\n".format(subj, rel, obj))
            sorted_fold_persons = sorted(fold_persons)
            for person1 in sorted_fold_persons:
                for person2 in sorted_fold_persons:
                    if person1 != person2 and (person1, person2) not in true_advised_by:
                        test_file.write("{}\t{}\t{}\t0\n".format(person1, "advisedBy", person2))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
