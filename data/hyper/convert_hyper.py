import logging

import sys


def main(argv):
    input_file, output_file = argv
    import csv

    with open(input_file, 'r') as tsvin, open(output_file, 'w') as csvout:
        tsvin = csv.reader(tsvin, delimiter='\t')
        tsvout = csv.writer(csvout, delimiter='\t')
        for row in tsvin:
            subset, superset, label = row
            tsvout.writerow([subset.lower(), "isa", superset.lower(), 1 if label == "True" else 0])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
