#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import sys
import os
from subprocess import call


def select(df, metric="HITS@10", model="ComplEx"):
    if metric is not None:
        df = df[(df.Metric == metric)]
    if model is not None:
        df = df[(df.Model == model)]
    return df


def generate_table(df, metric="HITS@10", model="ComplEx", compile_pdf=True):
    tmp_df = select(df, metric, model)
    print("Creating ./results/results_%s_%s.tex" % (model, metric))

    methods = ["STD", "LOG", "SMPL", "ADV"]
    fractions = [round(fraction / 10.0, 1) for fraction in range(1, 11)]

    fractions_str = ' & '.join(["{}\%".format(int(f * 100)) for f in fractions])

    with open("./results/results_%s_%s.tex" % (model, metric), "w") as f:
        f.write("""\\documentclass{standalone}
\\usepackage{booktabs}
\\begin{document}
\\begin{tabular}{c|""" + ''.join(['c' for _ in range(len(fractions))]) + """}
      \\toprule
      Method & """ + fractions_str + """ \\\\
      \\midrule
""")
        _m2d = {}

        for method in methods:
            _m2d[method] = []
            tmp = tmp_df[tmp_df.Method == method]
            data = []
            for fraction in fractions:
                data.append(tmp[tmp.Fraction == fraction]["Result"].values[0])
            def to_str(x, idx):
                return "%2.2f" % x if idx > 0 else "%2.3f" % x
            data = [to_str(x, idx) for idx, x in enumerate(data)]
            _m2d[method] = data

        method_to_data = {m: [] for m in methods}
        for idx in range(len(_m2d[methods[0]])):
            max_value = max([float(_m2d[m][idx]) for m in methods])
            for method in methods:
                value = _m2d[method][idx]
                if float(value) == max_value:
                    value = '\\textbf{' + value + '}'
                method_to_data[method] += [str(value)]

        for method in methods:
            data = method_to_data[method]
            f.write(method + ' & ' + " & ".join(data) + "\\\\\n")
        f.write("""  \\bottomrule
\\end{tabular}
\\end{document}""")
        f.close()
        if compile_pdf:
            current_dir = os.getcwd()
            os.chdir("./results/")
            call(["pdflatex", "./results_%s_%s.tex" % (model, metric)])
            os.chdir(current_dir)

if __name__ == '__main__':
    df = pd.DataFrame.from_csv('./results/results.tsv', sep='\t',
                               index_col=None)
    if len(sys.argv) == 1:
        with open("./results/results.tex", "w") as f:
            f.write("""\\documentclass{article}
\\usepackage{graphicx}
\\begin{document}
""")
            for metric in ["MRR", "HITS@1", "HITS@3", "HITS@5", "HITS@10"]:
                for model in ["TransE", "DistMult", "ComplEx"]:
                    generate_table(df, metric, model)
                    f.write("\\begin{figure}\n")
                    f.write("\includegraphics[]{results_%s_%s}\\\\\n"
                            % (model, metric))
                    f.write("\\caption{%s %s}\n" % (model, metric))
                    f.write("\\end{figure}")
            f.write("\end{document}")
            f.close()
    else:
        metric = sys.argv[1] if len(sys.argv) > 1 else "HITS@10"
        model = sys.argv[2] if len(sys.argv) > 2 else "ComplEx"
        generate_table(df, metric, model)


