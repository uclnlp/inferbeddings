import pandas as pd
import sys


def select(df, metric="HITS@10", model="ComplEx"):
    if metric is not None:
        df = df[(df.Metric == metric)]
    if model is not None:
        df = df[(df.Model == model)]
    return df


def generate_table(df, metric="HITS@10", model="ComplEx"):
    tmp_df = select(df, metric, model)
    print("Creating ./results/results_%s_%s.tex" % (model, metric))
    with open("./results/results_%s_%s.tex" % (model, metric), "w") as f:
        f.write("""\\documentclass{standalone}
\\usepackage{booktabs}
\\begin{document}
\\begin{tabular}{r|rrrr}
      \\toprule
      Fraction & STD & SMPL & LOG & ADV\\\\
      \\midrule
    """)

        for fraction in range(1, 11):
            fraction = round(fraction/10.0, 1)
            tmp = tmp_df[tmp_df.Fraction == fraction]
            data = []
            for method in ["STD", "SMPL", "LOG", "ADV"]:
                data.append(tmp[tmp.Method == method]["Result"].values[0])
            max_ix = []
            max_val = 10e10 if metric == "MR" else 0.0
            for i, val in enumerate(data):
                if (metric != "MR" and val > max_val) or \
                        (metric == "MR" and val < max_val):
                    max_ix = [i]
                    max_val = val
                elif val == max_val:
                    max_ix.append(i)
            data = ["%2.2f" % x for x in data]
            if len(max_ix) == 1:
                data[max_ix[0]] = "\\textbf{" + data[max_ix[0]] + "}"
            else:
                for ix in max_ix:
                    data[ix] = "\\emph{" + data[ix] + "}"
            data = [str(fraction)] + data
            f.write("  " + " & ".join(data) + "\\\\\n")
        f.write("""  \\bottomrule
\\end{tabular}
\\end{document}""")
        f.close()

if __name__ == '__main__':
    df = pd.DataFrame.from_csv('./results/results.tsv', sep='\t',
                               index_col=None)

    if len(sys.argv) == 1:
        for metric in ["MR", "MRR", "HITS@1", "HITS@3", "HITS@5", "HITS@10"]:
            for model in ["TransE", "DistMult", "ComplEx"]:
                generate_table(df, metric, model)
    else:
        metric = sys.argv[1] if len(sys.argv) > 1 else "HITS@10"
        model = sys.argv[2] if len(sys.argv) > 2 else "ComplEx"
        generate_table(df, metric, model)


