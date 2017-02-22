#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

with open("./results/model_to_adversarial_results.json", "r") as f:
    model_to_adversarial_results = json.load(f)
    f.close()
with open("./results/model_to_standard_results.json", "r") as f:
    model_to_standard_results = json.load(f)
    f.close()
with open("./results/model_to_naacl_results.json", "r") as f:
    model_to_naacl_results = json.load(f)
    f.close()
with open("./results/model_to_logic_results.json", "r") as f:
    model_to_logic_results = json.load(f)
    f.close()


def rename_metric(x):
    if x == "mr":
        return "MR"
    elif x == "mrr":
        return "MRR"
    elif x == "hits_at_1":
        return "HITS@1"
    elif x == "hits_at_3":
        return "HITS@3"
    elif x == "hits_at_5":
        return "HITS@5"
    elif x == "hits_at_10":
        return "HITS@10"

with open("./results/results.tsv", "w") as f:
    f.write("Method\tModel\tMetric\tFraction\tResult\n")

    for name, results in [
        ("ADV", model_to_adversarial_results),
        ("STD", model_to_standard_results),
        ("SMPL", model_to_naacl_results),
        ("LOG", model_to_logic_results)
    ]:
        for model in results:
            for metric in results[model]:
                result = results[model][metric]
                for i, x in enumerate(result):
                    renamed_metric = rename_metric(metric)
                    if renamed_metric == 'MRR':
                        f.write("%s\t%s\t%s\t%1.1f\t%2.3f\n" %
                                (name, model, renamed_metric,
                                 (int(i)+1)/10, float(x)))
                    else:
                        f.write("%s\t%s\t%s\t%1.1f\t%2.2f\n" %
                                (name, model, renamed_metric,
                                 (int(i)+1)/10, float(x)))
    f.close()