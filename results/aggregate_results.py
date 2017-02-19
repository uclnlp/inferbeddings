import json
from pprint import pprint

with open("./results/model_to_adversarial_results.txt", "r") as f:
    model_to_adversarial_results = json.load(f)
    f.close()
with open("./results/model_to_standard_results.txt", "r") as f:
    model_to_standard_results = json.load(f)
    f.close()
with open("./results/model_to_naacl_results.txt", "r") as f:
    model_to_naacl_results = json.load(f)
    f.close()
with open("./results/model_to_logic_results.txt", "r") as f:
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
                    f.write("%s\t%s\t%s\t%1.1f\t%2.2f\n" %
                            (name, model, rename_metric(metric),
                             (int(i)+1)/10, float(x)))
    f.close()