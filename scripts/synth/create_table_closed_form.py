import numpy as np

from collections import defaultdict

results = '/Users/tdmeeste/workspace/inferbeddings/logs/synth/synth_paper_closedform_aggregated.txt'

models_lst = ['DistMult', 'ComplEx']
clauses_lst = ['symm', 'impl', 'impl_inv']
confs_lst = ['0.0']
versions_lst = ['v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9']
disc_epochs_lst = ['10']
clause_weight_lst = ['1.0'] #['0.01','0.1','1.0','10.0', '100.0', '1000.0']

def string(s):
    return {'TransE': r"\mdl{TransE}",
            'DistMult' : r"\mdl{DistM.}",
            'ComplEx' : r"\mdl{Compl.}",
            'symm': r"\multirow{ 2}{*}{ $\begin{array} {l@{}} r(X_1, X_2) \\  \quad\Rightarrow r(X_2, X_1) \end{array}$ }",
            'impl': r"\multirow{ 2}{*}{ $\begin{array} {l@{}} r(X_1, X_2) \\  \quad\Rightarrow s(X_1, X_2) \end{array}$ }",
            'impl_inv' : r"\multirow{ 2}{*}{ $\begin{array} {l@{}} r(X_1, X_2) \\  \quad\Rightarrow s(X_2, X_1) \end{array}$ }"
            }[s]


# ('symm','DistMult','cube'): "$ 0 $",
# ('symm', 'DistMult','sphere'): "$ 0 $",
# ('symm', 'ComplEx', 'cube'): "$ 2\sum_i \vert r_i^{\text{I}}\vert $",
# ('symm', 'ComplEx', 'sphere'): "$ \max_i \left\{\vert r_i^{\text{I}}\vert\sqrt{2} \right\} $",
# ('impl', 'DistMult', 'cube'): "$ \sum_i\max\{0,\delta_i\} $",
# ('impl', 'DistMult', 'sphere'): "$ \max_i\{\vert\delta_i\vert\} $",
# ('impl', 'ComplEx', 'cube'): "$ \sum_i\max\{0,\delta_i^{\text{R}}\} + \max\{\delta_i^{\text{R}},\vert\delta_i^{\text{I}}\vert\}$",
# ('impl', 'ComplEx', 'sphere'): "$ \max_i\left\{ \sqrt{{\delta_i^{\text{R}}}^2+{\delta_i^{\text{I}}}^2} \right\}$",
# ('impl_inv', 'DistMult', 'cube'): "$  $",
# ('impl_inv', 'DistMult', 'sphere'): "$  $",
# ('impl_inv', 'ComplEx', 'cube'): "$  $",
# ('impl_inv', 'ComplEx', 'sphere'): "$  $"


def id2clause(id):
    if 'tag=impl_inv' in id:
        return 'impl_inv'  #first!!
    elif 'tag=impl' in id:
        return 'impl'
    for clause in ['symm', 'trans_single', 'trans_diff']:
        if 'tag=%s'%clause in id:
            return clause
    return None

def id2model(id):
    for model in models_lst:
        if 'model=%s'%model in id:
            return model
    return None

def id2adv_init_ground(id):
    if 'adv_init_ground=True' in id:
        return True
    elif 'adv_init_ground=False' in id:
        return False
    else:
        return None

def id2conf(id):
    for conf in confs_lst:
        if '_c%s'%conf in id:
            return conf
    return None

def id2version(id):
    for version in versions_lst:
        if '_%s_use'%version in id:
            return version
    return None

def id2disc_epochs(id):
    for disc_epoch in disc_epochs_lst:
        if 'disc_epochs=%s_'%disc_epoch in id:
            return disc_epoch
    return None

def id2use_clauses(id):
    return 'use_clauses=True' in id

def id2entity_space(id):
    return 'unit_sphere' if 'unit-sphere' in id else 'unit_cube'


def id2clause_weight(id):
    for clause_weight in clause_weight_lst:
        if 'clause_weight=%s_'%clause_weight in id:
            return clause_weight

from time import sleep


ID2AUC = {}
found = False
with open(results) as rID:
    for line in rID:
        auc, id = line.strip().split('\t')
        clause = id2clause(id)
        model = id2model(id)
        conf = id2conf(id)
        disc_epochs = id2disc_epochs(id)
        use_clauses = id2use_clauses(id)
        clause_weight = id2clause_weight(id)
        entity_space = id2entity_space(id)
        version = id2version(id)

        if not None in (clause, model, conf, disc_epochs, use_clauses, clause_weight, entity_space, version):
            ID2AUC[(clause, model, conf, disc_epochs, use_clauses, clause_weight, entity_space, version)] = float(auc)
        else:
            print((clause, model, conf, disc_epochs, use_clauses, clause_weight, entity_space, version))


ID2AUC_versions = {}
for (clause, model, conf, disc_epochs, use_clauses, clause_weight, entity_space, version), auc in ID2AUC.items():
    if not (clause, model, conf, disc_epochs, use_clauses, clause_weight, entity_space) in ID2AUC_versions:
        ID2AUC_versions[(clause, model, conf, disc_epochs, use_clauses, clause_weight, entity_space)] = []
    ID2AUC_versions[(clause, model, conf, disc_epochs, use_clauses, clause_weight, entity_space)].append(auc)


ID2MEAN = defaultdict(lambda: -1)
for k in ID2AUC_versions:
    ID2MEAN[k] = np.mean(ID2AUC_versions[k])

#construct table:
header = lambda title: r"""
\begin{table}[t!]
\centering
\caption{ """ \
+ title \
+ r""" }
\vspace{1em}
\resizebox{.7\columnwidth}{!}{%
\begin{tabular}{llcc}
    \toprule
\multirow{ 2}{*}{Clauses} & \multirow{ 2}{*}{Model} & $\alpha=1$ & $\alpha=1$  \\
&& cube & sphere \\
\midrule
"""
footer = r"""
\bottomrule
\end{tabular}
}
\end{table}
"""


caption = r"PR-AUC results on synthetic datasets for adversarial training with closed form expressions."
for conf in confs_lst:
    for clause_weight in clause_weight_lst:

        def results_line(clause, model):
            res = string(model) + " & "
            res_STD_sphere = ID2MEAN[(clause, model, conf, '10', False, '1.0', 'unit_sphere')]
            res_STD_cube = ID2MEAN[(clause, model, conf, '10', False, '1.0', 'unit_cube')]
            res_CF_sphere = ID2MEAN[(clause, model, conf, '10', True, clause_weight, 'unit_sphere')]
            res_CF_cube = ID2MEAN[(clause, model, conf, '10', True, clause_weight, 'unit_cube')]
            #resu = [res_STD_sphere, res_STD_cube, res_CF_sphere, res_CF_cube]
            resu = [res_CF_cube, res_CF_sphere]
            resu = [np.round(1000*res)/10. for res in resu]
            maxvalue = max(resu)
            resu_str = ["\\textbf{%.1f}"%res if res == maxvalue else "%.1f"%res for res in resu]

            res += " & ".join(resu_str)

            return res + r" \\"

        print(header(caption))

        for clause in clauses_lst:
            for model in models_lst:
                show_clause = string(clause) if model == models_lst[0] else ""
                line = show_clause + " & " + results_line(clause, model)
                print(line)
            if not clause == clauses_lst[-1]:
                print(r"\midrule")

        print(footer)









