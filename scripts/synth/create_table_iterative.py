import numpy as np

from collections import defaultdict

results = '/Users/tdmeeste/workspace/inferbeddings/logs/synth/synth_paper_iterative_aggregated.txt'

models_lst = ['DistMult', 'ComplEx']
clauses_lst = ['symm', 'impl', 'impl_inv', 'trans_single', 'trans_diff']
confs_lst = ['0.0']
versions_lst = ['v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9']
adv_weights_lst = ['0', '1']
adv_epochs_lst = ['0', '10']
disc_epochs_lst = ['10']


def string(s):
    return {'TransE' : r"\emph{ASR}-\mdl{TransE}",
            'DistMult' : r"\mdl{DistM.}",
            'ComplEx' : r"\mdl{Compl.}",
            'symm' : r"\multirow{ 2}{*}{ $\begin{array} {l@{}} r(X_1, X_2) \\  \quad\Rightarrow r(X_2, X_1) \end{array}$ }",
            'impl' : r"\multirow{ 2}{*}{ $\begin{array} {l@{}} r(X_1, X_2) \\  \quad\Rightarrow s(X_1, X_2) \end{array}$ }",
            'impl_inv' : r"\multirow{ 2}{*}{ $\begin{array} {l@{}} r(X_1, X_2) \\  \quad\Rightarrow s(X_2, X_1) \end{array}$ }",
            'trans_single': r"\multirow{ 2}{*}{$\begin{array} {l@{}} r(X_1, X_2) \wedge r(X_2, X_3) \\  \quad\Rightarrow r(X_1, X_3) \end{array}$}",
            'trans_diff': r"\multirow{ 2}{*}{$\begin{array} {l@{}} r(X_1, X_2) \wedge s(X_2, X_3) \\  \quad\Rightarrow t(X_1, X_3) \end{array}$}"
            }[s]

#'symm': r"$r(\x_2, \x_1) :- r(\x_1, \x_2)$",
#'impl': r"$s(\x_1, \x_2) :- r(\x_1, \x_2)$",
#'impl_inv': r"$s(\x_2, \x_1) :- r(\x_1, \x_2)$",
#'trans_single': r"$r(\x_1, \x_3) :- r(\x_1, \x_2), r(\x_2, \x_3)$",
#'trans_diff': r"$t(\x_1, \x_3) :- r(\x_1, \x_2), s(\x_2, \x_3)$"


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
        if '_%s.log'%version in id:
            return version
    return None

def id2adv_weight(id):
    for adv_weight in adv_weights_lst:
        if 'adv_weight=%s_'%adv_weight in id:
            return adv_weight
    return None

def id2adv_epochs(id):
    for adv_epoch in adv_epochs_lst:
        if 'adv_epochs=%s_'%adv_epoch in id:
            return adv_epoch
    return None

def id2disc_epochs(id):
    for disc_epoch in disc_epochs_lst:
        if 'disc_epochs=%s_'%disc_epoch in id:
            return disc_epoch
    return None

def id2entity_space(id):
    return 'unit_sphere' if 'unit-sphere' in id else 'unit_cube'


from time import sleep


ID2AUC = {}
found = False
with open(results) as rID:
    for line in rID:
        auc, id = line.strip().split('\t')
        clause = id2clause(id)
        model = id2model(id)
        adv_init_ground = id2adv_init_ground(id)
        conf = id2conf(id)
        adv_weight = id2adv_weight(id)
        adv_epochs = id2adv_epochs(id)
        disc_epochs = id2disc_epochs(id)
        entity_space = id2entity_space(id)
        version = id2version(id)

        if not None in (clause, model, adv_init_ground, conf, adv_weight, adv_epochs, disc_epochs, entity_space, version):
            ID2AUC[(clause, model, adv_init_ground, conf, adv_weight, adv_epochs, disc_epochs, entity_space, version)] = float(auc)


ID2AUC_versions = {}
for (clause, model, adv_init_ground, conf, adv_weight, adv_epochs, disc_epochs, entity_space, version), auc in ID2AUC.items():
    if not (clause, model, adv_init_ground, conf, adv_weight, adv_epochs, disc_epochs, entity_space) in ID2AUC_versions:
        ID2AUC_versions[(clause, model, adv_init_ground, conf, adv_weight, adv_epochs, disc_epochs, entity_space)] = []
    ID2AUC_versions[(clause, model, adv_init_ground, conf, adv_weight, adv_epochs, disc_epochs, entity_space)].append(auc)

ID2MEAN = defaultdict(lambda: -1)
for k in ID2AUC_versions:
    ID2MEAN[k] = np.mean(ID2AUC_versions[k])

#construct table:
title = r"PR-AUC results for \emph{ASR}-DistMult (DistM.) and \emph{ASR}-ComplEx (Compl.) on synthetic datasets with various types of clauses (with $r\not=s\not=t$). Comparison of standard models without clauses ($\alpha=0$), and iterative adversarial training with clauses ($\alpha=1$). "
header = r"""
\begin{table}[t!]
\centering
\caption{
""" + title + \
r"""
}
\label{synth}
\vspace{1em}
\resizebox{\columnwidth}{!}{
\begin{tabular}{llcccc}
    \toprule
\multirow{ 2}{*}{Clauses} & \multirow{ 2}{*}{Model} & $\alpha=0$ & $\alpha=0$ & $\alpha=1$ & $\alpha=1$  \\
&& cube & sphere & cube & sphere \\
\midrule
"""
footer = r"""
\bottomrule
\end{tabular}
}
\end{table}
"""

def results_line(clause, model):
    res = string(model) + " & "
    conf = "0.0"
    res_STD_cube = ID2MEAN[(clause, model, True, conf, '0', '10', '10','unit_cube')]
    res_STD_sphere = ID2MEAN[(clause, model, True, conf, '0', '10', '10','unit_sphere')]
    #res_SMPL = ID2MEAN[(clause, model, True, conf, '1', '0', '10')]
    #res_ASR_R = ID2MEAN[(clause, model, False, conf, '1', '1')]
    res_ASR_cube = ID2MEAN[(clause, model, True, conf, '1', '10', '10', 'unit_cube')]
    res_ASR_sphere = ID2MEAN[(clause, model, True, conf, '1', '10', '10', 'unit_sphere')]
    resu = [res_STD_cube, res_STD_sphere, res_ASR_cube, res_ASR_sphere]
    resu = [np.round(1000*res)/10. for res in resu]
    maxvalue = max(resu)
    resu_str = ["\\textbf{%.1f}"%res if res == maxvalue else "%.1f"%res for res in resu]

    res += " & ".join(resu_str)

    return res + r" \\"


print(header)




for clause in clauses_lst:
    for model in models_lst:
        show_clause = string(clause) if model == models_lst[0] else ""
        line = show_clause + " & " + results_line(clause, model)
        print(line)
    if not clause == clauses_lst[-1]:
        print(r"\midrule")




print(footer)









