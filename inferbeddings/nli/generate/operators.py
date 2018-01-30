# -*- coding: utf-8 -*-

import nltk


def _insert_subtree(tree1, st_idx, tree2):
    len_st = len(list(tree1.subtrees())[st_idx])
    res = []
    for i in range(len_st + 1):
        tree1_cp, tree2_cp = tree1.copy(deep=True), tree2.copy(deep=True)
        st = list(tree1_cp.subtrees())[st_idx]
        st.insert(i, tree2_cp)
        res += [tree1_cp]
    return res


def combine_trees(tree1, tree2):
    nb_sts = len(list(tree1.subtrees()))
    res = []
    for i in range(nb_sts):
        res += _insert_subtree(tree1, i, tree2)
    return res


def remove_subtree(tree):
    res = []
    for st in tree:
        if isinstance(st, nltk.Tree):
            tree_cp = tree.copy(deep=True)
            tree_cp.remove(st)
            res += [tree_cp]
            res += remove_subtree(st)
    return res
