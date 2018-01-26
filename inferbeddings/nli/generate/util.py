# -*- coding: utf-8 -*-


def insert_subtree(tree1, st_idx, tree2):
    len_st = len(list(tree1.subtrees())[st_idx])
    res = []
    for i in range(len_st + 1):
        tree1_cp, tree2_cp = tree1.copy(deep=True), tree2.copy(deep=True)
        st = list(tree1_cp.subtrees())[st_idx]
        st.insert(i, tree2_cp)
        res += [tree1_cp]
    return res


def combine_trees(tree1, tree2):
    nb_sts1 = len(list(tree1.subtrees()))
    nb_sts2 = len(list(tree2.subtrees()))

    res = []
    for i in range(nb_sts1):
        res += insert_subtree(tree1, i, tree2)
    for i in range(nb_sts2):
        res += insert_subtree(tree2, i, tree1)

    _res = set(tuple(t.leaves()) for t in res)
    return [list(r) for r in sorted(_res)]
