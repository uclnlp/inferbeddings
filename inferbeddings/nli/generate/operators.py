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
    return _remove_subtree(tree, tree)


def _remove_subtree(main_tree, subtree):
    res = []
    for st in subtree:
        if isinstance(st, nltk.Tree):
            main_tree_cp = main_tree.copy(deep=True)
            _remove_subtree_from_tree(main_tree_cp, st)
            res += [main_tree_cp]
            res += _remove_subtree(main_tree, st)
    return res


def _remove_subtree_from_tree(tree, subtree_to_remove):
    for st in tree:
        if isinstance(st, nltk.Tree):
            if st == subtree_to_remove:
                tree.remove(st)
            _remove_subtree_from_tree(st, subtree_to_remove)
    return
