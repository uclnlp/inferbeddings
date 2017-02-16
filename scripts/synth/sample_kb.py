from kb import KB, TRAIN_LABEL, DEV_LABEL, TEST_LABEL
import random


class SampleKB:
    def __init__(self, num_relations, num_entities,
                 arities=[0.0, 1.0, 0.0],
                 fb_densities=[0.0, 0.0, 0.0],
                 arg_densities=[0., 0.1, 0.0],
                 fact_prob=0.2,
                 num_symm=2,
                 num_impl=[0, 2, 0],
                 num_impl_inv=2,
                 num_impl_conj=[0, 2, 0],
                 num_trans_single=2,
                 num_trans_diff=2,
                 seed=0,
                 position_dependent_args=False,
                 position_densities=[0., 0.5, 0.0]):
        """
        :param num_relations:
        :param num_entities: number of distinct entities to generate
        :param arities:  fraction of arities
        :param arg_densities: fraction of entity combinations that are observed
        :param fact_prob:
        :param num_inv: number of 'inv' formulae   R(X0, X1) :- R(X1, X0)
        :param num_impl:
        :param num_impl_conj:
        :param num_trans:
        :param negated_head_prob:
        :param seed:
        :return:
        """
        random.seed(seed)
        self.kb = KB(seed=seed)

        num_relations_per_arity = [int(x * num_relations) for x in arities]

        entities = list(map(lambda x: "e" + str(x), range(1, num_entities+1)))

        entities_arg1 = []
        entities_arg2 = []
        entities_arg3 = []

        if position_dependent_args:
            arg1_boundary = int(len(entities)*position_densities[0])
            arg2_boundary = arg1_boundary + int(len(entities)*position_densities[1])
            entities_arg1 = entities[0:arg1_boundary]
            entities_arg2 = entities[arg1_boundary:arg2_boundary]
            entities_arg3 = entities[arg2_boundary:]
        else:
            entities_arg1 = entities
            entities_arg2 = entities
            entities_arg3 = entities

        pairs = [(x, y) for x in entities_arg1
                 for y in entities_arg2 if not x == y]

        triples = [(x, y, z) for x in entities_arg1
                    for y in entities_arg2 for z in entities_arg3
                    if not x == y and not y == z and not z == x]

        num_pair_samples = min(len(pairs), int(len(entities_arg1) *
                                               len(entities_arg2) *
                                               arg_densities[1]))
        num_triple_samples = min(len(triples), int(len(entities_arg1) *
                                                   len(entities_arg2) *
                                                   len(entities_arg3) *
                                                   arg_densities[2]))
        entities_per_arity = {
            1: entities_arg1,
            2: random.sample(pairs, num_pair_samples),
            3: random.sample(triples, num_triple_samples)
        }

        relations_per_arity = {}
        for arity in range(1, len(num_relations_per_arity) + 1):
            for i in range(1, num_relations_per_arity[arity - 1] + 1):
                fb_prefix = ""
                if fb_densities[arity-1] > random.uniform(0, 1.0):
                    fb_prefix = "REL$"
                if arity == 1:
                    rel = fb_prefix+"u"
                elif arity == 2:
                    rel = fb_prefix+"b"
                else:
                    rel = fb_prefix+"t"
                rel += str(i)

                if not arity in relations_per_arity:
                    relations_per_arity[arity] = list()
                relations_per_arity[arity].append(rel)

                for args in random.sample(entities_per_arity[arity],
                                          int(len(entities_per_arity[arity]) * fact_prob)):
                    self.kb.add_train(rel, args)

        inverse = []
        # sample symmetric relations r(X,Y) => r(Y,X)
        if 2 in relations_per_arity:
            symm = random.sample([(x, x) for x in relations_per_arity[2]], num_symm)
            inverse += symm

        # sampling implication, reversed: r1(X,Y) => r2(Y,X)
        if 2 in relations_per_arity:
            inverse += random.sample([(x, y) for x in relations_per_arity[2]
                                     for y in relations_per_arity[2]
                                     if not x == y], num_impl_inv)
        if len(inverse) > 0:
            self.kb.add_formulae("inv", {2: inverse})

        # sampling implications:
        # r1(X) => r2(X)
        # r1(X,Y) => r2(X,Y)
        implications_per_arity = {}
        for arity in range(1, len(num_relations_per_arity) + 1):
            if arity in relations_per_arity:
                implications_per_arity[arity] = \
                    random.sample([(x, y) for x in relations_per_arity[arity] for y in relations_per_arity[arity]
                                   if not x == y], num_impl[arity - 1])
        self.kb.add_formulae("impl", implications_per_arity)

        # sampling implications with conjunction in body:
        # r1(X,Y) ^ r2(X,Y) => r3(X,Y)
        # r1(X) ^ r2(X) => r3(X)
        implications_with_conjunction_per_arity = {}
        for arity in range(1, len(num_relations_per_arity) + 1):
            if arity in relations_per_arity and len(relations_per_arity[arity]) >= 3:
                implications_with_conjunction_per_arity[arity] = \
                    random.sample([(x, y, z) for x in relations_per_arity[arity]
                                   for y in relations_per_arity[arity]
                                   for z in relations_per_arity[arity]
                                   if not x == y and not y == z and not z == x],
                                  num_impl_conj[arity - 1])
        self.kb.add_formulae("impl_conj", implications_with_conjunction_per_arity)

        # sampling transitivities:
        transitivities = []
        # (1) simple transitivities  r(X,Y) ^ r(Y,Z) => r(X,Z)
        # (2) general transitivities  r1(X,Y) ^ r2(Y,Z) => r3(X,Z)  (r1, r2, r3 differ)

        if 2 in relations_per_arity:
            if num_trans_single > 0:
                transitivities += random.sample([(x, x, x)
                                                for x in relations_per_arity[2]], num_trans_single)
            if num_trans_diff > 0:
                transitivities += random.sample([(x, y, z)
                                                for x in relations_per_arity[2]
                                                for y in relations_per_arity[2]
                                                for z in relations_per_arity[2]
                                                if not x == y and
                                                not y == z and
                                                not z == x], num_trans_diff)
        if len(transitivities) > 0:
            self.kb.add_formulae("trans", {2: transitivities})

        # todo: sampling negation (also applies to all heads of formulae above):
        # r1 => !r2

    def get_kb(self):
        return self.kb


if __name__=="__main__":

    import sys
    import argparse
    import os

    #fixed args
    sampled_unobserved_per_true = 1 # number of false (unobserved) test facts added for each true test fact (inferred from clause)
    simple_transitivities = False
    seed = 846

    #input args
    argparser = argparse.ArgumentParser('create artificial dataset (train+test) with rules (all arity 2)')

    argparser.add_argument('--entities', '-E', required=True, type=int, help='number of entities')
    argparser.add_argument('--predicates', '-P', required=True, type=int, help='number of predicates')
    argparser.add_argument('--test-prob', type=float, default=0.5,
                           help='fraction of inferred facts (from formulae) to be added to test set')
    argparser.add_argument('--arg-density', type=float, default=0.1,
                           help='fraction of all possible pairs of entities observed')
    argparser.add_argument('--fact-prob', type=float, default=0.1,
                           help='for all observed pairs: fraction of those that occur with each relation')
    argparser.add_argument('--symm', type=int, default=0,
                           help='number of clauses  p(X0, X1) :- p(X1, X0)')
    argparser.add_argument('--impl', type=int, default=0,
                           help='number of clauses p(X0, X1) :- q(X0, X1)  (with p and q different)')
    argparser.add_argument('--impl-inv', type=int, default=0,
                           help='number of clauses  p(X0, X1) :- q(X1, X0)')
    argparser.add_argument('--impl-conj', type=int, default=0,
                           help='number of clauses r(X0, X1) :- p(X0, X1), q(X0, X1)')
    argparser.add_argument('--trans-single', type=int, default=0,
                           help='number of clauses r(X0, X2) :- r(X0, X1), r(X1, X2)')
    argparser.add_argument('--trans-diff', type=int, default=0,
                           help='number of clauses r(X0, X2) :- p(X0, X1), q(X1, X2)  (with p,q,r different)')
    argparser.add_argument('--dir', type=str, default='../../data/synth/sampled',
                           help='target directory')
    argparser.add_argument('--tag', type=str, default='synth',
                           help='experiment tag')

    args = argparser.parse_args(sys.argv[1:])
    cmd = ' '.join(arg for arg in sys.argv[1:])

    Ne = args.entities
    Nr = args.predicates
    test_prob = args.test_prob
    arg_density = args.arg_density
    fact_prob = args.fact_prob
    num_symm = args.symm
    num_impl = args.impl
    num_impl_inv = args.impl_inv
    num_impl_conj = args.impl_conj
    num_trans_single = args.trans_single
    num_trans_diff = args.trans_diff

    testKB = SampleKB(Nr, Ne,
                      arg_densities=[0, arg_density, 0],
                      fact_prob=fact_prob,
                      num_symm=num_symm,
                      num_impl_inv=num_impl_inv,
                      num_impl=[0, num_impl, 0],
                      num_impl_conj=[0, num_impl_conj, 0],
                      num_trans_single=num_trans_single,
                      num_trans_diff=num_trans_diff,
                      seed=seed
                      ).get_kb()

    N_original_facts = len(testKB.get_all_facts(of_types=TRAIN_LABEL))
#    for fact in testKB.get_all_facts(of_types=TRAIN_LABEL):
#        print(fact)
#    for clause in testKB.get_formulae_strings():
#        print(clause)
    for clause in testKB.get_formulae_for_ntp_strings():
        print(clause)

    testKB.apply_formulae(test_prob=test_prob, sampled_unobserved_per_true=sampled_unobserved_per_true)

    msg = ''
    msg += '#%d original purely random train facts (without formulae)\n'%N_original_facts
    train_facts = testKB.get_all_facts(of_types=(TRAIN_LABEL,))
    msg +='#%d train facts (after creating rules and adding inferred facts to train set with prob %.3f)\n'%(len(train_facts), 1.-test_prob)
    test_facts = testKB.get_all_facts(of_types=(TEST_LABEL,))
    test_facts_T = [f for f in test_facts if f[1]]
    test_facts_F = [f for f in test_facts if not f[1]]
    msg += '#%d test facts (%d True, %d False)\n'%(len(test_facts), len(test_facts_T), len(test_facts_F))
    print(msg)

    #create train / test file for inferbeddings
    train_file = os.path.join(args.dir, args.tag + '_train.tsv')
    test_file = os.path.join(args.dir, args.tag + '_test.tsv')
    clause_file = os.path.join(args.dir, args.tag + '_clauses.pl')
    readme_file = os.path.join(args.dir, args.tag + '_config.txt')

    with open(readme_file, 'w') as rf:
        rf.write('\n#command:\npython3 %s\n'%' '.join(list(sys.argv)))
        rf.write('\n#config:\n')
        for k in ['tag', 'entities', 'predicates', 'test_prob', 'arg_density', 'fact_prob',
                  'symm', 'impl', 'impl_inv', 'impl_conj', 'trans_single', 'trans_diff',
                  'dir']:
            rf.write('{}\t{}\n'.format(k, vars(args)[k]))
        rf.write('seed\t{}\n'.format(seed))
        rf.write('sampled_unobserved_per_true\t{}\n'.format(sampled_unobserved_per_true))
        rf.write('simple_transitivities\t{}\n'.format(simple_transitivities))
        rf.write('\n#stats:\n')
        rf.write(msg)


    with open(train_file, 'w') as trf:
        for fact in sorted(testKB.get_all_facts(of_types=TRAIN_LABEL)):
            pred, (subj, obj) = fact[0]
            trf.write('{}\t{}\t{}\n'.format(subj, pred, obj))

    with open(test_file, 'w') as tef:
        for fact in sorted(testKB.get_all_facts(of_types=TEST_LABEL)):
            pred, (subj, obj) = fact[0]
            truth = fact[1]
            tef.write('{}\t{}\t{}\t{}\n'.format(subj, pred, obj, {True: 1, False: 0}[truth]))

    with open(clause_file, 'w') as clf:
        for clause in testKB.get_formulae_for_ntp_strings():
            clf.write(clause+'\n')
