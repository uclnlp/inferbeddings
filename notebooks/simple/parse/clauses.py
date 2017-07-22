# -*- coding: utf-8 -*-

from parsimonious.grammar import Grammar, NodeVisitor

grammar = Grammar("""
    clause      = atom ( ":-" _ atom_list)? ("<" _ weight _ ">")?
    atom_list   = atom  ("," _ atom_list)? _
    atom        = neg? predicate "(" _ term_list ")" _
    neg         = "!" _
    term_list   = term  ("," _ term_list)? _
    term        = constant / variable
    predicate   = low_id / string
    constant    = low_id / string
    variable    = ~"[A-Z][a-z A-Z 0-9_]*"
    low_id      = ~"[a-z_./][a-z A-Z 0-9_./]*"
    string      = ~r"'[^']*'" / ~r"\\"[^\\"]*\\""
    _           = skip*
    skip        = ~r"\s+"
    weight      = float / "?"
    float       = ~"[-]?[0-9]+(\\.[0-9]+)?"
    """)


class Expr:
    def __repr__(self):
        return self.__dict__.__repr__()

    def __eq__(self, other):
        return isinstance(other, self.__class__) and str(other) == str(self)

    def __hash__(self):
        return self.__str__().__hash__()


class Variable(Expr):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


class Constant(Expr):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


class Predicate(Expr):
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return isinstance(other, self.__class__) and other.name == self.name

    def __hash__(self):
        return self.name.__hash__()

    def __repr__(self):
        return self.name


class Atom(Expr):
    def __init__(self, predicate: Predicate, *arguments, negated=False):
        self.predicate = predicate
        self.arguments = arguments
        self.negated = negated

    def __repr__(self):
        return "{}{}({})".format("!" if self.negated else "", self.predicate.name,
                                 ", ".join(str(x) for x in self.arguments))


class Clause(Expr):
    def __init__(self, head: Atom, *body, weight=1.0):
        self.head = head
        self.body = body
        self.weight = weight

    def __repr__(self):
        return str(self.head) if len(self.body) == 0 else "{head} :- {body}".format(
            head=self.head,
            body=", ".join(str(x) for x in self.body))


class ClauseVisitor(NodeVisitor):
    def visit_clause(self, node, visited_children):
        weight = 1.0 if len(visited_children[2]) == 0 else visited_children[2][0][2]
        if len(visited_children[1]) == 0:
            return Clause(visited_children[0], weight=weight)
        else:
            head, ((_, _, body),), _ = visited_children
            return Clause(head, *body, weight=weight)

    def visit_predicate(self, _, visited_children):
        return Predicate(visited_children[0])

    def visit_constant(self, _, visited_children):
        return Constant(visited_children[0])

    def visit_variable(self, node, _):
        return Variable(node.full_text[node.start:node.end])

    def visit_term(self, _, visited_children):
        return visited_children[0]

    def visit_float(self, node, _):
        text = node.full_text[node.start:node.end]
        return float(text)

    def visit_weight(self, node, visited_children):
        return None if node.full_text[node.start:node.end] == "?" else visited_children[0]

    def visit_term_list(self, _, visited_children):
        if len(visited_children[1]) == 0:
            return visited_children[:1]
        else:
            head, ((_, _, tail),), _ = visited_children
            return [head] + tail

    def visit_atom_list(self, _, visited_children):
        if len(visited_children[1]) == 0:
            return visited_children[:1]
        else:
            head, ((_, _, tail),), _ = visited_children
            return [head] + tail

    def visit_atom(self, _, visited_children):
        return Atom(visited_children[1], *visited_children[4], negated=len(visited_children[0]) == 1)

    def visit_low_id(self, node, _):
        return node.full_text[node.start:node.end]

    def visit_up_id(self, node, _):
        return node.full_text[node.start:node.end]

    def visit_string(self, node, _):
        return node.full_text[node.start:node.end]

    def generic_visit(self, node, visited_children):
        return visited_children

    def visit_neg(self, _1, _2):
        return True
