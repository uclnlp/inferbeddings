# -*- coding: utf-8 -*-

from nltk.parse.corenlp import CoreNLPParser


class Parser:
    def __init__(self, url='http://127.0.0.1:9000'):
        self.parser = Parser._create_parser(url=url)
        if self.parser is None:
            self.parser = Parser._create_parser(url='http://127.0.0.1:9000')
        if self.parser is None:
            self.parser = Parser._create_parser(url='http://hamburg.vpn:9000')
        if self.parser is None:
            self.parser = Parser._create_parser(url='http://data.neuralnoise.com:9000')

        if self.parser is None:
            raise ValueError('No parsers available')

    @staticmethod
    def _create_parser(url):
        try:
            parser = CoreNLPParser(url=url)
            parser.raw_parse('This is a test sentence.')
        except Exception:
            parser = None
        return parser

    def parse(self, sentence):
        sentence_str = None
        if isinstance(sentence, str):
            sentence_str = sentence
        elif isinstance(sentence, list):
            sentence_str = ' '.join(sentence)

        if sentence_str is None:
            raise ValueError("sentence needs to be string or list, got {} instead".format(type(sentence)))

        tree, = self.parser.raw_parse(sentence=sentence)
        return tree
