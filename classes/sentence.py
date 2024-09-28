from utils import token
from constants import dataset
from classes.tree import Tree
from classes.instance import Instance
from utils import tree


class Sentence:
    def __init__(
            self,
            upb_version
    ):
        self.upb_version = upb_version
        self.sent_id = None
        self.text = None
        self.words = None
        self.uposses = None
        self.deprels = None
        self.heads = None
        self.passes = None
        self.token_ids = None

        # Dependency tree
        self.root = None
        self.nodes = None
        self.depths = None

        # Indices
        self.upos_ids = None
        self.deprel_ids = None
        self.we_words = None
        self.we_offsets = None
        self.we_input_ids = None

        # Matrices
        self.dist_matrix = None
        self.deprel_matrix = None
        self.deparc_matrix = None
        self.deprel_ext_matrix = None
        self.deprel_path_matrix = None
        self.deparc_path_matrix = None
        self.deprel_ext_path_matrix = None
        self.path_len_matrix = None

    def assign_sent_id_or_text(self, sent_metadata):
        for metadata in sent_metadata:
            setattr(self, metadata['key'], metadata['value'])

    def validate_passes(self):
        for pas in self.passes:
            assert ('V' in pas)

    def construct_tree(self):
        self.nodes = [Tree(idx) for idx in range(len(self.heads))]

        for i in range(len(self.heads)):
            head_idx = self.heads[i]
            if head_idx == -1:
                self.root = self.nodes[i]
            else:
                self.nodes[head_idx].add_child(self.nodes[i], self.deprels[i])

        assert self.root

        self.depths = [node.depth() for node in self.nodes]

    def construct_matrices(self, model):
        adj_matrix = tree.construct_adjacency_matrix_from_tree(
            length=len(self.words),
            root=self.root,
            is_directed=False,
            is_self_loop=True
        )

        self.dist_matrix, predec_matrix = tree.construct_distance_matrix_from_adjacency_matrix(
            adj_matrix=adj_matrix,
            is_directed=False,
            is_self_loop=True
        )

        self.deprel_matrix, self.deparc_matrix, self.deprel_ext_matrix = tree.construct_dep_matrix_from_tree(
            length=len(self.words),
            root=self.root,
            model=model,
            upb_version=self.upb_version
        )

        self.deprel_path_matrix, self.deparc_path_matrix, self.deprel_ext_path_matrix, self.path_len_matrix = \
            tree.construct_dep_path_matrix(
                predec_matrix=predec_matrix,
                length=len(self.words),
                deprel_matrix=self.deprel_matrix,
                deparc_matrix=self.deparc_matrix,
                deprel_ext_matrix=self.deprel_ext_matrix
            )

    def initialize_passes(
            self,
            raw_tokens,
            loader_version,
            max_len,
            model,
            logger
    ):
        mandatory_attributes = dataset.mandatory_attributes_by_loader_version[loader_version]
        mandatory_attributes_len = len(mandatory_attributes)

        if loader_version == 3:
            self.passes = []

            for idx, raw_token in enumerate(raw_tokens):
                sanitized_predicate_sense = token.sanitize_underscore(
                    raw_token[mandatory_attributes['predicate_sense']]
                )

                if sanitized_predicate_sense:
                    self.passes.append({
                        'V': [idx]
                    })

                sanitized_argument_head = token.sanitize_underscore(raw_token[mandatory_attributes['argument_head']])

                if not sanitized_argument_head:
                    continue

                assert sanitized_predicate_sense

                pas = self.passes[-1]

                for pas_pair in sanitized_argument_head.split('|'):
                    [semantic_role, token_id] = pas_pair.split(':')

                    if '.' in token_id:
                        logger.info(f'Annotation on enhanced token. '
                                    f'Token ID: {token_id}. '
                                    f'Semantic role: {semantic_role}.')
                        continue

                    sanitized_semantic_role = token.sanitize_semantic_role(
                        semantic_role.upper(),
                        model.semantic_role_voc
                    )

                    assert sanitized_semantic_role
                    actual_idx = int(token_id) - 1
                    assert token_id == self.token_ids[actual_idx]

                    if pas.get(sanitized_semantic_role):
                        pas[sanitized_semantic_role].append(actual_idx)
                    else:
                        pas[sanitized_semantic_role] = [actual_idx]
        else:
            pas_count = max_len - mandatory_attributes_len
            self.passes = [dict() for _ in range(pas_count)]  # arg to idx

            pas_idx_v1 = 0

            for idx, raw_token in enumerate(raw_tokens):
                if loader_version == 1:
                    if raw_token[mandatory_attributes['is_predicate']].upper() == 'Y':
                        self.passes[pas_idx_v1]['V'] = [idx]
                        pas_idx_v1 += 1

                for i in range(mandatory_attributes_len, len(raw_token)):
                    pas_idx = i - mandatory_attributes_len

                    if loader_version == 2 and raw_token[i].upper() == 'V':
                        if self.passes[pas_idx].get('V'):
                            self.passes[pas_idx]['V'].append(idx)
                        else:
                            self.passes[pas_idx]['V'] = [idx]

                        continue

                    sanitized_semantic_role = token.sanitize_semantic_role(
                        raw_token[i].upper(),
                        model.semantic_role_voc
                    )

                    if sanitized_semantic_role:
                        if self.passes[pas_idx].get(sanitized_semantic_role):
                            self.passes[pas_idx][sanitized_semantic_role].append(idx)
                        else:
                            self.passes[pas_idx][sanitized_semantic_role] = [idx]

        self.validate_passes()

    def generate_instances(self, model):
        instances = [Instance(sentence=self, pas=pas) for pas in self.passes]
        for instance in instances:
            instance.initialize(model)

        return instances

    def construct_indices(self, model):
        pretrained_word_emb = model.get_network().gtn.embedding.pretrained_word_emb

        self.upos_ids = [model.pos_voc[upos] for upos in self.uposses]
        self.deprel_ids = [
            model.deprel_voc_by_version[self.upb_version][deprel] for deprel in self.deprels
        ]
        self.we_words, self.we_offsets = pretrained_word_emb.tokenize_sentence(
            self.words
        )
        self.we_input_ids = pretrained_word_emb.get_input_ids(
            self.we_words
        )

    def initialize(
            self,
            raw_tokens,
            loader_version,
            model,
            logger
    ):
        self.words = []
        self.uposses = []
        self.deprels = []
        self.heads = []
        self.token_ids = []

        mandatory_attributes = dataset.mandatory_attributes_by_loader_version[loader_version]
        max_len = 0

        for raw_token in raw_tokens:
            if len(raw_token) > max_len:
                max_len = len(raw_token)

            self.token_ids.append(raw_token[mandatory_attributes['token_id']])

            upos = raw_token[mandatory_attributes['upos']]
            token.validate_upos(upos, model.pos_voc)
            self.uposses.append(upos)

            if upos == 'PUNCT':
                self.words.append(raw_token[mandatory_attributes['lemma']])
            else:
                self.words.append(raw_token[mandatory_attributes['form']])

            deprel = raw_token[mandatory_attributes['deprel']]
            sanitized_deprel = token.sanitize_deprel(deprel, model.deprel_voc_by_version[self.upb_version])
            self.deprels.append(sanitized_deprel)

        # To validate head we need to retrieve all tokens first
        for raw_token in raw_tokens:
            head_token_id = raw_token[mandatory_attributes['head']]

            if head_token_id == '0':  # root
                self.heads.append(-1)
            else:
                actual_idx = int(head_token_id) - 1
                assert head_token_id == self.token_ids[actual_idx]
                self.heads.append(actual_idx)

        self.initialize_passes(
            raw_tokens=raw_tokens,
            loader_version=loader_version,
            max_len=max_len,
            model=model,
            logger=logger
        )
        self.construct_tree()
        self.construct_matrices(model=model)
        self.construct_indices(model)
