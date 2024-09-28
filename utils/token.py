from constants import dataset


def sanitize_underscore(string):
    return None if string == '_' else string


def validate_upos(upos, upos_voc):
    assert upos_voc[upos] != upos_voc.unk_idx


def sanitize_deprel(deprel, deprel_voc):
    sanitized_deprel = deprel

    if sanitized_deprel not in deprel_voc:
        sanitized_deprel = sanitized_deprel.split(':')[0]

        assert deprel_voc[sanitized_deprel] != deprel_voc.unk_idx

    return sanitized_deprel


def sanitize_semantic_role(semantic_role, semantic_role_voc):
    sanitized_semantic_role = sanitize_underscore(semantic_role)

    if sanitized_semantic_role is None:
        return sanitized_semantic_role

    sanitized_semantic_role = dataset.semantic_role_mapper[sanitized_semantic_role] \
        if dataset.semantic_role_mapper.get(sanitized_semantic_role) \
        else sanitized_semantic_role

    assert semantic_role_voc[sanitized_semantic_role] != semantic_role_voc.unk_idx

    return sanitized_semantic_role
