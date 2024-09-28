import re
from classes.sentence import Sentence
from utils import token


def is_comment(line):
    return line[0] == '#'


def is_skipped_sentence(line):
    is_propbank_label = re.match('^# propbank = (.*)', line)

    if is_propbank_label:
        value = is_propbank_label.groups()[0]

        return value == 'no-up' or value == 'diff-number-tokens'

    is_duplicate_label = re.match('^# duplicate: (.*)', line)

    return is_duplicate_label


def extract_generic_sent_metadata(line, set_name):
    sent_metadata = []

    is_text = re.match(
        '^# text = (.*)',
        line
    ) or re.match(
        '^# sentence-text: (.*)',
        line
    ) or re.match(
        '^# text: (.*)',
        line
    ) or re.match(  # upb1.0, pt_bosque
        '.* text="(.*)"$',
        line
    )

    if is_text:
        sent_metadata.append({
            'key': 'text',
            'value': is_text.groups()[0]
        })

    is_sentence_id = re.match(
        '^# sent_id = (.*)',
        line
    ) or re.match(
        '^# sent_id (.*)',
        line
    ) or re.match(  # upb1.0, es_ancora
        '^# orig_file_sentence (.*)',
        line
    ) or re.match(  # upb1.0, pt_bosque
        '.* ref="([^"]*)" .*',
        line
    )

    if is_sentence_id:
        sent_metadata.append({
            'key': 'sent_id',
            'value': set_name + is_sentence_id.groups()[0]
        })

    return sent_metadata


def load_sentences(
        file_path,
        set_name,
        model,
        loader_version,
        upb_version,
        logger
):
    file = open(file_path, 'r')

    line = file.readline()

    sentences = []

    while line:
        sent_block = []
        while line and line != '\n':
            stripped_line = line.strip()
            sent_block.append(stripped_line)
            line = file.readline()

        sentence = Sentence(
            upb_version=upb_version
        )
        raw_tokens = []

        is_skipped = False

        for sent_line in sent_block:
            if is_comment(sent_line):
                if is_skipped_sentence(sent_line):
                    is_skipped = True
                    break

                sent_metadata = extract_generic_sent_metadata(
                    line=sent_line,
                    set_name=set_name
                )

                sentence.assign_sent_id_or_text(sent_metadata)
            else:
                raw_token = sent_line.split('\t')
                token_id = token.sanitize_underscore(raw_token[0])

                if '-' in token_id or '.' in token_id:
                    continue

                raw_tokens.append(raw_token)

        if not is_skipped:
            sentence.initialize(
                raw_tokens=raw_tokens,
                loader_version=loader_version,
                model=model,
                logger=logger
            )

            sentences.append(sentence)

        while line and line == '\n':
            line = file.readline()

    file.close()
    return sentences


def load_instances(
        sentences,
        model
):
    instances = []

    for sentence in sentences:
        sent_instances = sentence.generate_instances(model)

        instances.extend(sent_instances)

    return instances
