import os
import math
import torch
import argparse
import statistics
from constants import dataset
from utils import logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Evaluation Result',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--num_runs',
        type=int,
        help='Number of runs.',
        required=True
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        help='Logging directory.',
        required=True
    )
    parser.add_argument(
        '--upb_version',
        type=int,
        help='UPB version.',
        required=True
    )

    args = parser.parse_args()

    all_evaluation = dict()

    for is_gold in [True, False]:
        acc_evaluation = dict()
        acc_evaluation['avg'] = {
            'dev': {
                0: {
                    'precision': [0] * args.num_runs,
                    'recall': [0] * args.num_runs,
                    'f1': [0] * args.num_runs
                },
                1: {
                    'precision': [0] * args.num_runs,
                    'recall': [0] * args.num_runs,
                    'f1': [0] * args.num_runs
                },
                2: {
                    'precision': [0] * args.num_runs,
                    'recall': [0] * args.num_runs,
                    'f1': [0] * args.num_runs
                },
                3: {
                    'precision': [0] * args.num_runs,
                    'recall': [0] * args.num_runs,
                    'f1': [0] * args.num_runs
                },
                4: {
                    'precision': [0] * args.num_runs,
                    'recall': [0] * args.num_runs,
                    'f1': [0] * args.num_runs
                },
                5: {
                    'precision': [0] * args.num_runs,
                    'recall': [0] * args.num_runs,
                    'f1': [0] * args.num_runs
                },
                6: {
                    'precision': [0] * args.num_runs,
                    'recall': [0] * args.num_runs,
                    'f1': [0] * args.num_runs
                }
            },
            'test': {
                0: {
                    'precision': [0] * args.num_runs,
                    'recall': [0] * args.num_runs,
                    'f1': [0] * args.num_runs
                },
                1: {
                    'precision': [0] * args.num_runs,
                    'recall': [0] * args.num_runs,
                    'f1': [0] * args.num_runs
                },
                2: {
                    'precision': [0] * args.num_runs,
                    'recall': [0] * args.num_runs,
                    'f1': [0] * args.num_runs
                },
                3: {
                    'precision': [0] * args.num_runs,
                    'recall': [0] * args.num_runs,
                    'f1': [0] * args.num_runs
                },
                4: {
                    'precision': [0] * args.num_runs,
                    'recall': [0] * args.num_runs,
                    'f1': [0] * args.num_runs
                },
                5: {
                    'precision': [0] * args.num_runs,
                    'recall': [0] * args.num_runs,
                    'f1': [0] * args.num_runs
                },
                6: {
                    'precision': [0] * args.num_runs,
                    'recall': [0] * args.num_runs,
                    'f1': [0] * args.num_runs
                }
            }
        }

        for idx_run in range(args.num_runs):
            eval_filename = os.path.join(
                args.log_dir,
                'evals',
                'gold' if is_gold else 'pred',
                f'eval_{args.upb_version}_run_{idx_run}.pkl'
            )
            evaluation = torch.load(f=eval_filename)

            for lang in dataset.metadata_by_version_to_lang_to_treebank[args.upb_version]:
                if acc_evaluation.get(lang) is None:
                    acc_evaluation[lang] = {}

                for set_name in ['dev', 'test']:
                    if acc_evaluation[lang].get(set_name) is None:
                        acc_evaluation[lang][set_name] = {}

                    for dep_dist in evaluation[lang][set_name]['scorer_out_by_dep_dist']:
                        if acc_evaluation[lang][set_name].get(dep_dist) is None:
                            acc_evaluation[lang][set_name][dep_dist] = {
                                'precision': [],
                                'recall': [],
                                'f1': []
                            }

                        acc_evaluation[lang][set_name][dep_dist]['precision'].append(evaluation[lang][set_name]['scorer_out_by_dep_dist'][dep_dist]['precision'])
                        acc_evaluation[lang][set_name][dep_dist]['recall'].append(evaluation[lang][set_name]['scorer_out_by_dep_dist'][dep_dist]['recall'])
                        acc_evaluation[lang][set_name][dep_dist]['f1'].append(evaluation[lang][set_name]['scorer_out_by_dep_dist'][dep_dist]['f1'])

                        if lang != 'en':
                            acc_evaluation['avg'][set_name][dep_dist]['precision'][idx_run] += evaluation[lang][set_name]['scorer_out_by_dep_dist'][dep_dist]['precision']
                            acc_evaluation['avg'][set_name][dep_dist]['recall'][idx_run] += evaluation[lang][set_name]['scorer_out_by_dep_dist'][dep_dist]['recall']
                            acc_evaluation['avg'][set_name][dep_dist]['f1'][idx_run] += evaluation[lang][set_name]['scorer_out_by_dep_dist'][dep_dist]['f1']

        final_evaluation = dict()
        num_langs = len(dataset.metadata_by_version_to_lang_to_treebank[args.upb_version]) - 1  # english not included
        enum_langs = list(dataset.metadata_by_version_to_lang_to_treebank[args.upb_version].keys()) + ['avg']

        for lang in enum_langs:
            final_evaluation[lang] = {
                'dev': {
                    0: {
                        'precision': {},
                        'recall': {},
                        'f1': {}
                    },
                    1: {
                        'precision': {},
                        'recall': {},
                        'f1': {}
                    },
                    2: {
                        'precision': {},
                        'recall': {},
                        'f1': {}
                    },
                    3: {
                        'precision': {},
                        'recall': {},
                        'f1': {}
                    },
                    4: {
                        'precision': {},
                        'recall': {},
                        'f1': {}
                    },
                    5: {
                        'precision': {},
                        'recall': {},
                        'f1': {}
                    },
                    6: {
                        'precision': {},
                        'recall': {},
                        'f1': {}
                    }
                },
                'test': {
                    0: {
                        'precision': {},
                        'recall': {},
                        'f1': {}
                    },
                    1: {
                        'precision': {},
                        'recall': {},
                        'f1': {}
                    },
                    2: {
                        'precision': {},
                        'recall': {},
                        'f1': {}
                    },
                    3: {
                        'precision': {},
                        'recall': {},
                        'f1': {}
                    },
                    4: {
                        'precision': {},
                        'recall': {},
                        'f1': {}
                    },
                    5: {
                        'precision': {},
                        'recall': {},
                        'f1': {}
                    },
                    6: {
                        'precision': {},
                        'recall': {},
                        'f1': {}
                    }
                }
            }

            for set_name in ['dev', 'test']:
                for dep_dist in [0, 1, 2, 3, 4, 5, 6]:
                    assert len(acc_evaluation[lang][set_name][dep_dist]['precision']) == args.num_runs
                    assert len(acc_evaluation[lang][set_name][dep_dist]['recall']) == args.num_runs
                    assert len(acc_evaluation[lang][set_name][dep_dist]['f1']) == args.num_runs

                    if lang == 'avg':
                        acc_evaluation[lang][set_name][dep_dist]['precision'] = [val / num_langs for val in acc_evaluation[lang][set_name][dep_dist]['precision']]
                        acc_evaluation[lang][set_name][dep_dist]['recall'] = [val / num_langs for val in acc_evaluation[lang][set_name][dep_dist]['recall']]
                        acc_evaluation[lang][set_name][dep_dist]['f1'] = [val / num_langs for val in acc_evaluation[lang][set_name][dep_dist]['f1']]

                    if args.num_runs > 1:
                        final_evaluation[lang][set_name][dep_dist]['precision']['stdev'] = statistics.stdev(
                            acc_evaluation[lang][set_name][dep_dist]['precision']
                        )
                        final_evaluation[lang][set_name][dep_dist]['recall']['stdev'] = statistics.stdev(
                            acc_evaluation[lang][set_name][dep_dist]['recall']
                        )
                        final_evaluation[lang][set_name][dep_dist]['f1']['stdev'] = statistics.stdev(
                            acc_evaluation[lang][set_name][dep_dist]['f1']
                        )
                        final_evaluation[lang][set_name][dep_dist]['precision']['stderr'] = final_evaluation[lang][set_name][dep_dist]['precision']['stdev'] / math.sqrt(args.num_runs)
                        final_evaluation[lang][set_name][dep_dist]['recall']['stderr'] = final_evaluation[lang][set_name][dep_dist]['recall']['stdev'] / math.sqrt(args.num_runs)
                        final_evaluation[lang][set_name][dep_dist]['f1']['stderr'] = final_evaluation[lang][set_name][dep_dist]['f1']['stdev'] / math.sqrt(args.num_runs)

                    final_evaluation[lang][set_name][dep_dist]['precision']['avg'] = sum(acc_evaluation[lang][set_name][dep_dist]['precision']) / args.num_runs
                    final_evaluation[lang][set_name][dep_dist]['recall']['avg'] = sum(acc_evaluation[lang][set_name][dep_dist]['recall']) / args.num_runs
                    final_evaluation[lang][set_name][dep_dist]['f1']['avg'] = sum(acc_evaluation[lang][set_name][dep_dist]['f1']) / args.num_runs

        all_evaluation['gold' if is_gold else 'pred'] = final_evaluation

    logger_ = logger.get_dep_eval_logger(
        log_dir=args.log_dir
    )

    enum_langs.remove('avg')
    enum_langs.insert(1, 'avg')

    for set_name in ['dev', 'test']:
        set_w = set_name.upper()
        logger_.info(set_w)
        for annotation_type in ['gold', 'pred']:
            annotation_type_w = annotation_type.upper()
            logger_.info(annotation_type_w)
            final_evaluation = all_evaluation[annotation_type]
            for lang in enum_langs:
                lang_w = lang.upper()
                logger_.info(f'{set_w} {annotation_type_w} {lang_w}')
                logger_.info('dep_dist f1 stderr stdev')
                for dep_dist in [0, 1, 2, 3, 4, 5, 6]:
                    log = f'{dep_dist} ' + '{:.2f} '.format(final_evaluation[lang][set_name][dep_dist]['f1']['avg'] * 100) + '{:.2f} '.format(final_evaluation[lang][set_name][dep_dist]['f1'].get('stderr', 0) * 100) + '{:.2f}'.format(final_evaluation[lang][set_name][dep_dist]['f1'].get('stdev', 0) * 100)
                    logger_.info(log)
            logger_.info('')

    logger.clean_logger()

    logger_ = logger.get_dep_stat_logger(
        log_dir=args.log_dir
    )

    # Dep stat
    enum_langs.remove('avg')

    for is_gold in [True, False]:
        annotation_type_w = 'GOLD' if is_gold else 'PRED'
        logger_.info(annotation_type_w)
        eval_filename = os.path.join(
            args.log_dir,
            'evals',
            'gold' if is_gold else 'pred',
            f'eval_{args.upb_version}_run_0.pkl'
        )
        evaluation = torch.load(f=eval_filename)

        for set_name in ['dev', 'test']:
            set_w = set_name.upper()
            logger_.info(set_w)
            percent_by_dep_dist = {
                0: 0,
                1: 0,
                2: 0,
                3: 0,
                4: 0,
                5: 0,
                6: 0
            }
            for lang in enum_langs:
                lang_w = lang.upper()
                logger_.info(f'{annotation_type_w} {set_w} {lang_w}')
                for dep_dist in [0, 1, 2, 3, 4, 5, 6]:
                    percent = round(evaluation[lang][set_name]['total_tokens_by_dep_dist'][dep_dist] / evaluation[lang][set_name]['total_tokens_by_dep_dist']['total'] * 100, 2)

                    if lang != 'en':
                        percent_by_dep_dist[dep_dist] += percent

                    log = f'{dep_dist} ' + ' {} '.format(evaluation[lang][set_name]['total_tokens_by_dep_dist'][dep_dist]) + ' {:.2f}'.format(percent)
                    logger_.info(log)
            logger_.info(f'{annotation_type_w} {set_w} AVG')
            for dep_dist in [0, 1, 2, 3, 4, 5, 6]:
                log = f'{dep_dist} ' + ' {:.2f}'.format(percent_by_dep_dist[dep_dist] / (len(enum_langs) - 1))
                logger_.info(log)

    logger.clean_logger()

    logger_ = logger.get_arg_stat_logger(
        log_dir=args.log_dir
    )

    NO_RELATION = 'NO-RELATION'
    enum_langs = list(dataset.metadata_by_version_to_lang_to_treebank[args.upb_version].keys())

    # Arg stat
    for is_gold in [True, False]:
        annotation_type_w = 'GOLD' if is_gold else 'PRED'
        logger_.info(annotation_type_w)
        eval_filename = os.path.join(
            args.log_dir,
            'evals',
            'gold' if is_gold else 'pred',
            f'eval_{args.upb_version}_run_0.pkl'
        )
        evaluation = torch.load(f=eval_filename)

        for set_name in ['dev', 'test']:
            set_w = set_name.upper()
            logger_.info(set_w)
            for lang in enum_langs:
                lang_w = lang.upper()
                if evaluation[lang][set_name].get('arguments_by_dep_dist') is None:
                    break

                logger_.info(f'{annotation_type_w} {set_w} {lang_w}')

                total = 0
                total_argument = 0

                arg_by_dep_dist = {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 0
                }
                arg_stat_by_dep_dist = {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 0
                }

                for dep_dist in [0, 1, 2, 3, 4, 5, 6]:
                    token = sum(list(evaluation[lang][set_name]['arguments_by_dep_dist'][dep_dist].values()))
                    no_rel_token = evaluation[lang][set_name]['arguments_by_dep_dist'][dep_dist].get(NO_RELATION, 0)
                    argument = token - no_rel_token
                    total_argument += argument
                    total += token

                    arg_by_dep_dist[dep_dist] = argument

                    arg_stats = []

                    for key in evaluation[lang][set_name]['arguments_by_dep_dist'][dep_dist]:
                        if key == NO_RELATION:
                            continue

                        arg_stats.append([
                            key,
                            round((evaluation[lang][set_name]['arguments_by_dep_dist'][dep_dist][key] / argument) * 100, 2)
                        ])

                    arg_stats.sort(key=lambda x: x[1], reverse=True)
                    arg_stat_by_dep_dist[dep_dist] = arg_stats

                assert evaluation[lang][set_name]['total_tokens_by_dep_dist']['total'] == total

                for dep_dist in [0, 1, 2, 3, 4, 5, 6]:
                    dep_percent = round(evaluation[lang][set_name]['total_tokens_by_dep_dist'][dep_dist] / evaluation[lang][set_name]['total_tokens_by_dep_dist']['total'] * 100, 2)
                    log = f'dep {dep_dist}' + ' {:.2f} (num: {})'.format(dep_percent, evaluation[lang][set_name]['total_tokens_by_dep_dist'][dep_dist])
                    logger_.info(log)

                for dep_dist in [0, 1, 2, 3, 4, 5, 6]:
                    arg_percent = round((arg_by_dep_dist[dep_dist] / total_argument) * 100, 2)
                    log = f'arg {dep_dist}' + ' {:.2f} (num: {})'.format(arg_percent, arg_by_dep_dist[dep_dist])
                    logger_.info(log)

                for dep_dist in [0, 1, 2, 3, 4, 5, 6]:
                    log = f'arg {dep_dist} {arg_stat_by_dep_dist[dep_dist]}'
                    logger_.info(log)

    logger.clean_logger()
