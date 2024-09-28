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
        '--upb_version',
        type=int,
        help='UPB version.',
        required=True
    )

    args = parser.parse_args()

    all_evaluation = dict()

    log_dirs = [
    ]
    max_log_dir = dict()
    max_f1 = dict()

    for log_dir in log_dirs:
        for is_gold in [True, False]:
            acc_evaluation = dict()
            acc_evaluation['avg'] = {
                'dev': {
                    'precision': [0] * args.num_runs,
                    'recall': [0] * args.num_runs,
                    'f1': [0] * args.num_runs
                },
                'test': {
                    'precision': [0] * args.num_runs,
                    'recall': [0] * args.num_runs,
                    'f1': [0] * args.num_runs
                }
            }

            for idx_run in range(args.num_runs):
                eval_filename = os.path.join(
                    log_dir,
                    'evals',
                    'gold' if is_gold else 'pred',
                    f'eval_{args.upb_version}_run_{idx_run}.pkl'
                )
                evaluation = torch.load(f=eval_filename)

                for lang in dataset.metadata_by_version_to_lang_to_treebank[args.upb_version]:
                    if acc_evaluation.get(lang) is None:
                        acc_evaluation[lang] = {
                            'dev': {
                                'precision': [],
                                'recall': [],
                                'f1': []
                            },
                            'test': {
                                'precision': [],
                                'recall': [],
                                'f1': []
                            }
                        }

                    for set_name in ['dev', 'test']:
                        acc_evaluation[lang][set_name]['precision'].append(evaluation[lang][set_name]['precision'])
                        acc_evaluation[lang][set_name]['recall'].append(evaluation[lang][set_name]['recall'])
                        acc_evaluation[lang][set_name]['f1'].append(evaluation[lang][set_name]['f1'])

                        if lang != 'en':
                            acc_evaluation['avg'][set_name]['precision'][idx_run] += evaluation[lang][set_name]['precision']
                            acc_evaluation['avg'][set_name]['recall'][idx_run] += evaluation[lang][set_name]['recall']
                            acc_evaluation['avg'][set_name]['f1'][idx_run] += evaluation[lang][set_name]['f1']

            final_evaluation = dict()
            num_langs = len(dataset.metadata_by_version_to_lang_to_treebank[args.upb_version]) - 1  # english not included
            enum_langs = list(dataset.metadata_by_version_to_lang_to_treebank[args.upb_version].keys()) + ['avg']

            for lang in enum_langs:
                final_evaluation[lang] = {
                    'dev': {
                        'precision': {},
                        'recall': {},
                        'f1': {}
                    },
                    'test': {
                        'precision': {},
                        'recall': {},
                        'f1': {}
                    }
                }

                for set_name in ['dev', 'test']:
                    assert len(acc_evaluation[lang][set_name]['precision']) == args.num_runs
                    assert len(acc_evaluation[lang][set_name]['recall']) == args.num_runs
                    assert len(acc_evaluation[lang][set_name]['f1']) == args.num_runs

                    if lang == 'avg':
                        acc_evaluation[lang][set_name]['precision'] = [val / num_langs for val in acc_evaluation[lang][set_name]['precision']]
                        acc_evaluation[lang][set_name]['recall'] = [val / num_langs for val in acc_evaluation[lang][set_name]['recall']]
                        acc_evaluation[lang][set_name]['f1'] = [val / num_langs for val in acc_evaluation[lang][set_name]['f1']]

                    if args.num_runs > 1:
                        final_evaluation[lang][set_name]['precision']['stdev'] = statistics.stdev(
                            acc_evaluation[lang][set_name]['precision']
                        )
                        final_evaluation[lang][set_name]['recall']['stdev'] = statistics.stdev(
                            acc_evaluation[lang][set_name]['recall']
                        )
                        final_evaluation[lang][set_name]['f1']['stdev'] = statistics.stdev(
                            acc_evaluation[lang][set_name]['f1']
                        )
                        final_evaluation[lang][set_name]['precision']['stderr'] = final_evaluation[lang][set_name]['precision']['stdev'] / math.sqrt(args.num_runs)
                        final_evaluation[lang][set_name]['recall']['stderr'] = final_evaluation[lang][set_name]['recall']['stdev'] / math.sqrt(args.num_runs)
                        final_evaluation[lang][set_name]['f1']['stderr'] = final_evaluation[lang][set_name]['f1']['stdev'] / math.sqrt(args.num_runs)

                    final_evaluation[lang][set_name]['precision']['avg'] = sum(acc_evaluation[lang][set_name]['precision']) / args.num_runs
                    final_evaluation[lang][set_name]['recall']['avg'] = sum(acc_evaluation[lang][set_name]['recall']) / args.num_runs
                    final_evaluation[lang][set_name]['f1']['avg'] = sum(acc_evaluation[lang][set_name]['f1']) / args.num_runs

            all_evaluation['gold' if is_gold else 'pred'] = final_evaluation

        for set_name in ['dev', 'test']:
            if not max_log_dir.get(set_name):
                max_log_dir[set_name] = dict()
                max_f1[set_name] = dict()

            for lang in ['avg', 'en']:
                if not max_log_dir[set_name].get(lang):
                    max_log_dir[set_name][lang] = dict()
                    max_f1[set_name][lang] = dict()

                for annotation_type in ['pred', 'gold', 'avg']:
                    if not max_log_dir[set_name][lang].get(annotation_type):
                        max_log_dir[set_name][lang][annotation_type] = []
                        max_f1[set_name][lang][annotation_type] = 0

                    if annotation_type == 'avg':
                        final_evaluation_pred = all_evaluation['pred']
                        f1_pred = final_evaluation_pred[lang][set_name]['f1']['avg'] * 100

                        final_evaluation_gold = all_evaluation['gold']
                        f1_gold = final_evaluation_gold[lang][set_name]['f1']['avg'] * 100

                        f1 = (f1_pred + f1_gold) / 2
                    else:
                        final_evaluation = all_evaluation[annotation_type]
                        f1 = final_evaluation[lang][set_name]['f1']['avg'] * 100

                    f1 = round(f1, 2)

                    if f1 == max_f1[set_name][lang][annotation_type]:
                        max_log_dir[set_name][lang][annotation_type].append(log_dir)
                    elif f1 > max_f1[set_name][lang][annotation_type]:
                        max_log_dir[set_name][lang][annotation_type] = [log_dir]
                        max_f1[set_name][lang][annotation_type] = f1

    for set_name in ['dev', 'test']:
        print(set_name)
        for lang in ['avg', 'en']:
            print(lang)
            for annotation_type in ['pred', 'gold', 'avg']:
                print(annotation_type)
                print(set_name, lang, annotation_type, max_log_dir[set_name][lang][annotation_type])
                print(set_name, lang, annotation_type, '{:.2f}'.format(max_f1[set_name][lang][annotation_type]))

# srun -p p --gres=gpu:1 --mem=64GB python read_evaluation_result2.py --num_runs 1 --upb_version 2
