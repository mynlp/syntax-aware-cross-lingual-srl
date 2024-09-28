import os
import math
import torch
import random
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from constants import dataset, word
from classes.timer import Timer
from classes.average_meter import AverageMeter
from models.semantic_role_labeler import SemanticRoleLabeler
from classes.upb_dataset import UPBDataset
from utils import logger, file, eval
from utils.parser import add_arguments, get_model_args, human_format
from tqdm import tqdm

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def draw_confusion_matrix(cm, labels, filename, logger, lang):
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['axes.titlesize'] = 18

    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            if i == j:
                annot[i, j] = '%d' % c
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%d' % c

    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=(16, 16))
    midpoint = (cm.values.max() - cm.values.min()) / 2
    # sns.set(font_scale=1.5)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax, center=midpoint,
                linewidths=0.01, cmap="Blues", cbar=False)
    plt.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(filename)
    # sns.set(font_scale=1.0)
    plt.close()


def validate(
    args,
    dev_loader,
    model,
    stats,
    mode,
    lang,
    idx_run,
    run_dep_dist_val=False
):
    eval_time = Timer()
    results = []
    results_by_dep_dist = {
        0: [],
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
        6: []
    }
    total_tokens_by_dep_dist = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        'total': 0
    }
    arguments_by_dep_dist = {
        0: {},
        1: {},
        2: {},
        3: {},
        4: {},
        5: {},
        6: {}
    }
    total_example = 0
    with torch.no_grad():
        pbar = tqdm(dev_loader)
        for ex in pbar:
            output = model.predict(ex)
            gold_labels = ex['sem_role_rep'].tolist()
            pred_dep_dists = ex['pred_dep_dist_rep'].tolist()
            for idx in range(len(gold_labels)):
                results.append({
                    'pred': model.semantic_role_voc[output['predictions'][idx]],
                    'gold': model.semantic_role_voc[gold_labels[idx]]
                })

                pred_dep_dist = pred_dep_dists[idx]

                if run_dep_dist_val and model.semantic_role_voc[gold_labels[idx]] != word.PAD_WORD:
                    gold_label = model.semantic_role_voc[gold_labels[idx]]
                    pred_label = model.semantic_role_voc[output['predictions'][idx]]
                    total_tokens_by_dep_dist['total'] += 1
                    if (pred_dep_dist >= 0 and pred_dep_dist <= 5):
                        total_tokens_by_dep_dist[pred_dep_dist] += 1
                        results_by_dep_dist[pred_dep_dist].append({
                            'pred': pred_label,
                            'gold': gold_label
                        })
                        if arguments_by_dep_dist[pred_dep_dist].get(gold_label) is None:
                            arguments_by_dep_dist[pred_dep_dist][gold_label] = 1
                        else:
                            arguments_by_dep_dist[pred_dep_dist][gold_label] += 1
                    elif pred_dep_dist >= 6:
                        total_tokens_by_dep_dist[6] += 1
                        results_by_dep_dist[6].append({
                            'pred': pred_label,
                            'gold': gold_label
                        })
                        if arguments_by_dep_dist[6].get(gold_label) is None:
                            arguments_by_dep_dist[6][gold_label] = 1
                        else:
                            arguments_by_dep_dist[6][gold_label] += 1

            pbar.set_description('%s' % 'Epoch = %d [validating ... ]' %
                                 stats['epoch'])
            total_example += ex['batch_size']

    scorer_out_by_dep_dist = {}

    if run_dep_dist_val:
        for dep_dist in results_by_dep_dist:
            scorer_out_by_dep_dist[dep_dist] = eval.score(
                results=results_by_dep_dist[dep_dist],
                labels=model.semantic_role_voc.tokens(),
                logger=args.logger,
                verbose=True
            )
            args.logger.info('Validation: dep dist = %d | precision = %.2f | recall = %.2f | f1 = %.2f |'
                             ' proportion = %.2f | %s time = %.2f (s) ' %
                             (dep_dist, scorer_out_by_dep_dist[dep_dist]['precision'] * 100, scorer_out_by_dep_dist[dep_dist]['recall'] * 100,
                              scorer_out_by_dep_dist[dep_dist]['f1'] * 100, total_tokens_by_dep_dist[dep_dist] / total_tokens_by_dep_dist['total'] * 100, mode, eval_time.time()))
            args.logger.info('\n' + scorer_out_by_dep_dist[dep_dist]['verbose_out'])

    scorer_out = eval.score(
        results=results,
        labels=model.semantic_role_voc.tokens(),
        logger=args.logger,
        verbose=True
    )
    args.logger.info('Validation: precision = %.2f | recall = %.2f | f1 = %.2f |'
                ' examples = %d | %s time = %.2f (s) ' %
                (scorer_out['precision'] * 100, scorer_out['recall'] * 100,
                 scorer_out['f1'] * 100, total_example, mode, eval_time.time()))
    args.logger.info('\n' + scorer_out['verbose_out'])

    # TODO: Arrange pred file
    # with open(args.pred_file, 'w') as fw:
    #     for item in results:
    #         fw.write(json.dumps(item) + '\n')

    if mode == 'test' and args.draw_conf_matrix:
        cm_filename = os.path.join(
            args.log_dir,
            'confusion_matrices',
            'gold' if args.gold else 'pred',
            f'confusion_matrix_{args.upb_version}_{lang}_run_{idx_run}.png'
        )
        draw_confusion_matrix(
            cm=scorer_out['confusion_matrix'],
            labels=scorer_out['labels'],
            filename=cm_filename,
            logger=args.logger,
            lang=lang
        )

    return {
        'arguments_by_dep_dist': arguments_by_dep_dist,
        'total_tokens_by_dep_dist': total_tokens_by_dep_dist,
        'scorer_out_by_dep_dist': scorer_out_by_dep_dist,
        'precision': scorer_out['precision'],
        'recall': scorer_out['recall'],
        'f1': scorer_out['f1'],
        'confusion_matrix': scorer_out['confusion_matrix'],
        'correct_by_relation': scorer_out['correct_by_relation'],
        'guessed_by_relation': scorer_out['guessed_by_relation'],
        'gold_by_relation': scorer_out['gold_by_relation']
    }


def rescale_lr(
    steps_per_epoch,
    step,
    args,
    model
):
    warmup_steps = args.warmup_epochs * steps_per_epoch
    cooldown_steps = warmup_steps + args.cooldown_epochs * steps_per_epoch
    training_steps = args.num_epochs * steps_per_epoch

    if step < warmup_steps:
        lr_scale = min(1., float(step + 1) / warmup_steps)
        model.optimizer.param_groups[0]['lr'] = lr_scale * args.learning_rate
    elif step < cooldown_steps:
        progress = float(step - warmup_steps) / float(max(1, cooldown_steps - warmup_steps))
        lr_scale = (1. - progress)
        model.optimizer.param_groups[0]['lr'] = args.min_lr + lr_scale * (args.learning_rate - args.min_lr)
    else:
        progress = float(step - cooldown_steps) / float(max(1, training_steps - cooldown_steps))
        lr_scale = (1. - progress)
        model.optimizer.param_groups[0]['lr'] = lr_scale * args.min_lr


def train(
    args,
    train_loader,
    model,
    stats,
    steps_per_epoch
):
    cl_loss = AverageMeter()
    epoch_time = Timer()

    pbar = tqdm(train_loader)
    pbar.set_description('%s' % 'Epoch = %d [loss = x.xx]' % stats['epoch'])

    starting_step = (stats['epoch'] - 1) * steps_per_epoch

    for idx, ex in enumerate(pbar):
        bsz = ex['batch_size']
        step = starting_step + idx

        if args.use_slanted_triangle_learning:
            rescale_lr(
                steps_per_epoch=steps_per_epoch,
                step=step,
                args=args,
                model=model
            )

        loss = model.update(ex)
        cl_loss.update(loss, bsz)

        log_info = 'Epoch = %d [loss = %.2f]' % (stats['epoch'], cl_loss.avg)
        pbar.set_description('%s' % log_info)

    args.logger.info('train: Epoch %d | loss = %.2f | Time for epoch = %.2f (s)' %
                     (stats['epoch'], cl_loss.avg, epoch_time.time()))


def supervised_training(
    args,
    model,
    start_epoch,
    train_loader,
    train_lang,
    dev_loader,
    steps_per_epoch,
    idx_run
):
    logger.add_handler(
        log_dir=args.log_dir,
        log_filename=f'train_{args.upb_version}_{train_lang}_run_{idx_run}',
        is_gold=args.gold
    )

    stats = {
        'timer': Timer(),
        'epoch': start_epoch,
        'best_valid': 0,
        'no_improvement': 0
    }

    for epoch in range(start_epoch, args.num_epochs + 1):
        stats['epoch'] = epoch
        train(
            args=args,
            train_loader=train_loader,
            model=model,
            stats=stats,
            steps_per_epoch=steps_per_epoch
        )
        result = validate(
            args=args,
            dev_loader=dev_loader,
            model=model,
            stats=stats,
            mode='dev',
            lang=train_lang,
            idx_run=idx_run
        )
        valid_metric_perf = float(result['{}'.format(args.valid_metric)])

        # Save best valid
        if valid_metric_perf > stats['best_valid']:
            args.logger.info('Best valid: %s = %.4f (epoch %d, %d updates)' %
                             (args.valid_metric, valid_metric_perf, stats['epoch'], model.updates))
            model.save(args.model_file)
            stats['best_valid'] = valid_metric_perf
            stats['no_improvement'] = 0
        else:
            stats['no_improvement'] += 1
            if stats['no_improvement'] >= args.num_early_stop:
                args.logger.info(f'Early stopping because there is no significant improvement after '
                                 f'{stats["no_improvement"]} epochs.')
                break
            if not args.use_slanted_triangle_learning:
                # if validation performance decreases, we decay the learning rate
                if epoch > args.num_decay_epoch:
                    old_lr = model.optimizer.param_groups[0]['lr']
                    new_lr = old_lr * args.lr_decay
                    model.optimizer.param_groups[0]['lr'] = new_lr
                    args.logger.info('Decaying the learning rate from {:.6} to {:.6} [rate:{}].'
                                     .format(old_lr, new_lr, args.lr_decay))
                    if new_lr < args.min_lr:
                        args.logger.info('Training stopped as the learning rate: {:.6} drops '
                                    'below the threshold {}.'.format(new_lr, args.min_lr))
                        break

    logger.remove_last_handler()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Cross-lingual Semantic Role Labeling',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    add_arguments(parser)

    args = parser.parse_args()
    random.seed(args.random_seed)
    file.create_folder_if_not_exist(args.model_dir)
    file.create_folder_if_not_exist(os.path.join(args.dataset_dir, 'gold'))
    file.create_folder_if_not_exist(os.path.join(args.dataset_dir, 'pred'))
    file.create_folder_if_not_exist(os.path.join(args.log_dir, 'logs', 'gold'))
    file.create_folder_if_not_exist(os.path.join(args.log_dir, 'logs', 'pred'))
    file.create_folder_if_not_exist(os.path.join(args.log_dir, 'evals', 'gold'))
    file.create_folder_if_not_exist(os.path.join(args.log_dir, 'evals', 'pred'))
    file.create_folder_if_not_exist(os.path.join(args.log_dir, 'confusion_matrices', 'gold'))
    file.create_folder_if_not_exist(os.path.join(args.log_dir, 'confusion_matrices', 'pred'))

    for i in range(args.idx_start_run):
        seed = random.randrange(1, 100)

    for idx_run in range(args.idx_start_run, args.idx_start_run + args.num_runs):
        seed = random.randrange(1, 100)

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        args.logger = logger.get_logger(
            log_dir=args.log_dir,
            idx_run=idx_run,
            is_gold=args.gold
        )

        args.cuda = torch.cuda.is_available()
        args.device_count = torch.cuda.device_count()
        args.parallel = args.device_count > 1
        args.logger.info(f'Is cuda available: {args.cuda}. Device count: {args.device_count}.')
        args.logger.info(f'Run: {idx_run}. Seed: {seed}.')

        train_lang = 'en'
        set_name = 'train'

        args.model_file = os.path.join(
            args.model_dir,
            f'model_{args.upb_version}_{train_lang}_run_{idx_run}.mdl'
        )
        start_epoch = 1

        if args.test:
            if not os.path.isfile(args.model_file):
                raise IOError(f'No such file: {args.model_file}.')
            model = SemanticRoleLabeler.load(args.model_file, args.logger)
        else:
            model = SemanticRoleLabeler(get_model_args(args))
            model.init_optimizer()
            args.logger.info(f'Trainable #parameters [total] {human_format(model.get_network().count_parameters())}')
            table = model.get_network().layer_wise_parameters()
            args.logger.info(f'Breakdown of the trainable parameters\n{table}')

        if args.cuda:
            model.cuda()

        if args.parallel:
            model.parallelize()

        dataset_dist = None if args.test else dict()
        dataset_dist_filename = os.path.join(
            args.dataset_dir,
            f'dataset_dist_{args.upb_version}.pkl'
        )

        dev_loader_by_lang = dict()

        dev_dataset = UPBDataset(
            model=model,
            args=args,
            set_name='dev',
            lang='en',
            dataset_dist=dataset_dist
        )

        dev_loader_by_lang['en'] = torch.utils.data.DataLoader(
            dev_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_data_workers,
            collate_fn=dev_dataset.batchify,
            pin_memory=args.cuda
        )

        test_loader_by_lang = dict()

        test_dataset = UPBDataset(
            model=model,
            args=args,
            set_name='test',
            lang='en',
            dataset_dist=dataset_dist
        )

        test_loader_by_lang['en'] = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_data_workers,
            collate_fn=test_dataset.batchify,
            pin_memory=args.cuda
        )

        if not args.test:
            train_dataset = UPBDataset(
                model=model,
                args=args,
                set_name=set_name,
                lang=train_lang,
                dataset_dist=dataset_dist
            )

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_data_workers,
                collate_fn=train_dataset.batchify,
                pin_memory=args.cuda
            )

            steps_per_epoch = math.ceil(len(train_dataset) / args.batch_size)

            supervised_training(
                args=args,
                model=model,
                start_epoch=start_epoch,
                train_loader=train_loader,
                train_lang=train_lang,
                dev_loader=dev_loader_by_lang['en'],
                steps_per_epoch=steps_per_epoch,
                idx_run=idx_run
            )

        for lang in dataset.metadata_by_version_to_lang_to_treebank[args.upb_version]:
            if lang == train_lang:
                continue

            args.logger.info(f'Load dataset from {lang} dev.')

            dev_dataset = UPBDataset(
                model=model,
                args=args,
                set_name='dev',
                lang=lang,
                dataset_dist=dataset_dist
            )

            dev_loader_by_lang[lang] = torch.utils.data.DataLoader(
                dev_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_data_workers,
                collate_fn=dev_dataset.batchify,
                pin_memory=args.cuda
            )

            args.logger.info(f'Load dataset from {lang} test.')

            test_dataset = UPBDataset(
                model=model,
                args=args,
                set_name='test',
                lang=lang,
                dataset_dist=dataset_dist
            )

            test_loader_by_lang[lang] = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_data_workers,
                collate_fn=test_dataset.batchify,
                pin_memory=args.cuda
            )

        torch.save(dataset_dist, dataset_dist_filename)

        evaluation = dict()
        log_filename = os.path.join(
            args.log_dir,
            'evals',
            'gold' if args.gold else 'pred',
            f'eval_{args.upb_version}_run_{idx_run}.pkl'
        )

        for lang in dataset.metadata_by_version_to_lang_to_treebank[args.upb_version]:
            stats = {
                'timer': Timer(),
                'epoch': 0,
                'best_valid': 0,
                'no_improvement': 0
            }

            logger.add_handler(
                log_dir=args.log_dir,
                log_filename=f'dev_{args.upb_version}_{lang}_run_{idx_run}',
                is_gold=args.gold
            )

            args.logger.info(f'Validate model using {lang} dev.')

            dev_result = validate(
                args=args,
                dev_loader=dev_loader_by_lang[lang],
                model=model,
                stats=stats,
                mode='dev',
                lang=lang,
                idx_run=idx_run,
                run_dep_dist_val=True
            )

            logger.remove_last_handler()

            logger.add_handler(
                log_dir=args.log_dir,
                log_filename=f'test_{args.upb_version}_{lang}_run_{idx_run}',
                is_gold=args.gold
            )

            args.logger.info(f'Validate model using {lang} test.')

            test_result = validate(
                args=args,
                dev_loader=test_loader_by_lang[lang],
                model=model,
                stats=stats,
                mode='test',
                lang=lang,
                idx_run=idx_run,
                run_dep_dist_val=True
            )

            logger.remove_last_handler()

            evaluation[lang] = {
                'dev': {
                    'arguments_by_dep_dist': dev_result['arguments_by_dep_dist'],
                    'total_tokens_by_dep_dist': dev_result['total_tokens_by_dep_dist'],
                    'scorer_out_by_dep_dist': dev_result['scorer_out_by_dep_dist'],
                    'precision': dev_result['precision'],
                    'recall': dev_result['recall'],
                    'f1': dev_result['f1'],
                    'confusion_matrix': dev_result['confusion_matrix'],
                    'correct_by_relation': dev_result['correct_by_relation'],
                    'guessed_by_relation': dev_result['guessed_by_relation'],
                    'gold_by_relation': dev_result['gold_by_relation']
                },
                'test': {
                    'arguments_by_dep_dist': test_result['arguments_by_dep_dist'],
                    'total_tokens_by_dep_dist': test_result['total_tokens_by_dep_dist'],
                    'scorer_out_by_dep_dist': test_result['scorer_out_by_dep_dist'],
                    'precision': test_result['precision'],
                    'recall': test_result['recall'],
                    'f1': test_result['f1'],
                    'confusion_matrix': test_result['confusion_matrix'],
                    'correct_by_relation': test_result['correct_by_relation'],
                    'guessed_by_relation': test_result['guessed_by_relation'],
                    'gold_by_relation': test_result['gold_by_relation']
                }
            }

        torch.save(evaluation, log_filename)

        logger.clean_logger()
