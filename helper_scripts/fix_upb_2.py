import os
from utils import file, dataset


if __name__ == '__main__':
    # TODO: Recheck if we want to use train dataset or argument span in the future
    folders = [
        'UP_Czech-FicTree',
        'UP_Czech-PDT',
        'UP_Czech-CAC',
        'UP_Dutch-Alpino',
        'UP_Dutch-LassySmall',
        'UP_Ukrainian-IU',
        'UP_Finnish-TDT',
        'UP_Italian-ISDT',
        'UP_Spanish-AnCora'
    ]

    for folder in folders:
        folder_path = f'../../resources/UniversalPropositions/{folder}'
        filenames = os.listdir(folder_path)
        filenames = [filename for filename in filenames if filename.endswith('.conllup')]
        assert len(filenames) == 3

        for filename in filenames:
            file_path = f'{folder_path}/{filename}'
            src = open(file_path, 'r')
            file.create_folder_if_not_exist(f'{folder_path}_fixed')
            dest = open(f'{folder_path}_fixed/{filename}', 'w')

            if 'dev' in filename:
                metadata = open(f'{folder_path}_fixed/metadata-dev.txt', 'w')
            elif 'test' in filename:
                metadata = open(f'{folder_path}_fixed/metadata-test.txt', 'w')
            elif 'train' in filename:
                metadata = open(f'{folder_path}_fixed/metadata-train.txt', 'w')

            line = src.readline()

            sentences = []

            while line:
                sent_block = []
                while line and line != '\n':
                    stripped_line = line.strip()
                    sent_block.append(stripped_line)
                    line = src.readline()

                comments = []
                all_token_attrs = []

                is_enhanced = False
                idxs = []

                for sent_line in sent_block:
                    if dataset.is_comment(sent_line):
                        comments.append(f'{sent_line}\n')
                    else:
                        token_attrs = sent_line.split('\t')
                        if '-' not in token_attrs[0]:
                            idxs.append(len(all_token_attrs))

                        if '.' in token_attrs[0]:
                            is_enhanced = True

                        all_token_attrs.append(token_attrs)

                if is_enhanced:
                    is_problem = False

                    # Shift and check on predicate level
                    count_sub = 0
                    for idx, token_attrs in enumerate(all_token_attrs):
                        if '.' in token_attrs[0]:
                            count_sub += 1

                        if count_sub > 0 and (
                                all_token_attrs[idx][1] != '_'
                                or all_token_attrs[idx][2] != '_'
                                or all_token_attrs[idx][3] != '_'
                        ):
                            # if count_sub > 1 and not is_problem:
                            #     metadata.writelines(comments)
                            #     metadata.write('\n')
                            #     is_problem = True

                            new_idx = idxs[int(all_token_attrs[idx][0]) - 1]
                            assert '-' not in all_token_attrs[idx][0], print(comments, all_token_attrs[new_idx][0])
                            assert '-' not in all_token_attrs[new_idx][0], print(comments, all_token_attrs[new_idx][0])
                            assert all_token_attrs[new_idx][1] == '_', print(comments, all_token_attrs[new_idx][0])
                            assert all_token_attrs[new_idx][2] == '_', print(comments, all_token_attrs[new_idx][0])
                            assert all_token_attrs[new_idx][3] == '_', print(comments, all_token_attrs[new_idx][0])
                            all_token_attrs[new_idx][1:4] = all_token_attrs[idx][1:4]
                            all_token_attrs[idx][1:4] = ['_', '_', '_']

                            if '.' in all_token_attrs[new_idx][0] and not is_problem:
                                metadata.writelines(comments)
                                metadata.write('\n')
                                is_problem = True

                    # Check on argument level
                    if not is_problem:
                        for token_attrs in all_token_attrs:
                            if token_attrs[2] != '_':
                                tmp = [chunk.split(':') for chunk in token_attrs[2].split('|')]

                                for tm in tmp:
                                    assert len(tm) == 2
                                    if '.' in all_token_attrs[idxs[int(tm[1]) - 1]][0] and not is_problem:
                                        metadata.writelines(comments)
                                        metadata.write('\n')
                                        is_problem = True

                            # We are not going to use argument spans annotation
                            # if token_attrs[3] != '_':
                            #     tmp = [chunk.split(':') for chunk in token_attrs[3].split('|')]
                            #
                            #     for tm in tmp:
                            #         assert len(tm) == 2
                            #         [fr, to] = tm[1].split('-')
                            #         if (
                            #                 '.' in all_token_attrs[idxs[int(fr) - 1]][0] or
                            #                 '.' in all_token_attrs[idxs[int(to) - 1]][0]
                            #         ) and not is_problem:
                            #             metadata.writelines(comments)
                            #             metadata.write('\n')
                            #             is_problem = True

                    # Adjust arguments
                    dest.writelines(comments)
                    for token_attrs in all_token_attrs:
                        if token_attrs[2] != '_':
                            tmp = [chunk.split(':') for chunk in token_attrs[2].split('|')]

                            for tm in tmp:
                                assert len(tm) == 2
                                temp = tm[1]
                                tm[1] = str(all_token_attrs[idxs[int(tm[1]) - 1]][0])
                                if (
                                        folder in ['UP_Dutch-Alpino', 'UP_Finnish-TDT', 'UP_Dutch-LassySmall'] and
                                        '.' not in token_attrs[0] and
                                        '.' in tm[1]
                                ):
                                    print(f'Adjust {tm[1]} to')
                                    tm[1] = str(all_token_attrs[idxs[int(temp) - 2]][0])
                                    print(tm[1])

                            tmp = [':'.join(chunk) for chunk in tmp]
                            token_attrs[2] = '|'.join(tmp)

                        if token_attrs[3] != '_':
                            tmp = [chunk.split(':') for chunk in token_attrs[3].split('|')]

                            for tm in tmp:
                                assert len(tm) == 2
                                tms = tm[1].split('-')
                                temp = tms[0]
                                tms[0] = str(all_token_attrs[idxs[int(tms[0]) - 1]][0])
                                if (
                                        folder in ['UP_Dutch-Alpino', 'UP_Finnish-TDT', 'UP_Dutch-LassySmall'] and
                                        '.' not in token_attrs[0] and
                                        '.' in tms[0]
                                ):
                                    print(f'Adjust {tms[0]} to')
                                    tms[0] = str(all_token_attrs[idxs[int(temp) - 2]][0])
                                    print(tms[0])
                                temp = tms[1]
                                tms[1] = str(all_token_attrs[idxs[int(tms[1]) - 1]][0])
                                if (
                                        folder in ['UP_Dutch-Alpino', 'UP_Finnish-TDT', 'UP_Dutch-LassySmall'] and
                                        '.' not in token_attrs[0] and
                                        '.' in tms[1]
                                ):
                                    print(f'Adjust {tms[1]} to')
                                    tms[1] = str(all_token_attrs[idxs[int(temp) - 2]][0])
                                    print(tms[1])

                                tm[1] = '-'.join(tms)

                            tmp = [':'.join(chunk) for chunk in tmp]
                            token_attrs[3] = '|'.join(tmp)

                        lline = '\t'.join(token_attrs)
                        dest.write(f'{lline}\n')
                    dest.write('\n')
                else:
                    dest.writelines(comments)
                    for token_attrs in all_token_attrs:
                        lline = '\t'.join(token_attrs)
                        dest.write(f'{lline}\n')
                    dest.write('\n')

                while line and line == '\n':
                    line = src.readline()

            src.close()
            dest.close()
            metadata.close()
