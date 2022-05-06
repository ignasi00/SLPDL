
import pandas as pd
import pathlib
import sys

from docopts.help_create_lists import parse_args
from preparation.lists.create_commands_list import commands_train_list, digit_train_list, commands_test_list
from preparation.lists.utils.split_list import split_list, split_pandas


def save_pandas(save_path, list_of_items, columns=None):
    columns = columns or list_of_items.columns
    list_of_items.to_csv(save_path, header=False, index=False, columns=columns)

if __name__ == "__main__":
    args = parse_args(sys.argv)
    (type_, path, outputs_root, probs, names, file_key, text_key, speaker_key, repetitions, verbose) = args

    pathlib.Path(outputs_root).mkdir(parents=True, exist_ok=True)

    if type_ == "commands":
        list_of_dicts = commands_train_list(path, file_key=file_key, text_key=text_key, speaker_key=speaker_key)
        panda_table = pd.DataFrame(list_of_dicts)
        #.to_dict(orient='records')

        lists = split_pandas(panda_table, probs=probs, verbose=verbose)
        for l, n in zip(lists, names):
            save_pandas(f'{outputs_root}{n}', l, columns=[file_key, text_key, speaker_key])

    elif type_ == "digits":
        list_of_dicts = digit_train_list(path, repetitions=repetitions, file_key=file_key, text_key=text_key, speaker_key=speaker_key)
        panda_table = pd.DataFrame(list_of_dicts)
        #.to_dict(orient='records')

        lists = split_pandas(panda_table, probs=probs, verbose=verbose)
        for l, n in zip(lists, names):
            save_pandas(f'{outputs_root}{n}', l, columns=[file_key, text_key, speaker_key])
    
    elif type_ == "commands_test":
        list_of_dicts = commands_test_list(path, file_key=file_key)
        panda_table = pd.DataFrame(list_of_dicts)
        #.to_dict(orient='records')

        lists = split_pandas(panda_table, probs=probs, verbose=verbose)
        for l, n in zip(lists, names):
            save_pandas(f'{outputs_root}{n}', l, columns=[file_key])

