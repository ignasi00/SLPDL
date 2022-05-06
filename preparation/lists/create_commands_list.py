
# Initially the data labels are managed by the folder structure or file name.
# The train lists (train, val, kfolds, etc) will contain columns with the path to the file, the text and the speaker (the only important information of the file path is the name but it is comfortable with all the path).
# The test lists will contain will only contain filenames.
# 

import os


FILE = 'wav'
TEXT = 'text'
SPEAKER = 'speaker'

def commands_train_list(path, file_key=FILE, text_key=TEXT, speaker_key=SPEAKER):
    # From folders structure
    
    def generate_entry(root, name):
        entry = {
            str(file_key)       : os.path.join(root, name),
            str(text_key)       : root.split('/')[-1],
            str(speaker_key)    : name.split('_')[0]
        }
        return entry
    
    list_ = []
    for root, _, files in os.walk(path):
        list_.extend((generate_entry(root, name) for name in files))

    return list_

def digit_train_list(path, repetitions=None, file_key=FILE, text_key=TEXT, speaker_key=SPEAKER):
    # From filename, repetition filter is a list like object
    def generate_entry(f):
        filename_split = f.name.split('_')
        repetition = int(filename_split[-1].split('.')[0])
        if repetitions is not None and repetition not in repetitions : return None
        entry =  {
            str(file_key)       : f.path,
            str(text_key)       : filename_split[0],
            str(speaker_key)    : filename_split[1]
        }
        return entry

    return [generate_entry(f) for f in os.scandir(path) if f.is_file() and generate_entry(f) is not None]

def commands_test_list(path, file_key=FILE):
    # Just list folder files
    return [{str(file_key) : f.path} for f in os.scandir(path) if f.is_file()]
