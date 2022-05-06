
from docopt import docopt


OUTPUT_ROOT = './data_lists/'

FILE = 'wav'
TEXT = 'text'
SPEAKER = 'speaker'

VERBOSE = False


DOCTEXT = f"""
Usage:
  create_list.py commands [options] (--probs=<p>... --names=<n>...)
  create_list.py digits [options] (--probs=<p>... --names=<n>...) [--repetitions=<r>...|--first_n_repetitions=<fnr>]
  create_list.py commands_test --data_root=<dr> --outputs_root=<or> [--file_key=<fk>] (--probs=<p>... --names=<n>...) [--verbose=<v>]
  create_list.py -h | --help

Options:
  -h --help                             Show this screen.
  --data_root=<dr>                      Str. Path of the folder where the data is stored.
  --probs=<p>                           Float vector. Positive numbers that represent a fraction of the data available, they will be normalized (divided by their sum).
  --names=<n>                           Str vector. Name associated to each probs value, also the list output name.
  --outputs_root=<or>                   Str. Path where the list will be generated [default: {OUTPUT_ROOT}].
  --file_key=<fk>                       Str. When the generate list use a filename element and a title is needed, this string will be used [default: {FILE}].
  --text_key=<tk>                       Str. When the generate list use a text element and a title is needed, this string will be used [default: {TEXT}].
  --speaker_key=<sk>                    Str. When the generate list use a speaker element and a title is needed, this string will be used [default: {SPEAKER}].
  --repetitions=<r>                     Int vector. Index of the allowed recording version; if no recordings are selected, all of them  will be used [default: {None}].
  --first_n_repetitions=<fnr>           Int. Instead of selecting the recording version indexes one by one, this option allows to generate them as in list(range(<fnr>)) [default: {None}].
  --verbose=<v>                         Bool. Some list creation script may show information as it runs, this parameter manage it [default: {VERBOSE}].

"""


def parse_args(argv):
    opts = docopt(DOCTEXT, argv=argv[1:], help=True, version=None, options_first=False)

    type_ = None
    if opts['commands'] : type_ = "commands"
    elif opts['digits'] : type_ = "digits"
    elif opts['commands_test'] : type_ = "commands_test"

    data_root = opts['--data_root']
    outputs_root = opts['--outputs_root']

    probs = [float(x) for x in opts['--probs']]
    probs = [x/sum(probs) for x in probs]
    names = opts['--names']

    file_key = opts['--file_key']
    text_key = opts['--text_key']
    speaker_key = opts['--speaker_key']

    repetitions = None
    if opts['--repetitions'] is not None and 'None' not in opts['--repetitions'] : repetitions = [int(x) for x in opts['--repetitions']]
    elif opts['--repetitions'] is not None and opts['--first_n_repetitions'] != 'None' : repetitions = list(range(int(opts['--first_n_repetitions'])))

    verbose = opts['--verbose'] == "True"

    args = (type_, data_root, outputs_root, probs, names, file_key, text_key, speaker_key, repetitions, verbose)
    return args

