import os
from pathlib import Path
import argparse


def validate_dir(rel_path=''):
    """
    Validates path and creates parent/child folders if path is not existent
    :param rel_path: Relative path from the current directory to the target directory
    :type rel_path: str
    :return: Path as a string
    """
    curr_dir = os.getcwd()
    Path('%s/%s/' % (curr_dir, rel_path)).mkdir(parents=True, exist_ok=True)
    return '%s/%s/' % (curr_dir, rel_path)


def argparse_jmak_param(arguments):
    parser = argparse.ArgumentParser(
        description='Find the best JMAK parameters for CPD data and save them to file.'
    )
    parser.add_argument('--do_each', action='store_true', dest='do_each',
                        help='If set, there is one model per region in each transcript. '
                             'Otherwise beginning, centre of a gene and end are combined in one single model.')
    parser.add_argument('--min_f_trans', type=float, default=.5,
                        help='Minimum allowed value for maximum fraction in transcribed regions '
                             'during parameter search.')
    parser.add_argument('--min_f_igr', type=float, default=.4,
                        help='Minimum allowed value for maximum fraction in intergenic regions '
                             'during parameter search.')
    parser.add_argument('--max_f', type=float, default=1.,
                        help='Maximum allowed value for maximum fraction during parameter search.')
    parser.add_argument('--delta_f', type=float, default=.01,
                        help='Step size when increasing maximum fraction during parameter search.')
    parser.add_argument('--num_cpus', type=int, default=1,
                        help='Number of processes used for speeding up computations.')
    parser.add_argument('--verbosity', type=int, default=2,
                        help='Verbosity flag')
    parser.add_argument('--save_fig', action='store_true', dest='save_fig',
                        help='If set, figures are saved instead of displayed.')
    parser.add_argument('--test', action='store_true', dest='test',
                        help='Run test function for debugging.')

    parsed_args = parser.parse_args(arguments)
    return parsed_args


def argparse_bioplotter(arguments):
    parser = argparse.ArgumentParser(
        description='Plot available biological data wrt the parameter space. Train the JMAK model and save'
                    'them to a file first.'
    )

    parser.add_argument('--bio_type', type=str, required=True,
                        help='Pass the data type that is used in order to determine the colour gradient.'
                             'Possible is: slam | nucl | size | meres'
                        )
    parser.add_argument('--do_each', action='store_true', dest='do_each',
                        help='If set, there is one model per region in each transcript. '
                             'Otherwise beginning, centre of a gene and end are combined in one single model.')
    parser.add_argument('--save_fig', action='store_true', dest='save_fig',
                        help='If set, figures are saved instead of displayed.')
    parser.add_argument('--size_scaling', type=float, default=100.,
                        help='Beta parameter is mulitplied by this value to determine the size.')
    parser.add_argument('--size_power', type=float, default=3.,
                        help='Scaled beta parameter is taken to the power of size_power to make small changes visible.')
    parser.add_argument('--power_norm', type=float, default=1.,
                        help='The colour gradient c is transformed c^power_norm.')

    parsed_args = parser.parse_args(arguments)
    return parsed_args

