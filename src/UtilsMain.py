import argparse

from src.Model import RegionModel
from src.DataLoader import create_time_data


def create_models(train_data, test_data, do_each=True):
    (train_trans, train_ntrans, train_igr, train_start_igr, _, train_transcriptome) = train_data
    (test_trans, test_ntrans, test_igr, test_start_igr, _, test_transcriptome) = test_data
    if do_each:
        iter = zip(
            [
                train_trans[:, :, 0],
                train_trans[:, :, 1],
                train_trans[:, :, 2],
                train_ntrans[:, :, 0],
                train_ntrans[:, :, 1],
                train_ntrans[:, :, 2],
                train_igr,
                test_trans[:, :, 0],
                test_trans[:, :, 1],
                test_trans[:, :, 2],
                test_ntrans[:, :, 0],
                test_ntrans[:, :, 1],
                test_ntrans[:, :, 2],
                test_igr
            ],
            [
                'Train genes start',
                'Train genes centre',
                'Train genes end',
                'Train NTS start',
                'Train NTS centre',
                'Train NTS end',
                'Train intergenic regions',
                'Test genes start',
                'Test genes centre',
                'Test genes end',
                'Test NTS start',
                'Test NTS centre',
                'Test NTS end',
                'Test intergenic regions'
            ],
            [
                'data/jmak/train_trans_start.csv',
                'data/jmak/train_trans_centre.csv',
                'data/jmak/train_trans_end.csv',
                'data/jmak/train_ntrans_start.csv',
                'data/jmak/train_ntrans_centre.csv',
                'data/jmak/train_ntrans_end.csv',
                'data/jmak/train_igr.csv',
                'data/jmak/test_trans_start.csv',
                'data/jmak/test_trans_centre.csv',
                'data/jmak/test_trans_end.csv',
                'data/jmak/test_ntrans_start.csv',
                'data/jmak/test_ntrans_centre.csv',
                'data/jmak/test_ntrans_end.csv',
                'data/jmak/test_igr.csv'
            ]
        )
    else:
        iter = zip(
            [
                train_trans,
                train_ntrans,
                train_igr,
                test_trans,
                test_ntrans,
                test_igr
            ],
            [
                'Train genes total',
                'Train NTS total',
                'Train intergenic regions',
                'Test genes total',
                'Test NTS total',
                'Test intergenic regions'
            ],
            [
                'data/jmak/train_trans_total.csv',
                'data/jmak/train_ntrans_total.csv',
                'data/jmak/train_igr.csv',
                'data/jmak/test_trans_total.csv',
                'data/jmak/test_ntrans_total.csv',
                'data/jmak/test_igr.csv'
            ]
        )

    model_list = []
    for data, name, file_name in iter:
        if 'train' in name.lower():
            if 'trans' in file_name.lower():
                chrom_list = train_transcriptome['chr'].to_list()
            else:
                chrom_list = map(lambda x: x[0], train_start_igr)
        else:
            if 'trans' in file_name.lower():
                chrom_list = test_transcriptome['chr'].to_list()
            else:
                chrom_list = map(lambda x: x[0], test_start_igr)
        if not do_each and 'trans' in file_name:
            num_pos = 3
        elif 'igr' in file_name:
            num_pos = 2
        else:
            num_pos = 1
        region_model = RegionModel(
            create_time_data(num_pos, len(data)),
            data.reshape(len(data), -1),
            name=name
        )
        region_model.load_models(file_name, compare_chrom_list=chrom_list)
        model_list.append(region_model)

    return model_list


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
                             'Possible are: slam | nucl | size | meres'
                        )
    parser.add_argument('--use_sum', action='store_true', dest='use_sum',
                        help='If set, when the biological data is loaded, the sum is taken instead of the mean.'
                             'Only important for bio_type=slam or bio_type=nucl')
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
    parser.add_argument('--save_prefix', type=str, default='',
                        help='Set identifying string that goes in front of every saved file name.')

    parsed_args = parser.parse_args(arguments)
    return parsed_args


def argparse_predict(arguments):
    parser = argparse.ArgumentParser(
        description='Learn function to approximate JMAK parameters given biological data. Train the JMAK model and save'
                    'them to a file first.'
    )

    parser.add_argument('--bio_type', type=str, required=True,
                        help='Pass the data type that is used in order to determine the function '
                             'and to make predictions Possible are: slam | nucl | size | meres')
    parser.add_argument('--do_each', action='store_true', dest='do_each',
                        help='If set, there is one model per region in each transcript. '
                             'Otherwise beginning, centre of a gene and end are combined in one single model.')
    parser.add_argument('--save_prefix', type=str, default='',
                        help='Set identifying string that goes in front of every saved file name.')
    parser.add_argument('--save_fig', action='store_true', dest='save_fig',
                        help='If set, figures are saved instead of displayed.')
    parser.add_argument('--to_pickle', action='store_true', dest='to_pickle',
                        help='If set, learnt prediction models are saved in a pickle file.')
    parser.add_argument('--kernel_func_type', type=str, default='eqk',
                        help='Pass the kernel function that is used for the Gaussian process. '
                             'Possible are: eqk | gaussian')
    parser.add_argument('--kernel_search_verbosity', type=int, default=0,
                        help='Set the verbosity level for finding optimal hyper parameters.')
    parser.add_argument('--kernel_scaling_init', type=int, default=10,
                        help='Set the initialisation scaling of the kernel hyperparameters. The initial value'
                             'is then set to {random number} * kernel_scaling_init.')
    parser.add_argument('--noise_scaling_init', type=int, default=5,
                        help='Set the initialisation scaling of the noise. The initial value is determined by '
                             '{random number} * noise_scaling_init.')
    parser.add_argument('--min_m', type=float, default=.4,
                        help='Minimum value for shaping parameter m for learnt parameter map.')
    parser.add_argument('--max_m', type=float, default=5.,
                        help='Maximum value for shaping parameter m for learnt parameter map.')
    parser.add_argument('--min_beta', type=float, default=8e-3,
                        help='Minimum value for scaling parameter beta for learnt parameter map.')
    parser.add_argument('--max_beta', type=float, default=3.5e-2,
                        help='Maximum value for shaping scaling parameter beta for learnt parameter map.')
    parser.add_argument('--min_mf', type=float, default=.5,
                        help='Minimum value for maximum fraction parameter for learnt parameter map.')
    parser.add_argument('--num_param_values', type=int, default=100,
                        help='Number of parameter values that are learnt for m, beta, and mf (maximum fraction). '
                             'The resulting number of learnt parameter combinations is num_param_values^3')
    parser.add_argument('--num_cpus', type=int, default=1,
                        help='Number of processes used for speeding up computations.')
    parser.add_argument('--hist_bins', type=int, default=100,
                        help='Number of bins that are created when computing histograms.')
    parser.add_argument('--verbosity', type=int, default=0,
                        help='Verbosity level.')
    parser.add_argument('--time_scale', type=int, default=140,
                        help='Time scale for which the repair dynamics are computed. Default is 140 minutes.')

    parsed_args = parser.parse_args(arguments)
    return parsed_args
