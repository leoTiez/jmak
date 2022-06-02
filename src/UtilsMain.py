import argparse

import numpy as np
import matplotlib.pyplot as plt

from src.Model import RegionModel
from src.DataLoader import create_time_data
from src.Utils import validate_dir


def create_models(
        data,
        do_each=True,
        no_tcr=False,
        verbosity=0,
        save_fig=True,
        save_prefix='',
        plot_derivative=False,
        time_scale=140
):
    (trans, ntrans, igr, start_igr, _, transcriptome) = data
    repair_prediction = []
    repair_derivative = []
    if do_each and not no_tcr:
        iter = zip(
            [
                trans[:, :, 0],
                trans[:, :, 1],
                trans[:, :, 2],
                ntrans[:, :, 0],
                ntrans[:, :, 1],
                ntrans[:, :, 2],
                igr
            ],
            [
                'TCR region start (TS)',
                'TCR region centre (TS)',
                'TCR region end (TS)',
                'TCR region start (NTS)',
                'TCR region centre (NTS)',
                'TCR region end (NTS)',
                'Non-TCR region'
            ],
            [
                'data/jmak/trans_start.csv',
                'data/jmak/trans_centre.csv',
                'data/jmak/trans_end.csv',
                'data/jmak/ntrans_start.csv',
                'data/jmak/ntrans_centre.csv',
                'data/jmak/ntrans_end.csv',
                'data/jmak/igr.csv'
            ]
        )
    elif not do_each and not no_tcr:
        iter = zip(
            [
                trans,
                ntrans,
                igr
            ],
            [
                'Genes total',
                'NTS total',
                'Intergenic regions'
            ],
            [
                'data/jmak/trans_total.csv',
                'data/jmak/ntrans_total.csv',
                'data/jmak/igr.csv'
            ]
        )
    else:
        iter = zip(
            [
                trans,
                ntrans,
                igr[:, :, 0],
                igr[:, :, 1]
            ],
            [
                'All TS',
                'All NTS',
                'IGR Strand +',
                'IGR Strand -',
            ],
            [
                'data/jmak/all_trans.csv',
                'data/jmak/all_ntrans.csv',
                'data/jmak/igr_plus.csv',
                'data/jmak/igr_minus.csv'
            ]
        )
    model_list = []
    for data, name, file_name in iter:
        if 'trans' in file_name.lower():
            chrom_list = transcriptome['chr'].to_list()
        else:
            chrom_list = map(lambda x: x[0], start_igr)
        if not do_each and 'trans' in file_name and not no_tcr:
            num_pos = 3
        elif 'igr' in file_name and not no_tcr:
            num_pos = 2
        else:
            num_pos = 1
        region_model = RegionModel(
            create_time_data(num_pos, len(data)),
            data.reshape(len(data), -1),
            name=name
        )
        region_model.load_models(file_name, compare_chrom_list=chrom_list)
        if verbosity > 4:
            if 'genes' in name.lower():
                if no_tcr:
                    example_genes = filter(lambda x: x.name in ['chrI YAL048C', 'chrVIII YHL025W'], region_model.models)
                    for gene in example_genes:
                        plt.figure(figsize=(8, 7))
                        plt.scatter(gene.time_points, gene.data_points, color='blue', label='Data')
                        plt.plot(np.arange(140), gene.repair_fraction_over_time(140), 'b--', label='Estimation')
                        if plot_derivative:
                            plt.plot(
                                np.arange(140),
                                gene.repair_derivative_over_time(140),
                                color='orange',
                                linestyle='--',
                                label='Derivative'
                            )
                        plt.legend(loc='center right', fontsize=20)
                        plt.title('Repair evolution: %s' % gene.name, fontsize='30')
                        plt.xticks(fontsize='16')
                        plt.yticks(fontsize='16')
                        plt.xlabel('Time (min)', fontsize='20')
                        plt.ylabel('Repair fraction', fontsize='20')
                        if save_fig:
                            directory = validate_dir('figures/examples')
                            plt.savefig('%s/%s_repair_evolution_%s.png' % (directory, save_prefix, gene.name))
                            plt.close('all')
                        else:
                            plt.show()
                else:
                    print('Example plot for KJMA model only in no_tcr setup.')
        if verbosity > 4:
            repair_prediction = region_model.get_total_repair_fraction(time_scale)
            repair_derivative = region_model.get_total_repair_derivative(time_scale)
            x_time = np.arange(time_scale)
            fig, ax1 = plt.subplots(figsize=(8, 7))
            ax1.plot(x_time, np.nanmean(repair_prediction, axis=0), 'b--', label=r'$f(t)$')
            ax1.fill_between(
                x_time,
                np.nanmean(repair_prediction, axis=0) - np.nanstd(repair_prediction, axis=0),
                np.nanmean(repair_prediction, axis=0) + np.nanstd(repair_prediction, axis=0),
                alpha=.2,
                color='blue'
            )

            ax2 = ax1.twinx()
            ax2.plot(
                x_time,
                np.nanmean(repair_derivative, axis=0),
                linestyle='--',
                color='orange',
                label=r'10 x $df(t)$'
            )
            ax2.fill_between(
                x_time,
                np.nanmean(repair_derivative, axis=0) - np.nanstd(repair_derivative, axis=0),
                np.nanmean(repair_derivative, axis=0) + np.nanstd(repair_derivative, axis=0),
                alpha=.2,
                color='orange'
            )
            # ax1.legend(loc='center right', fontsize=24)
            ax1.tick_params(axis='x', labelsize=20)
            ax1.tick_params(axis='y', labelsize=20)
            ax2.tick_params(axis='y', labelsize=20)
            ax1.set_xlabel('Time (min)', fontsize=24)
            ax1.set_ylabel(r'$f(t)$', color='blue', fontsize=24)
            ax2.set_ylabel(r'$df(t)$', color='orange', fontsize=24)

            fig.suptitle('Repair in %s' % region_model.name, fontsize=32)
            fig.tight_layout()

            if save_fig:
                directory = validate_dir('figures/examples')
                plt.savefig('%s/%s_region_repair_evolution_%s.png' % (directory, save_prefix, region_model.name))
                plt.close('all')
            else:
                plt.show()

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
    parser.add_argument('--no_tcr', action='store_true', dest='no_tcr',
                        help='If set, programme does not distinguish between TCR and rest.')

    parsed_args = parser.parse_args(arguments)
    return parsed_args


def argparse_model_comp_param(arguments):
    parser = argparse.ArgumentParser(
        description='Compare different model performances.'
    )
    parser.add_argument('--do_each', action='store_true', dest='do_each',
                        help='If set, there is one model per region in each transcript. '
                             'Otherwise beginning, centre of a gene and end are combined in one single model.')
    parser.add_argument('--verbosity', type=int, default=2,
                        help='Verbosity flag')
    parser.add_argument('--min_m', type=float, default=-1.,
                        help='Filter for KJMA models with minimum m. Default -1 (no filtering).')
    parser.add_argument('--max_m', type=float, default=-1,
                        help='Filter for KJMA models with maximum m. Default -1 (no filtering).')
    parser.add_argument('--min_beta', type=float, default=-1.,
                        help='Filter for KJMA models with minimum beta. Default -1 (no filtering).')
    parser.add_argument('--max_beta', type=float, default=-1.,
                        help='Filter for KJMA models with maximum beta. Default -1 (no filtering).')
    parser.add_argument('--save_fig', action='store_true', dest='save_fig',
                        help='If set, figures are saved instead of displayed.')
    parser.add_argument('--test', action='store_true', dest='test',
                        help='Run test function for debugging.')
    parser.add_argument('--no_tcr', action='store_true', dest='no_tcr',
                        help='If set, programme does not distinguish between TCR and rest.')
    parser.add_argument('--save_prefix', type=str, default='',
                        help='Identifier for saved files.')

    parsed_args = parser.parse_args(arguments)
    return parsed_args


def argparse_bioplotter(arguments):
    parser = argparse.ArgumentParser(
        description='Plot available biological data wrt the parameter space. Train the JMAK model and save'
                    'them to a file first.'
    )

    parser.add_argument('--bio_type', type=str, required=True,
                        help='Pass the data type that is used in order to determine the colour gradient.'
                             'Possible are: netseq | nucl | abf1 | h2a | size | meres'
                        )
    parser.add_argument('--use_sum', action='store_true', dest='use_sum',
                        help='If set, when the biological data is loaded, the sum is taken instead of the mean.'
                             'Only important for sequencing data, ie netseq, nucl, abf1, and h2a')
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
    parser.add_argument('--no_tcr', action='store_true', dest='no_tcr',
                        help='If set, programme does not distinguish between TCR and rest.')
    parser.add_argument('--g_bins', type=int, default=50,
                        help='Number of bins for the colour gradient.')

    parsed_args = parser.parse_args(arguments)
    return parsed_args


def argparse_predict(arguments):
    parser = argparse.ArgumentParser(
        description='Learn function to approximate JMAK parameters given biological data. Train the JMAK model and save'
                    'them to a file first.'
    )

    parser.add_argument('--bio_type', type=str, required=True,
                        help='Pass the data type that is used in order to determine the function '
                             'and to make predictions. Possible are: netseq | nucl | abf1 | h2a | size | meres')
    parser.add_argument('--bio_index', type=str, default='',
                        help='Several time points are available for ABF1 and NUCL. Pass the time identifier'
                             'with this key word. '
                             'Possible are for NUCL: nouv | 0min | 30min .'
                             'Possible are for ABF1: nouv | uv . '
                        )
    parser.add_argument('--ml_type', type=str, required=True,
                        help='Define the applied machine learning approach which is used to find the parameter map. '
                             'Possible are: knn | gp | lin')
    parser.add_argument('--neg_random', action='store_true', dest='neg_random',
                        help='If set, the biological feature data is randomly shuffled to create a negative control.')
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
    parser.add_argument('--num_cpus', type=int, default=1,
                        help='Number of processes used for speeding up computations.')
    parser.add_argument('--hist_bins', type=int, default=100,
                        help='Number of bins that are created when computing histograms.')
    parser.add_argument('--plotted_dp', type=int, default=500,
                        help='Numper of points plotted in the parameter map plot.')
    parser.add_argument('--load_if_exist', action='store_true', dest='load_if_exist',
                        help='If set, the ml model is loaded from a file '
                             'if it is exists instead of creating a new model')
    parser.add_argument('--rm_percentile', type=float, default=5.,
                        help='The amount of outliers that is removed from the parameter data. Ignore all values'
                             'lower than rm_percentile / 2. and larger than 100 - rm_percentile / 2.')
    parser.add_argument('--verbosity', type=int, default=0,
                        help='Verbosity level.')
    parser.add_argument('--time_scale', type=int, default=140,
                        help='Time scale for which the repair dynamics are computed. Default is 140 minutes.')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Determines the number of created classes if --classify flag is set.')
    parser.add_argument('--kneighbour', type=int, default=10,
                        help='Number of neighbour in kNN machine learning model')
    parser.add_argument('--no_tcr', action='store_true', dest='no_tcr',
                        help='If set, programme does not distinguish between TCR and rest.')
    parser.add_argument('--min_m', type=float, default=.5,
                        help='Minimum m value that is used for finding and testing correlation.')
    parser.add_argument('--max_m', type=float, default=6.,
                        help='Maximum m value that is used for finding and testing correlation.')
    parser.add_argument('--min_beta', type=float, default=1./200.,
                        help='Minimum beta value that is used for finding and testing correlation.')
    parser.add_argument('--max_beta', type=float, default=.05,
                        help='Maximum beta value that is used for finding and testing correlation.')
    parser.add_argument('--test_ratio', type=float, default=.3,
                        help='Ratio of the test data vs training data')
    parser.add_argument('--use_mode', action='store_true', dest='use_mode',
                        help='If true and classification is set, use mode instead of frequency bins to determine '
                             'high and low class')
    parsed_args = parser.parse_args(arguments)
    return parsed_args


def argparse_errorplotter(arguments):
    parser = argparse.ArgumentParser(
        description='Load array files from different experiments and plot error in violin plots.'
    )

    parser.add_argument('--save_prefix', type=str, required=True,
                        help='Save prefix that was used when array files where created. '
                             'Used for string based matching.')
    parser.add_argument('--array_dir', type=str, default='arrays',
                        help='Directory where arrays are stored.')
    parser.add_argument('--pthresh', type=float, default=.00001,
                        help='Significance threshold. Should be set between 0 and 1.')
    parser.add_argument('--max_iter', type=int, default=100,
                        help='Maximum number of experiments that are included.')
    parser.add_argument('--save_fig', action='store_true', dest='save_fig',
                        help='If set, created plot is saved to file instead of displayed.')

    parsed_args = parser.parse_args(arguments)
    return parsed_args


def argparse_derivative(arguments):
    parser = argparse.ArgumentParser(
        description='Find the derivative and compare to XR'
    )
    parser.add_argument('--do_each', action='store_true', dest='do_each',
                        help='If set, there is one model per region in each transcript. '
                             'Otherwise beginning, centre of a gene and end are combined in one single model.')
    parser.add_argument('--save_prefix', type=str, default='',
                        help='Set identifying string that goes in front of every saved file name.')
    parser.add_argument('--save_fig', action='store_true', dest='save_fig',
                        help='If set, figures are saved instead of displayed.')
    parser.add_argument('--num_cpus', type=int, default=1,
                        help='Number of processes used for speeding up computations.')
    parser.add_argument('--verbosity', type=int, default=0,
                        help='Verbosity level.')
    parser.add_argument('--time_scale', type=int, default=140,
                        help='Time scale for which the repair dynamics are computed. Default is 140 minutes.')
    parser.add_argument('--no_tcr', action='store_true', dest='no_tcr',
                        help='If set, programme does not distinguish between TCR and rest.')
    parser.add_argument('--min_m', type=float, default=.5,
                        help='Minimum m value that is used for finding and testing correlation.')
    parser.add_argument('--max_m', type=float, default=6.,
                        help='Maximum m value that is used for finding and testing correlation.')
    parser.add_argument('--min_beta', type=float, default=1. / 200.,
                        help='Minimum beta value that is used for finding and testing correlation.')
    parser.add_argument('--max_beta', type=float, default=.05,
                        help='Maximum beta value that is used for finding and testing correlation.')
    parser.add_argument('--plt_dc', action='store_true', dest='plt_dc',
                        help='If set, distance correlation is printed in the plot. Otherwise it is given as an'
                             'output in the console.'
                        )
    parser.add_argument('--per_time', action='store_true', dest='per_time',
                        help='If set, differences between time points which are assumed to represent repair rate '
                             'are calculated per time step'
                        )

    parsed_args = parser.parse_args(arguments)
    return parsed_args


def argparse_kjmapredict_param(arguments):
    parser = argparse.ArgumentParser(
        description='Make KJMA repair predictions for given area'
    )
    parser.add_argument('--chrom', type=str, required=True,
                        help='Chromosome of the region of interest')
    parser.add_argument('--start', type=int, required=True,
                        help='Start coordinate of the region of interest')
    parser.add_argument('--end', type=int, required=True,
                        help='End coordinate of the region of interest')
    parser.add_argument('-t', type=float, nargs='*', default=[45],
                        help='List of prediction time points')
    parser.add_argument('--min_f', type=float, default=.5,
                        help='Minimum allowed value for maximum fraction '
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
    parser.add_argument('--save_prefix', type=str, default='',
                        help='Save prefix for files.')
    parsed_args = parser.parse_args(arguments)
    return parsed_args
