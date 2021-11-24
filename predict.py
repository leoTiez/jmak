#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle

from src.DataLoader import load_chrom_data, load_chrom_split, load_bio_data
from src.UtilsMain import create_models, argparse_predict
from src.Utils import validate_dir
from src.Model import BayesianParameterMap

NUCLEOSOME_INDEX = {
    'nouv': 0,
    '0min': 1,
    '30min': 2
}


def main_nucl(args):
    do_each = args.do_each
    save_fig = args.save_fig
    save_prefix = args.save_prefix
    to_pickle = args.to_pickle
    kernel_func_type = args.kernel_func_type
    kernel_search_verbosity = args.kernel_search_verbosity
    kernel_scaling_init = args.kernel_scaling_init
    noise_scaling_init = args.noise_scaling_init
    min_m = args.min_m
    max_m = args.max_m
    min_mf = args.min_mf
    min_beta = args.min_beta
    max_beta = args.max_beta
    num_param_values = args.num_param_values
    num_cpus = args.num_cpus
    hist_bins = args.hist_bins
    verbosity = args.verbosity
    time_scale = args.time_scale

    nucl_index = NUCLEOSOME_INDEX['0min']

    nucl_paths = [
        'data/seq/nucl_wt_nouv.bw',
        'data/seq/nucl_wt_0min.bw',
        'data/seq/nucl_wt_30min.bw'
    ]

    print('Load CPD')
    train_chrom = load_chrom_split('train')
    test_chrom = load_chrom_split('test')
    train_data, test_data = load_chrom_data(train_chrom_list=train_chrom, test_chrom_list=test_chrom)
    region_model_list = create_models(train_data, test_data, do_each=do_each)

    (_, _, _, train_start_igr, train_end_igr, train_transcriptome) = train_data
    (_, _, _, test_start_igr, test_end_igr, test_transcriptome) = test_data

    train_nucl_trans, test_nucl_trans = load_bio_data(
        zip(train_transcriptome['chr'].to_list(), train_transcriptome['start'].to_list()),
        zip(train_transcriptome['chr'].to_list(), train_transcriptome['end'].to_list()),
        zip(test_transcriptome['chr'].to_list(), test_transcriptome['start'].to_list()),
        zip(test_transcriptome['chr'].to_list(), test_transcriptome['end'].to_list()),
        nucl_paths,
        use_directionality=False
    )
    train_nucl_igr, test_nucl_igr = load_bio_data(
        train_start_igr,
        train_end_igr,
        test_start_igr,
        test_end_igr,
        nucl_paths,
        use_directionality=False
    )

    train_data, test_data = [], []
    for rm in region_model_list:
        if 'gene' in rm.name.lower():
            train_nucl, test_nucl = train_nucl_trans[nucl_index], test_nucl_trans[nucl_index]
        else:
            train_nucl, test_nucl = train_nucl_igr[nucl_index], test_nucl_igr[nucl_index]
        if 'test' in rm.name.lower():
            test_data.append((rm, test_nucl))
        else:
            train_data.append((rm, train_nucl))

    gp_models = []
    for rm, train_nucl in train_data:
        temp_m = np.asarray(list(rm.get_model_parameter('m', do_filter=False)))
        temp_beta = np.asarray(list(rm.get_model_parameter('beta', do_filter=False)))
        mask = ~np.isnan(temp_m)
        mask = np.logical_and(temp_m < 6., mask)
        mask = np.logical_and(temp_beta < .5, mask)
        rm.models = [model for num, model in enumerate(rm.models) if mask[num]]
        b_model = BayesianParameterMap(
            rm,
            bio_data=train_nucl[mask],
            kernel_func_type=kernel_func_type,
            kernel_scaling_init=kernel_scaling_init,
            kernel_search_verbosity=kernel_search_verbosity,
            noise_scaling_init=noise_scaling_init,
            min_m=min_m,
            max_m=max_m,
            min_mf=min_mf,
            min_beta=min_beta,
            max_beta=max_beta,
            num_param_values=num_param_values
        )

        b_model.learn(
            num_cpus=num_cpus,
            verbosity=verbosity,
            hist_bins=hist_bins,
            save_fig=save_fig,
            save_prefix=save_prefix,
            estimate_time=time_scale
        )
        if to_pickle:
            path = validate_dir('models')
            pfile = open('%s/%s_%s_pickle_file.pkl' % (path, save_prefix, rm.name), 'wb')
            pickle.dump(b_model, file=pfile)
            pfile.close()
        b_model.plot_parameter_map(verbosity=verbosity, save_fig=save_fig, save_prefix=save_prefix)
        gp_models.append(b_model)

    for num, (gp, (rm, test_nucl)) in enumerate(zip(gp_models, test_data)):
        temp_m = np.asarray(list(rm.get_model_parameter('m', do_filter=False)))
        temp_beta = np.asarray(list(rm.get_model_parameter('beta', do_filter=False)))
        mask = ~np.isnan(temp_m)
        mask = np.logical_and(temp_m < 6., mask)
        mask = np.logical_and(temp_beta < .5, mask)

        prediction_list, est_m_list, est_max_frac_list, est_beta_list = gp.estimate(test_nucl[mask])
        true_repair_list = rm.get_total_repair_fraction(time_scale=time_scale)[mask]

        pred_error_list = []
        for true_repair, pred, jmak_model in zip(true_repair_list, prediction_list, rm.models):
            error = BayesianParameterMap.error(
                np.arange(time_scale),
                real_data=true_repair,
                est_mean=np.mean(pred, axis=0),
                est_var=np.var(pred, axis=0)
            )
            pred_error_list.append(error)
            if verbosity > 1:
                plt.figure(figsize=(8, 7))
                plt.plot(np.arange(time_scale), true_repair, color='red', label='True')
                plt.plot(np.arange(time_scale), np.mean(pred, axis=0), color='blue', label='Prediction')
                plt.fill_between(
                    np.arange(time_scale),
                    np.mean(pred, axis=0) - np.var(pred, axis=0),
                    np.mean(pred, axis=0) + np.var(pred, axis=0),
                    alpha=.2,
                    color='blue'
                )
                plt.scatter(
                    jmak_model.time_points.reshape(-1),
                    jmak_model.data_points.reshape(-1),
                    color='blue',
                    label='Data'
                )
                plt.legend(loc='lower right')
                plt.xlabel('Time')
                plt.ylabel('Repair fraction')
                plt.title('Prediction vs data: %s\nError: %.2f' % (jmak_model.name, error))
                if save_fig:
                    directory = validate_dir('figures/predict_models/jmak')
                    plt.savefig('%s/%s_%s_prediction.png' % (directory, save_prefix, jmak_model.name))
                    plt.close('all')
                else:
                    plt.show()

        plt.figure(figsize=(8, 7))
        plt.hist(pred_error_list, bins=hist_bins)
        plt.title('Validation error histogram: %s' % gp.rmodel.name)
        if save_fig:
            directory = validate_dir('figures/predict_models')
            plt.savefig('%s/%s_%s_prediction_err.png' % (directory, save_prefix, gp.rmodel.name))
            plt.close('all')
        else:
            plt.show()

        fig, ax = plt.subplots(3, 1, figsize=(8, 7))
        ax[0].hist(
            [k for x in est_m_list for k in x],
            bins='auto',
            histtype='step',
            color='magenta',
            density=True,
            label='Prediction'
        )
        ax[0].hist(
            np.asarray(list(rm.get_model_parameter('m', do_filter=False)))[mask],
            bins='auto',
            histtype='step',
            color='red',
            density=True,
            label='True'
        )
        ax[0].legend(loc='upper right')
        ax[0].set_ylabel('Density')
        ax[0].set_title('m')

        ax[1].hist(
            [k for x in est_max_frac_list for k in x],
            bins='auto',
            histtype='step',
            color='deepskyblue',
            density=True,
            label='Prediction'
        )
        ax[1].hist(
            np.asarray(list(rm.get_model_parameter('max_frac', do_filter=False)))[mask],
            bins='auto',
            histtype='step',
            color='red',
            density=True,
            label='True'
        )
        ax[1].legend(loc='upper right')
        ax[1].set_ylabel('Density')
        ax[1].set_title('Maximum fraction')

        ax[2].hist(
            [k for x in est_beta_list for k in x],
            bins='auto',
            histtype='step',
            color='lime',
            density=True,
            label='Prediction'
        )
        ax[2].hist(
            np.asarray(list(rm.get_model_parameter('beta', do_filter=False)))[mask],
            bins='auto',
            histtype='step',
            color='red',
            density=True,
            label='True'
        )
        ax[2].legend(loc='upper right')
        ax[2].set_ylabel('Density')
        ax[2].set_title(r'$\beta$')
        ax[2].set_xlabel('Value')

        fig.suptitle('Parameter distributions: %s' % rm.name)
        fig.tight_layout()
        if save_fig:
            directory = validate_dir('figures/predict_models')
            plt.savefig('%s/%s_%s_parameter_distribution.png' % (directory, save_prefix, gp.rmodel.name))
            plt.close('all')
        else:
            plt.show()


if __name__ == '__main__':
    args = argparse_predict(sys.argv[1:])
    main_nucl(args)

