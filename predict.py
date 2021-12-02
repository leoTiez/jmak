#!/usr/bin/env python3
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.models import load_model

from src.DataLoader import load_chrom_data, load_chrom_split, load_bio_data, load_meres
from src.UtilsMain import create_models, argparse_predict
from src.Utils import validate_dir
from src.Model import BayesianParameterMap, NNParameterMap, ParameterMap


NUCLEOSOME_INDEX = {
    'nouv': 0,
    '0min': 1,
    '30min': 2
}

SLAM_INDEX = {
    'nouv': 0,
    '20min': 1,
    '120min': 2
}

NUCL_PATHS = [
    'data/seq/nucl_wt_nouv.bw',
    'data/seq/nucl_wt_0min.bw',
    'data/seq/nucl_wt_30min.bw'
]

SLAM_PATHS = [
    'data/seq/nouv_slam_mins_new.bw',
    'data/seq/nouv_slam_plus_new.bw',
    'data/seq/20m_slam_mins_new.bw',
    'data/seq/20m_slam_plus_new.bw',
    'data/seq/120m_slam_mins_new.bw',
    'data/seq/120m_slam_plus_new.bw'
]


def convert_bio_data(
        bio_type,
        bio_index,
        train_transcriptome,
        test_transcriptome,
        train_start_igr,
        train_end_igr,
        test_start_igr,
        test_end_igr
):
    train_trans, test_trans, train_igr, test_igr = None, None, None, None
    if bio_type.lower() == 'nucl' or bio_type == 'slam':
        bio_data_paths = NUCL_PATHS if bio_type.lower() == 'nucl' else SLAM_PATHS
        use_directionality = False if bio_type.lower() == 'nucl' else True
        idx = NUCLEOSOME_INDEX[bio_index] if bio_type.lower() == 'nucl' else SLAM_INDEX[bio_index]
        train_trans, test_trans = load_bio_data(
            zip(train_transcriptome['chr'].to_list(), train_transcriptome['start'].to_list()),
            zip(train_transcriptome['chr'].to_list(), train_transcriptome['end'].to_list()),
            zip(test_transcriptome['chr'].to_list(), test_transcriptome['start'].to_list()),
            zip(test_transcriptome['chr'].to_list(), test_transcriptome['end'].to_list()),
            bio_data_paths,
            use_directionality=use_directionality
        )
        train_igr, test_igr = load_bio_data(
            train_start_igr,
            train_end_igr,
            test_start_igr,
            test_end_igr,
            bio_data_paths,
            use_directionality=use_directionality
        )
        train_trans, test_trans = train_trans[idx], test_trans[idx]
        train_igr, test_igr = train_igr[idx], test_igr[idx]

    elif bio_type.lower() == 'size':
        train_trans = np.abs(train_transcriptome['start'].to_numpy('int') - train_transcriptome['end'].to_numpy('int'))
        test_trans = np.abs(test_transcriptome['start'].to_numpy('int') - test_transcriptome['end'].to_numpy('int'))

    elif bio_type.lower() == 'meres':
        c_pos = load_meres('centromeres')
        t_pos = load_meres('telomeres')
        train_trans_diff_cen_tel = []
        train_trans_chrom_compare = []
        test_trans_diff_cen_tel = []
        test_trans_chrom_compare = []
        for row in c_pos.iterrows():
            chrom = row[1]['chr']
            centromere = row[1]['pos']
            telomere = t_pos[t_pos['chr'] == chrom]['pos'].values[0]
            train_trans_chrom = train_transcriptome[train_transcriptome['chr'] == chrom]
            if len(train_trans_chrom) > 0:
                centre = ((train_trans_chrom['start'] + train_trans_chrom['end']) / 2.).to_numpy('float')
                train_trans_diff_cen_tel.extend(
                    np.minimum(np.abs(centre - centromere), np.abs(centre - telomere))
                )
                train_trans_chrom_compare.extend([chrom] * len(train_trans_chrom.index))

            test_trans_chrom = test_transcriptome[test_transcriptome['chr'] == chrom]
            if len(test_trans_chrom) > 0:
                centre = ((test_trans_chrom['start'] + test_trans_chrom['end']) / 2.).to_numpy('float')
                test_trans_diff_cen_tel.extend(
                    np.minimum(np.abs(centromere - centre), np.abs(telomere - centre))
                )
                test_trans_chrom_compare.extend([chrom] * len(test_trans_chrom.index))

        train_trans = np.asarray(train_trans_diff_cen_tel)
        test_trans = np.asarray(test_trans_diff_cen_tel)
        if not np.all([x == y for x, y in zip(train_transcriptome['chr'].to_list(), train_trans_chrom_compare)]):
            raise ValueError('Train chromosomes (transcript) did not match')
        if not np.all([x == y for x, y in zip(test_transcriptome['chr'].to_list(), test_trans_chrom_compare)]):
            raise ValueError('Test chromosomes (transcript) did not match')

        train_igr_diff_cen_tel = []
        train_igr_chrom_compare = []
        test_igr_diff_cen_tel = []
        test_igr_chrom_compare = []
        for row in c_pos.iterrows():
            chrom = row[1]['chr']
            centromere = row[1]['pos']
            telomere = t_pos[t_pos['chr'] == chrom]['pos'].values[0]

            chrom_list = list(filter(lambda x: x[0] == chrom, train_start_igr))
            if chrom_list:
                train_chrom_igr, train_chrom_igr_start = zip(*chrom_list)
                _, train_chrom_igr_end = zip(*filter(lambda x: x[0] == chrom, train_end_igr))
                centre = (np.asarray(train_chrom_igr_start, dtype='float')
                          + np.asarray(train_chrom_igr_end, dtype='float')) / 2.
                train_igr_diff_cen_tel.extend(
                    np.minimum(np.abs(centre - centromere), np.abs(centre - telomere))
                )
                train_igr_chrom_compare.extend(train_chrom_igr)

            chrom_list = list(filter(lambda x: x[0] == chrom, test_start_igr))
            if chrom_list:
                test_chrom_igr, test_chrom_igr_start = zip(*chrom_list)
                _, test_chrom_igr_end = zip(*filter(lambda x: x[0] == chrom, test_end_igr))
                centre = (np.asarray(test_chrom_igr_start, dtype='float')
                          + np.asarray(test_chrom_igr_end, dtype='float')) / 2.
                test_igr_diff_cen_tel.extend(
                    np.minimum(np.abs(centre - centromere), np.abs(centre - telomere))
                )
                test_igr_chrom_compare.extend(test_chrom_igr)

        chr_train, _ = zip(*train_start_igr)
        chr_test, _ = zip(*test_start_igr)
        if not np.all([x == y for x, y in zip(chr_train, train_igr_chrom_compare)]):
            raise ValueError('Train chromosomes (IGR) did not match')
        if not np.all([x == y for x, y in zip(chr_test, test_igr_chrom_compare)]):
            raise ValueError('Test chromosomes (IGR) did not match')
        train_igr = np.asarray(train_igr_diff_cen_tel)
        test_igr = np.asarray(test_igr_diff_cen_tel)
    else:
        raise ValueError('Unknown bio type.')

    return train_trans, test_trans, train_igr, test_igr


def main_nucl(args):
    bio_type = args.bio_type
    bio_index = args.bio_index
    ml_type = args.ml_type
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
    step_size = args.step_size
    num_epochs = args.num_epochs
    k_fold = args.k_fold
    plotted_dp = args.plotted_dp
    load_if_exist = args.load_if_exist
    rm_percentile = args.rm_percentile
    neg_random = args.neg_random
    discretise_bio = args.classify
    num_classes = args.num_classes
    layer_sizes = [256]

    if verbosity > 0:
        print('Load CPD')
    train_chrom = load_chrom_split('train')
    test_chrom = load_chrom_split('test')
    train_data, test_data = load_chrom_data(train_chrom_list=train_chrom, test_chrom_list=test_chrom)
    region_model_list = create_models(train_data, test_data, do_each=do_each)

    (_, _, _, train_start_igr, train_end_igr, train_transcriptome) = train_data
    (_, _, _, test_start_igr, test_end_igr, test_transcriptome) = test_data

    if verbosity > 0:
        print('Load bio data')
        print('Bio type: %s' % bio_type)

    train_trans, test_trans, train_igr, test_igr = convert_bio_data(
        bio_type,
        bio_index,
        train_transcriptome,
        test_transcriptome,
        train_start_igr,
        train_end_igr,
        test_start_igr,
        test_end_igr
    )

    if bio_type.lower() == 'size':
        region_model_list = list(filter(lambda x: 'gene' in x.name.lower() or 'nts' in x.name.lower(), region_model_list))
    train_data, test_data = [], []
    for rm in region_model_list:
        if 'gene' in rm.name.lower() or 'nts' in rm.name.lower():
            traind, testd = train_trans, test_trans
        else:
            traind, testd = train_igr, test_igr

        if 'test' in rm.name.lower():
            test_data.append((rm, testd))
        else:
            train_data.append((rm, traind))

    ml_models = []
    if verbosity > 0:
        print('Learn models')
    for rm, train_nucl in train_data:
        if verbosity > 0:
            print('Create parameter mapping for model %s' % rm.name)
        temp_m = np.asarray(list(rm.get_model_parameter('m', do_filter=False)))
        temp_beta = np.asarray(list(rm.get_model_parameter('beta', do_filter=False)))
        mask = ~np.isnan(temp_m)
        mask = np.logical_and(temp_m < 6., mask)
        mask = np.logical_and(temp_beta < .5, mask)

        rm.models = [model for num, model in enumerate(rm.models) if mask[num]]
        if load_if_exist:
            mfile_name = ''
            path = '%s/models' % os.getcwd()
            mfile_name = '%s/%s_%s_pickle_file_%s.pkl' % (path, save_prefix, rm.name, ml_type)
            if ml_type.lower() == 'nn':
                nnfile_name = '%s/%s_%s_keras_file_%s.h5' % (path, save_prefix, rm.name, ml_type)
            else:
                raise ValueError('ML type not understood.')

            if os.path.isfile(mfile_name):
                if verbosity > 0:
                    print('Found model file. Load model.')
                b_model = pickle.load(open(mfile_name, 'rb'))
                if ml_type.lower() == 'nn':
                    if os.path.isfile(nnfile_name):
                        b_model.nn = load_model(nnfile_name)
                        ml_models.append(b_model)
                        continue
                    else:
                        if verbosity > 0:
                            print('Could not find nn file. Create new ml object.')
                        b_model = None
                else:
                    ml_models.append(b_model)
                    continue
            else:
                if verbosity > 0:
                    print('Could not find existing ml file. Create new ml object.')
        if verbosity > 1:
            print('Init model')
        if ml_type.lower() == 'gp':
            b_model = BayesianParameterMap(
                rm,
                bio_data=train_nucl[mask],
                discretise_bio=discretise_bio,
                num_bins=num_classes,
                randomise=neg_random,
                kernel_func_type=kernel_func_type,
                kernel_scaling_init=kernel_scaling_init,
                kernel_search_verbosity=kernel_search_verbosity,
                noise_scaling_init=noise_scaling_init,
                kernel_search_step=step_size,
                min_m=min_m,
                max_m=max_m,
                min_mf=min_mf,
                min_beta=min_beta,
                max_beta=max_beta,
                num_param_values=num_param_values,
                rm_percentile=rm_percentile,
                num_cpus=num_cpus,
            )
        elif ml_type == 'nn':
            b_model = NNParameterMap(
                rm,
                bio_data=train_nucl[mask],
                discretise_bio=discretise_bio,
                num_bins=num_classes,
                layer_sizes=layer_sizes,
                randomise=neg_random,
                step_size=step_size,
                num_epocs=num_epochs,
                k_fold=k_fold,
                min_m=min_m,
                max_m=max_m,
                min_mf=min_mf,
                min_beta=min_beta,
                max_beta=max_beta,
                num_cpus=num_cpus,
                rm_percentile=rm_percentile,
                num_param_values=num_param_values
            )
        else:
            raise ValueError('ML type not understood.')

        if verbosity > 1:
            print('Train model')
        b_model.learn(
            num_cpus=num_cpus,
            verbosity=verbosity,
            hist_bins=hist_bins,
            save_fig=save_fig,
            save_prefix=save_prefix,
            estimate_time=time_scale
        )
        if to_pickle:
            if verbosity > 1:
                print('Save model to file')
            path = validate_dir('models')
            if ml_type.lower() == 'nn':
                # Save nn separately as it cannot be pickled.
                b_model.nn.save('%s/%s_%s_keras_file_%s.h5' % (path, save_prefix, rm.name, ml_type))
                b_model.nn = None

            pfile = open('%s/%s_%s_pickle_file_%s.pkl' % (path, save_prefix, rm.name, ml_type), 'wb')
            pickle.dump(b_model, file=pfile)
            pfile.close()
            if ml_type.lower() == 'nn':
                b_model.nn = load_model('%s/%s_%s_keras_file_%s.h5' % (path, save_prefix, rm.name, ml_type))

        if verbosity > 1:
            print('Plot parameter map')
        b_model.plot_parameter_map(
            verbosity=verbosity,
            save_fig=save_fig,
            save_prefix=save_prefix,
            plotted_dp=plotted_dp
        )
        ml_models.append(b_model)

    if verbosity > 0:
        print('Predict')

    for num, (ml, (rm, test_nucl)) in enumerate(zip(ml_models, test_data)):
        if verbosity > 1:
            print('Predict model %s' % rm.name)

        temp_m = np.asarray(list(rm.get_model_parameter('m', do_filter=False)))
        temp_beta = np.asarray(list(rm.get_model_parameter('beta', do_filter=False)))
        mask = ~np.isnan(temp_m)
        mask = np.logical_and(temp_m < 6., mask)
        mask = np.logical_and(temp_beta < .5, mask)

        prediction_list, est_m_list, est_max_frac_list, est_beta_list = ml.estimate(test_nucl[mask])

        fig, ax = plt.subplots(3, 1, figsize=(8, 7))
        ax[0].hist(
            est_m_list,
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
            est_max_frac_list,
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
            est_beta_list,
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
            plt.savefig('%s/%s_%s_parameter_distribution.png' % (directory, save_prefix, ml.rmodel.name))
            plt.close('all')
        else:
            plt.show()

        true_repair_list = rm.get_total_repair_fraction(time_scale=time_scale)[mask]

        pred_error_list = []
        inv_models = [im for num, im in enumerate(rm.models) if mask[num]]
        for true_repair, pred, jmak_model in zip(true_repair_list, prediction_list, inv_models):
            error = ParameterMap.error(
                np.arange(time_scale),
                real_data=true_repair,
                est_mean=pred,
                est_var=None
            )
            pred_error_list.append(error)
            if verbosity > 3:
                plt.figure(figsize=(8, 7))
                plt.plot(np.arange(time_scale), true_repair, color='cyan', label='True')
                plt.plot(np.arange(time_scale), pred, color='blue', label='Prediction')
                plt.scatter(
                    jmak_model.time_points.reshape(-1),
                    jmak_model.data_points.reshape(-1),
                    color='cyan',
                    label='Data'
                )
                plt.legend(loc='lower right')
                plt.xlabel('Time')
                plt.ylabel('Repair fraction')
                plt.title('Prediction vs data: %s\nError: %.2f' % (jmak_model.name, error))
                if save_fig:
                    directory = validate_dir('figures/predict_models/jmak')
                    plt.savefig('%s/%s_%s_%s_prediction.png' % (directory, save_prefix, ml.rmodel.name, jmak_model.name))
                    plt.close('all')
                else:
                    plt.show()

        if save_fig:
            directory = validate_dir('arrays')
            np.savetxt('%s/%s_%s.csv' % (directory, save_prefix, ml.rmodel.name), np.asarray(pred_error_list), delimiter=',')
        plt.figure(figsize=(8, 7))
        plt.hist(pred_error_list, bins='auto')
        plt.title('Validation error histogram: %s\nMean error: %.2f' % (ml.rmodel.name, np.mean(pred_error_list)))
        if save_fig:
            directory = validate_dir('figures/predict_models')
            plt.savefig('%s/%s_%s_prediction_err.png' % (directory, save_prefix, ml.rmodel.name))
            plt.close('all')
        else:
            plt.show()


if __name__ == '__main__':
    args = argparse_predict(sys.argv[1:])
    main_nucl(args)

