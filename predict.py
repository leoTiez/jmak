#!/usr/bin/env python3
import sys
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
import pickle

from src.DataLoader import load_chrom_data, load_chrom_split, load_bio_data, load_meres
from src.UtilsMain import create_models, argparse_predict
from src.Utils import validate_dir, frequency_bins
from src.Model import RegionModel
from src.PredictionModels import GPParameterMap, KNNParameterMap, LinearParameterMap


NUCLEOSOME_INDEX = {
    'nouv': 0,
    '0min': 1,
    '30min': 2
}

ABF1_INDEX = {
    'nouv': 0,
    'uv': 1
}

NUCL_PATHS = [
    'data/seq/nucl_wt_nouv.bw',
    'data/seq/nucl_wt_0min.bw',
    'data/seq/nucl_wt_30min.bw'
]

NETSEQ_PATHS = [
    'data/seq/wt_netseq_minus.bw',
    'data/seq/wt_netseq_plus.bw'
]

ABF1_PATHS = [
    'data/seq/wt_abf1_nouv.bw',
    'data/seq/wt_abf1_uv.bw'
]

H2AZ_PATHS = [
    'data/seq/wt_h2a_0m.bw'
]


def convert_bio_data(
        bio_type,
        bio_index,
        transcriptome,
        start_igr,
        end_igr
):
    trans_bio, igr_bio = None, None
    if bio_type.lower() in ['nucl', 'netseq', 'abf1', 'h2a']:
        if bio_type.lower() == 'nucl':
            bio_data_paths = NUCL_PATHS
            use_directionality = False
            idx = NUCLEOSOME_INDEX[bio_index]
        elif bio_type.lower() == 'netseq':
            bio_data_paths = NETSEQ_PATHS
            use_directionality = True
            idx = 0
        elif bio_type.lower() == 'abf1':
            bio_data_paths = ABF1_PATHS
            use_directionality = False
            idx = ABF1_INDEX[bio_index]
        else:
            bio_data_paths = H2AZ_PATHS
            use_directionality = False
            idx = 0
        trans_bio = load_bio_data(
            zip(transcriptome['chr'].to_list(), transcriptome['start'].to_list()),
            zip(transcriptome['chr'].to_list(), transcriptome['end'].to_list()),
            bio_data_paths,
            use_directionality=use_directionality
        )
        igr_bio = load_bio_data(
            start_igr,
            end_igr,
            bio_data_paths,
            use_directionality=use_directionality
        )
        trans_bio = trans_bio[idx]
        igr_bio = igr_bio[idx]

    elif bio_type.lower() == 'size':
        trans_bio = np.abs(transcriptome['start'].to_numpy('int') - transcriptome['end'].to_numpy('int'))

    elif 'meres' in bio_type.lower():
        split = bio_type.lower().split('_')
        do_relative = False
        if len(split) == 2 and split[0] == 'rel':
            do_relative = True
        c_pos = load_meres('centromeres')
        t_pos = load_meres('telomeres')
        trans_diff_cen_tel, trans_chrom_compare = [], []
        igr_diff_cen_tel, igr_chrom_compare = [], []
        for row in c_pos.iterrows():
            chrom = row[1]['chr']
            centromere = row[1]['pos']
            telomere = t_pos[t_pos['chr'] == chrom]['pos'].values
            if do_relative:
                denominator = np.maximum(np.abs(centromere - telomere[0]), np.abs(centromere - telomere[1])) / 2.

            trans_chrom = transcriptome[transcriptome['chr'] == chrom]
            if len(trans_chrom) > 0:
                centre = ((trans_chrom['start'] + trans_chrom['end']) / 2.).to_numpy('float')
                distance = np.minimum(
                    np.minimum(np.abs(centre - centromere), np.abs(centre - telomere[0])),
                    np.abs(centre - telomere[1])
                )
                if do_relative:
                    distance /= denominator
                trans_diff_cen_tel.extend(distance)
                trans_chrom_compare.extend([chrom] * len(trans_chrom.index))

            chrom_list = list(filter(lambda x: x[0] == chrom, start_igr))
            if chrom_list:
                train_chrom_igr, train_chrom_igr_start = zip(*chrom_list)
                _, train_chrom_igr_end = zip(*filter(lambda x: x[0] == chrom, end_igr))
                centre = (np.asarray(train_chrom_igr_start, dtype='float')
                          + np.asarray(train_chrom_igr_end, dtype='float')) / 2.
                distance = np.minimum(
                    np.minimum(np.abs(centre - centromere), np.abs(centre - telomere[0])),
                    np.abs(centre - telomere[1])
                )
                if do_relative:
                    distance /= denominator
                igr_diff_cen_tel.extend(distance)
                igr_chrom_compare.extend(train_chrom_igr)

        trans_bio = np.asarray(trans_diff_cen_tel)
        if not np.all([x == y for x, y in zip(transcriptome['chr'].to_list(), trans_chrom_compare)]):
            raise ValueError('Chromosomes (transcript) did not match')

        chr_igr, _ = zip(*start_igr)
        if not np.all([x == y for x, y in zip(chr_igr, igr_chrom_compare)]):
            raise ValueError('Chromosomes (IGR) did not match')

        igr_bio = np.asarray(igr_diff_cen_tel)
    else:
        raise ValueError('Unknown bio type.')

    return trans_bio, igr_bio


def main(args):
    bio_type = args.bio_type
    bio_index = args.bio_index
    ml_type = args.ml_type
    do_each = args.do_each
    save_fig = args.save_fig
    save_prefix = args.save_prefix
    to_pickle = args.to_pickle
    num_cpus = args.num_cpus
    verbosity = args.verbosity
    plotted_dp = args.plotted_dp
    load_if_exist = args.load_if_exist
    rm_percentile = args.rm_percentile
    neg_random = args.neg_random
    num_classes = args.num_classes
    k_neighbours = args.kneighbour
    no_tcr = args.no_tcr
    used_transcriptoms = [True, False, False] if not no_tcr else [True, True, True]
    num_bins = 3 if not no_tcr else 1
    test_ratio = .3

    if verbosity > 0:
        print('Load CPD')
    chrom_list = load_chrom_split()

    data = load_chrom_data(chrom_list=chrom_list, used_transcriptomes=used_transcriptoms, num_trans_bins=num_bins)
    region_model_list = create_models(data, do_each=do_each, no_tcr=no_tcr)

    (_, _, _, start_igr, end_igr, transcriptome) = data

    if verbosity > 0:
        print('Load bio data')
        print('Bio type: %s' % bio_type)

    trans_bio, igr_bio = convert_bio_data(
        bio_type,
        bio_index,
        transcriptome,
        start_igr,
        end_igr
    )

    if bio_type.lower() in ['netseq', 'size']:
        region_model_list = list(filter(lambda x: 'gene' in x.name.lower() or 'nts' in x.name.lower(), region_model_list))
    train_data, test_data = [], []
    for rm in region_model_list:
        mask = np.zeros(len(rm.models), dtype='bool')
        mask[np.random.choice(mask.size, size=np.round(mask.size * test_ratio).astype('int'), replace=False)] = True
        test_m = np.asarray(list(rm.get_model_parameter('m', do_filter=False)))[mask]
        test_mf = np.asarray(list(rm.get_model_parameter('max_frac', do_filter=False)))[mask]
        test_beta = np.asarray(list(rm.get_model_parameter('beta', do_filter=False)))[mask]
        rm.models = [jmak_model for num, jmak_model in enumerate(rm.models) if not mask[num]]
        test_params = np.asarray([test_m, test_mf, test_beta]).T
        if 'gene' in rm.name.lower() or 'nts' in rm.name.lower():
            traind, testd = trans_bio[~mask], trans_bio[mask]
        else:
            traind, testd = igr_bio[~mask], igr_bio[mask]

        train_data.append((rm, traind))
        test_data.append((test_params, testd))
    ml_models = []
    if verbosity > 0:
        print('Learn models')
        if neg_random:
            print('Randomise model')

    if verbosity > 4:
        if not no_tcr:
            if bio_type.lower() not in ['size', 'netseq']:
                if do_each:
                    fig, _ = plt.subplots(4, 2, figsize=(8, 7))
                    plt.subplot2grid((4, 2), (3, 0), colspan=2, fig=fig)
                else:
                    fig, _ = plt.subplots(2, 2, figsize=(8, 7))
                    plt.subplot2grid((2, 2), (1, 0), colspan=2, fig=fig)
                ax = fig.get_axes()
            else:
                if do_each:
                    fig, ax = plt.subplots(3, 2, figsize=(8, 7))
                else:
                    fig, ax = plt.subplots(1, 2, figsize=(8, 7))
                ax = ax.reshape(-1)
        else:
            if bio_type.lower() not in ['size', 'netseq']:
                fig, ax = plt.subplots(2, 2, figsize=(8, 7))
                ax = ax.reshape(-1)
            else:
                fig, ax = plt.subplots(1, 2, figsize=(8, 7))
                ax = ax.reshape(-1)
        for (rm, traind), a in zip(train_data, ax):
            if bio_type.lower() not in ['netseq', 'nucl', 'abf1', 'h2a']:
                val, _, _ = a.hist(traind, bins=100, label=rm.name, density=True)
            else:
                val, _, _ = a.hist(traind, bins=100, range=(0, 15), label=rm.name, density=True)
            bin_borders = frequency_bins(traind, 2)
            a.vlines(bin_borders[1], ymin=0, ymax=np.max(val), colors='red')

            a.set_title(rm.name)
        title_suffix = '' if not no_tcr else ' | No TCR'
        if bio_type.lower() == 'netseq':
            name = 'NET-seq%s' % title_suffix
        elif bio_type.lower() == 'nucl':
            name = 'Nucleosome density%s' % title_suffix
        elif bio_type.lower() == 'abf1':
            name = 'Abf1%s' % title_suffix
        elif bio_type.lower() == 'h2a':
            name = 'H2A.Z%s' % title_suffix
        elif bio_type.lower() == 'size':
            name = 'Size%s' % title_suffix
        elif 'meres' in bio_type.lower():
            is_relative_prefix = '' if bio_type.lower() == 'meres' else 'Relative '
            name = '%smeres%s' % (is_relative_prefix, title_suffix)
        else:
            raise ValueError('Unknown bio_type')
        fig.suptitle(name)
        fig.tight_layout()

        if not save_prefix:
            directory = validate_dir('figures/meta_distributions')
            suffix = '' if not no_tcr else '_notcr'
            plt.savefig('%s/%s_%s_distribution%s.png' % (directory, save_prefix, bio_type.lower(), suffix))
            plt.close('all')
        else:
            plt.show()

    for rm, traind in train_data:
        if verbosity > 0:
            print('Create parameter mapping for model %s' % rm.name)
        temp_m = np.asarray(list(rm.get_model_parameter('m', do_filter=False)))
        temp_beta = np.asarray(list(rm.get_model_parameter('beta', do_filter=False)))
        mask = ~np.isnan(temp_m)
        mask = np.logical_and(temp_m < 6., mask)
        mask = np.logical_and(np.logical_and(temp_beta < .05, mask), temp_beta > 1. / 200.)

        rm.models = [model for num, model in enumerate(rm.models) if mask[num]]
        if load_if_exist:
            path = '%s/models' % os.getcwd()
            mfile_name = '%s/%s_%s_pickle_file_%s.pkl' % (path, save_prefix, rm.name, ml_type)
            if os.path.isfile(mfile_name):
                if verbosity > 0:
                    print('Found model file. Load model.')
                try:
                    pfile = open(mfile_name, 'rb')
                    b_model = pickle.load(pfile)
                    ml_models.append(b_model)
                    pfile.close()
                    continue
                except:
                    if verbosity > 0:
                        print('Could not find existing ml file. Create new ml object.')
            else:
                if verbosity > 0:
                    print('Could not find existing ml file. Create new ml object.')
        if verbosity > 1:
            print('Init model')
        if ml_type.lower() == 'gp':
            b_model = GPParameterMap(
                rm,
                bio_data=traind[mask],
                num_bins=num_classes,
                randomise=neg_random,
                rm_percentile=rm_percentile
            )

        elif ml_type == 'lin':
            b_model = LinearParameterMap(
                rm,
                bio_data=traind[mask],
                num_bins=num_classes,
                randomise=neg_random,
                rm_percentile=rm_percentile,
            )
        elif ml_type.lower() == 'knn':
            b_model = KNNParameterMap(
                rm,
                traind[mask],
                k_neighbours=k_neighbours,
                randomise=neg_random,
                num_cpus=num_cpus,
                num_bins=num_classes
            )
        else:
            raise ValueError('ML type not understood.')

        if verbosity > 1:
            print('Train model')
        b_model.learn(verbosity=verbosity)
        if to_pickle:
            if verbosity > 1:
                print('Save model to file')
            path = validate_dir('models')
            pfile = open('%s/%s_%s_pickle_file_%s.pkl' % (path, save_prefix, rm.name, ml_type), 'wb')
            pickle.dump(b_model, file=pfile)
            pfile.close()

        if verbosity > 1:
            print('Plot parameter map')
        b_model.plot_parameter_map(
            verbosity=verbosity,
            save_fig=save_fig,
            save_prefix=save_prefix,
            plotted_dp=plotted_dp,
            levels=num_classes if ml_type.lower() in ['knn', 'lin'] else 15
        )

        mean, var = b_model.estimate(b_model.data_params, convert_dp=False)
        b_model.plot_error(
            mean,
            b_model.bio_data,
            var,
            b_model.data_params,
            convert_params=False,
            convert_val=False,
            save_fig=save_fig,
            save_prefix='%s_train' % save_prefix
        )
        ml_models.append(b_model)

    if verbosity > 0:
        print('Predict')

    for num, (ml, (input_param, testd)) in enumerate(zip(ml_models, test_data)):
        if verbosity > 1:
            print('Predict model %s' % ml.rmodel.name)
        input_m, input_mf, input_beta = input_param[:, 0], input_param[:, 1], input_param[:, 2]
        mask = ~np.isnan(input_m)
        mask = np.logical_and(input_m < 6., mask)
        mask = np.logical_and(input_beta < .05, mask)
        params = np.asarray([input_m, input_mf, input_beta]).T[mask]
        mean_pred, var_pred = ml.estimate(params)
        ml.plot_error(
            mean_pred,
            testd[mask],
            var_pred,
            params,
            convert_val=True if ml_type.lower() in ['knn', 'lin'] else False,
            save_fig=save_fig,
            save_prefix='%s_test' % save_prefix
        )


if __name__ == '__main__':
    args = argparse_predict(sys.argv[1:])
    main(args)

