#!/usr/bin/env python3
import sys
import numpy as np

from src.DataLoader import load_chrom_data, load_chrom_split, load_bio_data, create_time_data, load_meres
from src.Model import RegionModel
from src.UtilsMain import argparse_bioplotter, create_models


def main(args):
    do_each = args.do_each
    save_fig = args.save_fig
    size_scaling = args.size_scaling
    size_power = args.size_power
    power_norm = args.power_norm
    bio_type = args.bio_type
    save_prefix = args.save_prefix
    use_sum = args.use_sum

    slam_paths = [
        'data/seq/nouv_slam_mins_new.bw',
        'data/seq/nouv_slam_plus_new.bw',
        'data/seq/20m_slam_mins_new.bw',
        'data/seq/20m_slam_plus_new.bw',
        'data/seq/120m_slam_mins_new.bw',
        'data/seq/120m_slam_plus_new.bw'
    ]

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

    trans_rmodels = list(filter(lambda x: 'gene' in x.name.lower() or 'nts' in x.name.lower(), region_model_list))
    igr_rmodels = list(filter(lambda x: 'intergenic' in x.name.lower(), region_model_list))

    if bio_type.lower() == 'slam':
        # SLAM-seq data
        print('Load SLAM-seq')
        train_slam, test_slam = load_bio_data(
            zip(train_transcriptome['chr'].to_list(), train_transcriptome['start'].to_list()),
            zip(train_transcriptome['chr'].to_list(), train_transcriptome['end'].to_list()),
            zip(test_transcriptome['chr'].to_list(), test_transcriptome['start'].to_list()),
            zip(test_transcriptome['chr'].to_list(), test_transcriptome['end'].to_list()),
            slam_paths,
            use_directionality=True,
            use_sum=use_sum
        )

        for tm in trans_rmodels:
            slam_data = train_slam if 'train' in tm.name.lower() else test_slam
            for sd, n in zip(slam_data, ['nouv', '20m', '120m']):
                for mr in [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25)]:
                    tm.plot_parameter_with_gradient(
                        sd,
                        cmap='PiYG',
                        alpha=.7,
                        m_range=mr,
                        size_scaling=size_scaling,
                        size_power=size_power,
                        power_norm=power_norm,
                        num_handles=8,
                        save_fig=save_fig,
                        save_prefix='%s_slam_%s_%s_%s' % (save_prefix, n, mr[0], mr[1])
                    )

    elif bio_type.lower() == 'nucl':
        # Nucleosome data
        print('Load MNase-seq')
        train_nucl_trans, test_nucl_trans = load_bio_data(
            zip(train_transcriptome['chr'].to_list(), train_transcriptome['start'].to_list()),
            zip(train_transcriptome['chr'].to_list(), train_transcriptome['end'].to_list()),
            zip(test_transcriptome['chr'].to_list(), test_transcriptome['start'].to_list()),
            zip(test_transcriptome['chr'].to_list(), test_transcriptome['end'].to_list()),
            nucl_paths,
            use_directionality=False,
            use_sum=use_sum
        )
        train_nucl_igr, test_nucl_igr = load_bio_data(
            train_start_igr,
            train_end_igr,
            test_start_igr,
            test_end_igr,
            nucl_paths,
            use_directionality=False,
            use_sum=use_sum
        )

        for tm in trans_rmodels:
            nucl_data = train_nucl_trans if 'train' in tm.name.lower() else test_nucl_trans
            for sd, n in zip(nucl_data, ['nouv', '0m', '30m']):
                for mr in [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25)]:
                    tm.plot_parameter_with_gradient(
                        sd,
                        cmap='BrBG',
                        alpha=.7,
                        m_range=mr,
                        size_scaling=size_scaling,
                        size_power=size_power,
                        power_norm=power_norm,
                        save_fig=save_fig,
                        save_prefix='%s_nucl_%s_%s_%s' % (save_prefix, n, mr[0], mr[1])
                    )

        for tm in igr_rmodels:
            nucl_data = train_nucl_igr if 'train' in tm.name.lower() else test_nucl_igr
            for sd, n in zip(nucl_data, ['nouv', '0m', '30m']):
                for mr in [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25)]:
                    tm.plot_parameter_with_gradient(
                        sd,
                        cmap='BrBG',
                        alpha=.7,
                        m_range=mr,
                        size_scaling=size_scaling,
                        size_power=size_power,
                        power_norm=power_norm,
                        save_fig=save_fig,
                        save_prefix='%s_nucl_%s_%s_%s' % (save_prefix, n, mr[0], mr[1])
                    )

    elif bio_type.lower() == 'size':
        # size
        print('Size')
        train_size = np.abs(train_transcriptome['start'].to_numpy('int') - train_transcriptome['end'].to_numpy('int'))
        test_size = np.abs(test_transcriptome['start'].to_numpy('int') - test_transcriptome['end'].to_numpy('int'))
        for tm in trans_rmodels:
            size_data = train_size if 'train' in tm.name.lower() else test_size
            for mr in [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25)]:
                tm.plot_parameter_with_gradient(
                    size_data,
                    cmap='Spectral',
                    alpha=.7,
                    m_range=mr,
                    size_scaling=size_scaling,
                    size_power=size_power,
                    power_norm=power_norm,
                    save_fig=save_fig,
                    save_prefix='%s_size_%s_%s' % (save_prefix, mr[0], mr[1])
                )

    elif bio_type.lower() == 'meres':
        # Telomeres and centromeres
        print('Telomeres and centromeres')
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

        train_trans_diff_cen_tel = np.asarray(train_trans_diff_cen_tel)
        test_trans_diff_cen_tel = np.asarray(test_trans_diff_cen_tel)
        if not np.all([x == y for x, y in zip(train_transcriptome['chr'].to_list(), train_trans_chrom_compare)]):
            raise ValueError('Train chromosomes (transcript) did not match')
        if not np.all([x == y for x, y in zip(test_transcriptome['chr'].to_list(), test_trans_chrom_compare)]):
            raise ValueError('Test chromosomes (transcript) did not match')
        for tm in trans_rmodels:
            dist_cen_tel = train_trans_diff_cen_tel if 'train' in tm.name.lower() else test_trans_diff_cen_tel
            for mr in [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25)]:
                tm.plot_parameter_with_gradient(
                    dist_cen_tel,
                    cmap='RdGy',
                    alpha=.7,
                    m_range=mr,
                    size_scaling=size_scaling,
                    size_power=size_power,
                    power_norm=power_norm,
                    save_fig=save_fig,
                    save_prefix='%s_meres_%s_%s' % (save_prefix, mr[0], mr[1])
                )

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
        train_igr_diff_cen_tel = np.asarray(train_igr_diff_cen_tel)
        test_igr_diff_cen_tel = np.asarray(test_igr_diff_cen_tel)
        for tm in igr_rmodels:
            dist_cen_tel = train_igr_diff_cen_tel if 'train' in tm.name.lower() else test_igr_diff_cen_tel
            for mr in [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25)]:
                tm.plot_parameter_with_gradient(
                    dist_cen_tel,
                    cmap='RdGy',
                    alpha=.7,
                    m_range=mr,
                    size_scaling=size_scaling,
                    size_power=size_power,
                    power_norm=power_norm,
                    save_fig=save_fig,
                    save_prefix='%s_meres_%s_%s' % (save_prefix, mr[0], mr[1])
                )


if __name__ == '__main__':
    args = argparse_bioplotter(sys.argv[1:])
    main(args)


