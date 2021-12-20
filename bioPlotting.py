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
    no_tcr = args.no_tcr
    used_transcriptoms = [True, False, False] if not no_tcr else [True, True, True]
    num_bins = 3 if not no_tcr else 1

    netseq_paths = [
        'data/seq/wt_netseq_minus.bw',
        'data/seq/wt_netseq_plus.bw'
    ]

    abf1_paths = [
        'data/seq/wt_abf1_nouv.bw',
        'data/seq/wt_abf1_uv.bw'
    ]

    h2aZ_paths = [
        'data/seq/wt_h2a_0m.bw'
    ]

    nucl_paths = [
        'data/seq/nucl_wt_nouv.bw',
        'data/seq/nucl_wt_0min.bw',
        'data/seq/nucl_wt_30min.bw'
    ]

    print('Load CPD')
    chrom_list = load_chrom_split()
    data = load_chrom_data(chrom_list=chrom_list, used_transcriptomes=used_transcriptoms, num_trans_bins=num_bins)
    region_model_list = create_models(data, do_each=do_each, no_tcr=no_tcr)

    (_, _, _, start_igr, end_igr, transcriptome) = data

    trans_rmodels = list(filter(lambda x: 'gene' in x.name.lower() or 'nts' in x.name.lower(), region_model_list))
    igr_rmodels = list(filter(lambda x: 'intergenic' in x.name.lower(), region_model_list))

    if bio_type.lower() == 'netseq':
        # NET-seq data
        print('Load NET-seq')
        netseq_data = load_bio_data(
            zip(transcriptome['chr'].to_list(), transcriptome['start'].to_list()),
            zip(transcriptome['chr'].to_list(), transcriptome['end'].to_list()),
            netseq_paths,
            use_directionality=True,
            use_sum=use_sum
        )

        for tm in trans_rmodels:
            for netd in netseq_data:
                for mr in [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25)]:
                    tm.plot_parameter_with_gradient(
                        netd,
                        cmap='PiYG',
                        alpha=.7,
                        m_range=mr,
                        title_add='| NET-seq',
                        size_scaling=size_scaling,
                        size_power=size_power,
                        power_norm=power_norm,
                        num_handles=8,
                        save_fig=save_fig,
                        save_prefix='%s_netseq_%s_%s' % (save_prefix, mr[0], mr[1])
                    )

    elif bio_type.lower() == 'nucl':
        # Nucleosome data
        print('Load MNase-seq')
        nucl_data_trans = load_bio_data(
            zip(transcriptome['chr'].to_list(), transcriptome['start'].to_list()),
            zip(transcriptome['chr'].to_list(), transcriptome['end'].to_list()),
            nucl_paths,
            use_directionality=False,
            use_sum=use_sum
        )
        nucl_data_igr = load_bio_data(
            start_igr,
            end_igr,
            nucl_paths,
            use_directionality=False,
            use_sum=use_sum
        )

        for tm in trans_rmodels:
            for sd, n in zip(nucl_data_trans, ['nouv', '0m', '30m']):
                for mr in [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25)]:
                    tm.plot_parameter_with_gradient(
                        sd,
                        cmap='BrBG',
                        alpha=.7,
                        m_range=mr,
                        title_add='| Nucleosome density %s' % n,
                        size_scaling=size_scaling,
                        size_power=size_power,
                        power_norm=power_norm,
                        save_fig=save_fig,
                        save_prefix='%s_nucl_%s_%s_%s' % (save_prefix, n, mr[0], mr[1])
                    )

        for tm in igr_rmodels:
            for sd, n in zip(nucl_data_igr, ['nouv', '0m', '30m']):
                for mr in [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25)]:
                    tm.plot_parameter_with_gradient(
                        sd,
                        cmap='BrBG',
                        alpha=.7,
                        m_range=mr,
                        title_add='| Nucleosome density %s' % n,
                        size_scaling=size_scaling,
                        size_power=size_power,
                        power_norm=power_norm,
                        save_fig=save_fig,
                        save_prefix='%s_nucl_%s_%s_%s' % (save_prefix, n, mr[0], mr[1])
                    )

    elif bio_type.lower() == 'size':
        # size
        print('Size')
        size_list = np.abs(transcriptome['start'].to_numpy('int') - transcriptome['end'].to_numpy('int'))
        for tm in trans_rmodels:
            for mr in [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25)]:
                tm.plot_parameter_with_gradient(
                    size_list,
                    cmap='Spectral',
                    alpha=.7,
                    title_add='| Size',
                    m_range=mr,
                    size_scaling=size_scaling,
                    size_power=size_power,
                    power_norm=power_norm,
                    save_fig=save_fig,
                    save_prefix='%s_size_%s_%s' % (save_prefix, mr[0], mr[1])
                )

    elif bio_type.lower() == 'abf1':
        # Load abf1 data
        abf1_data_trans = load_bio_data(
            zip(transcriptome['chr'].to_list(), transcriptome['start'].to_list()),
            zip(transcriptome['chr'].to_list(), transcriptome['end'].to_list()),
            abf1_paths,
            use_directionality=False,
            use_sum=use_sum
        )

        abf1_data_igr = load_bio_data(
            start_igr,
            end_igr,
            abf1_paths,
            use_directionality=False,
            use_sum=use_sum
        )

        for tm in trans_rmodels:
            for ad, n in zip(abf1_data_trans, ['nouv', '0m']):
                for mr in [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25)]:
                    tm.plot_parameter_with_gradient(
                        ad,
                        cmap='PuOr',
                        alpha=.7,
                        m_range=mr,
                        title_add='| Abf1 %s' % n,
                        size_scaling=size_scaling,
                        size_power=size_power,
                        power_norm=power_norm,
                        save_fig=save_fig,
                        save_prefix='%s_abf1_%s_%s_%s' % (save_prefix, n, mr[0], mr[1])
                    )

        for tm in igr_rmodels:
            for ad, n in zip(abf1_data_igr, ['nouv', '0m']):
                for mr in [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25)]:
                    tm.plot_parameter_with_gradient(
                        ad,
                        cmap='PuOr',
                        alpha=.7,
                        m_range=mr,
                        title_add='| Abf1 %s' % n,
                        size_scaling=size_scaling,
                        size_power=size_power,
                        power_norm=power_norm,
                        save_fig=save_fig,
                        save_prefix='%s_abf1_%s_%s_%s' % (save_prefix, n, mr[0], mr[1])
                    )
    elif bio_type.lower() == 'h2a':
        # Load h2aZ data
        h2aZ_data_trans = load_bio_data(
            zip(transcriptome['chr'].to_list(), transcriptome['start'].to_list()),
            zip(transcriptome['chr'].to_list(), transcriptome['end'].to_list()),
            h2aZ_paths,
            use_directionality=False,
            use_sum=use_sum
        )

        h2aZ_data_igr = load_bio_data(
            start_igr,
            end_igr,
            h2aZ_paths,
            use_directionality=False,
            use_sum=use_sum
        )

        for tm in trans_rmodels:
            for mr in [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25)]:
                tm.plot_parameter_with_gradient(
                    h2aZ_data_trans[0],
                    cmap='coolwarm',
                    alpha=.7,
                    title_add='| H2A.Z',
                    m_range=mr,
                    size_scaling=size_scaling,
                    size_power=size_power,
                    power_norm=power_norm,
                    save_fig=save_fig,
                    save_prefix='%s_h2az_%s_%s' % (save_prefix, mr[0], mr[1])
                )

        for im in igr_rmodels:
            for mr in [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25)]:
                im.plot_parameter_with_gradient(
                    h2aZ_data_igr[0],
                    cmap='coolwarm',
                    alpha=.7,
                    title_add='| H2A.Z',
                    m_range=mr,
                    size_scaling=size_scaling,
                    size_power=size_power,
                    power_norm=power_norm,
                    save_fig=save_fig,
                    save_prefix='%s_h2az_%s_%s' % (save_prefix, mr[0], mr[1])
                )

    elif 'meres' in bio_type.lower():
        # Telomeres and centromeres
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

        trans_diff_cen_tel = np.asarray(trans_diff_cen_tel)
        if not np.all([x == y for x, y in zip(transcriptome['chr'].to_list(), trans_chrom_compare)]):
            raise ValueError('Chromosomes (transcript) did not match')

        chr_igr, _ = zip(*start_igr)
        if not np.all([x == y for x, y in zip(chr_igr, igr_chrom_compare)]):
            raise ValueError('Chromosomes (IGR) did not match')

        igr_diff_cen_tel = np.asarray(igr_diff_cen_tel)
        for tm in trans_rmodels:
            for mr in [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25)]:
                tm.plot_parameter_with_gradient(
                    trans_diff_cen_tel,
                    cmap='RdGy',
                    alpha=.7,
                    m_range=mr,
                    title_add='| meres' if bio_type.lower() == 'meres' else '| Relative meres',
                    size_scaling=size_scaling,
                    size_power=size_power,
                    power_norm=power_norm,
                    save_fig=save_fig,
                    save_prefix='%s_meres_%s_%s' % (save_prefix, mr[0], mr[1])
                )

        igr_diff_cen_tel = np.asarray(igr_diff_cen_tel)
        for tm in igr_rmodels:
            for mr in [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25)]:
                tm.plot_parameter_with_gradient(
                    igr_diff_cen_tel,
                    cmap='RdGy',
                    alpha=.7,
                    m_range=mr,
                    title_add='| meres' if bio_type.lower() == 'meres' else '| Relative meres',
                    size_scaling=size_scaling,
                    size_power=size_power,
                    power_norm=power_norm,
                    save_fig=save_fig,
                    save_prefix='%s_%s_%s_%s' % (save_prefix, bio_type.lower(), mr[0], mr[1])
                )

    else:
        raise ValueError('Bio data identifier not understood. Choose between: '
                         'netseq | nucl | size | abf1 | h2a | meres | rel_meres')


if __name__ == '__main__':
    args = argparse_bioplotter(sys.argv[1:])
    main(args)


