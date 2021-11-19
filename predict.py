#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from src.DataLoader import load_chrom_data, load_chrom_split, load_bio_data, create_time_data
from src.Model import RegionModel, BayesianParameterMap


def create_models(train_data, test_data, do_each=True):
    (train_trans, train_ntrans, train_igr, _, _, _) = train_data
    (test_trans, test_ntrans, test_igr, _, _, _) = test_data
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
        num_pos = 3 if not do_each and 'trans' in file_name else 1
        region_model = RegionModel(
            create_time_data(num_pos, len(data)),
            data.reshape(len(data), -1),
            name=name
        )
        region_model.load_models(file_name)
        model_list.append(region_model)

    return model_list


def main():
    do_each = True
    save_fig = False
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
    train_chrom = load_chrom_split('train')[:1]
    test_chrom = load_chrom_split('test')[:1]
    train_data, test_data = load_chrom_data(train_chrom_list=train_chrom, test_chrom_list=test_chrom)
    region_model_list = create_models(train_data, test_data, do_each=do_each)

    print('Make region models')
    (_, _, _, train_start_igr, train_end_igr, train_transcriptome) = train_data
    (_, _, _, test_start_igr, test_end_igr, test_transcriptome) = test_data

    # SLAM-seq data
    print('Load SLAM-seq')
    train_slam, test_slam = load_bio_data(
        zip(train_transcriptome['chr'].to_list(), train_transcriptome['start'].to_list()),
        zip(train_transcriptome['chr'].to_list(), train_transcriptome['end'].to_list()),
        zip(test_transcriptome['chr'].to_list(), test_transcriptome['start'].to_list()),
        zip(test_transcriptome['chr'].to_list(), test_transcriptome['end'].to_list()),
        slam_paths,
        use_directionality=True
    )

    print('Make figures')
    trans_rmodels = list(filter(lambda x: 'gene' in x.name.lower(), region_model_list))
    igr_rmodels = list(filter(lambda x: 'intergenic' in x.name.lower(), region_model_list))
    for tm in trans_rmodels:
        slam_data = train_slam if 'train' in tm.name.lower() else test_slam
        for sd, n in zip(slam_data, ['nouv', '20m', '120m']):
            for mr in [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25)]:
                tm.plot_parameter_with_gradient(
                    sd,
                    cmap='PiYG',
                    alpha=.7,
                    m_range=mr,
                    size_scaling=120,
                    power_norm=.3,
                    save_fig=save_fig,
                    save_prefix='slam_%s_%s_%s' % (n, mr[0], mr[1])
                )

    # Nucleosome data
    print('Load MNase-seq')
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

    print('Make figures')
    for tm in trans_rmodels:
        nucl_data = train_nucl_trans if 'train' in tm.name.lower() else test_nucl_trans
        for sd, n in zip(nucl_data, ['nouv', '20m', '120m']):
            for mr in [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25)]:
                tm.plot_parameter_with_gradient(
                    sd,
                    cmap='BrBG',
                    alpha=.7,
                    m_range=mr,
                    size_scaling=120,
                    power_norm=1.,
                    save_fig=save_fig,
                    save_prefix='nucl_%s_%s_%s' % (n, mr[0], mr[1])
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
                    size_scaling=120,
                    power_norm=1.,
                    save_fig=save_fig,
                    save_prefix='nucl_%s_%s_%s' % (n, mr[0], mr[1])
                )

    print('Make figures')
    train_size = np.abs(train_transcriptome['start'].to_numpy('int') - train_transcriptome['end'].to_numpy('int'))
    test_size = np.abs(test_transcriptome['start'].to_numpy('int') - test_transcriptome['end'].to_numpy('int'))
    for tm in trans_rmodels:
        size_data = train_size if 'train' in tm.name.lower() else test_size
        for mr in [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25)]:
            tm.plot_parameter_with_gradient(
                size_data,
                cmap='RdGy',
                alpha=.7,
                m_range=mr,
                size_scaling=120,
                power_norm=.4,
                save_fig=save_fig,
                save_prefix='size_%s_%s' % (mr[0], mr[1])
            )


if __name__ == '__main__':
    main()


