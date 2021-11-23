#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from src.DataLoader import load_chrom_data, load_chrom_split, load_bio_data, create_time_data, load_meres
from src.Model import RegionModel, BayesianParameterMap

NUCLEOSOME_INDEX = {
    'nouv': 0,
    '0min': 1,
    '30min': 2
}


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
            if 'genes' in name.lower():
                chrom_list = train_transcriptome['chr'].to_list()
            else:
                chrom_list = map(lambda x: x[0], train_start_igr)
        else:
            if 'genes' in name.lower():
                chrom_list = test_transcriptome['chr'].to_list()
            else:
                chrom_list = map(lambda x: x[0], test_start_igr)
        num_pos = 3 if not do_each and 'trans' in file_name else 1
        region_model = RegionModel(
            create_time_data(num_pos, len(data)),
            data.reshape(len(data), -1),
            name=name
        )
        region_model.load_models(file_name, compare_chrom_list=chrom_list)
        model_list.append(region_model)

    return model_list


def main_nucl():
    do_each = True
    save_fig = False
    kernel_func_type = 'eqk'
    kernel_search_verbosity = 2
    kernel_scaling_init = 10
    noise_scaling_init = 5
    min_m = .4
    max_m = 4.5
    min_mf = .4
    min_beta = 8e-3
    max_beta = 3.5e-2
    num_param_values = 10
    num_cpus = 4
    nucl_index = NUCLEOSOME_INDEX['nouv']

    nucl_paths = [
        'data/seq/nucl_wt_nouv.bw',
        'data/seq/nucl_wt_0min.bw',
        'data/seq/nucl_wt_30min.bw'
    ]

    print('Load CPD')
    train_chrom = load_chrom_split('train')[:2]
    test_chrom = load_chrom_split('test')[:2]
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

    gp_models = []
    test_data = []
    for rm in region_model_list:
        temp_m = np.asarray(list(rm.get_model_parameter('m', do_filter=False)))
        mask = ~np.isnan(temp_m)
        mask = temp_m < 6
        if 'gene' in rm.name.lower():
            train_nucl, test_nucl = train_nucl_trans[nucl_index], test_nucl_trans[nucl_index]
        else:
            train_nucl, test_nucl = train_nucl_igr[nucl_index], test_nucl_igr[nucl_index]
        rm.models = [model for num, model in enumerate(rm.models) if mask[num]]
        test_data.append(test_nucl)
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
        b_model.learn(num_cpus=num_cpus)
        gp_models.append(b_model)


if __name__ == '__main__':
    main_nucl()

