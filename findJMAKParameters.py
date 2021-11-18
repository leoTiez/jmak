#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from src.DataLoader import load_chrom_split, load_chrom_data, create_time_data
from src.Model import RegionModel


def create_model(
        data,
        num_pos,
        name,
        min_f=.4,
        max_f=1.,
        delta_f=.01,
        jmak_name_list=None,
        heatmap_color='Dark2',
        num_cpus=5,
        verbosity=0,
        save_fig=True,
        save_name=''
):
    region_model = RegionModel(
        create_time_data(num_pos, len(data)),
        data,
        name=name
    )
    region_model.fit_models(
        min_f=min_f,
        max_f=max_f,
        delta_f=delta_f,
        names=jmak_name_list,
        verbosity=verbosity,
        num_cpus=num_cpus,
        save_fig=save_fig,
        save_prefix=save_name
    )
    region_model.plot_parameter_histogram(
        heatmap_color,
        norm_gamma=1,
        save_fig=save_fig,
        save_prefix=save_name
    )
    region_model.to_file(save_name)


def test_main():
    param_chrom = load_chrom_split('parameter')
    param_data, _ = load_chrom_data(param_chrom, [])
    (param_trans, param_ntrans, param_igr, param_igr_start, param_igr_end, param_transcriptom) = param_data
    create_model(
        param_trans,
        3,
        name='Test Parameter Model',
        jmak_name_list=param_transcriptom['ORF'].to_list(),
        verbosity=2,
        save_fig=True,
        save_name='test_param'
    )


def main():
    train_chrom = load_chrom_split('train')
    test_chrom = load_chrom_split('test')

    min_f_trans = .5
    min_f_igr = .4
    max_f = 1.
    delta_f = .01
    num_cpus = 20
    heatmap_color_trans = 'Greens'
    heatmap_color_ntrans = 'Oranges'
    heatmap_color_igr = 'Blues'
    verbosity = 3
    save_fig = True

    train_data, test_data = load_chrom_data(train_chrom_list=train_chrom, test_chrom_list=test_chrom)
    (train_trans, train_ntrans, train_igr, train_igr_start, train_igr_end, train_transcriptome) = train_data
    (test_trans, test_ntrans, test_igr, test_igr_start, test_igr_end, test_transcriptome) = test_data

    # Train
    create_model(
        train_trans,
        3,
        'Train genes',
        min_f=min_f_trans,
        max_f=max_f,
        delta_f=delta_f,
        jmak_name_list=train_transcriptome['ORF'].to_list(),
        heatmap_color=heatmap_color_trans,
        num_cpus=num_cpus,
        verbosity=verbosity,
        save_fig=save_fig,
        save_name='train_trans'
    )
    create_model(
        train_ntrans,
        3,
        'Train NTS',
        min_f=min_f_igr,
        max_f=max_f,
        delta_f=delta_f,
        jmak_name_list=train_transcriptome['ORF'].to_list(),
        heatmap_color=heatmap_color_ntrans,
        num_cpus=num_cpus,
        verbosity=verbosity,
        save_fig=save_fig,
        save_name='train_ntrans'
    )
    create_model(
        train_igr,
        1,
        'Train intergenic regions',
        min_f=min_f_igr,
        max_f=max_f,
        delta_f=delta_f,
        jmak_name_list=None,
        heatmap_color=heatmap_color_igr,
        num_cpus=num_cpus,
        verbosity=verbosity,
        save_fig=save_fig,
        save_name='train_igr'
    )

    # Test
    create_model(
        test_trans,
        3,
        'Test genes',
        min_f=min_f_trans,
        max_f=max_f,
        delta_f=delta_f,
        jmak_name_list=test_transcriptome['ORF'].to_list(),
        heatmap_color=heatmap_color_trans,
        num_cpus=num_cpus,
        verbosity=verbosity,
        save_fig=save_fig,
        save_name='test_trans'
    )
    create_model(
        test_ntrans,
        3,
        'Test NTS',
        min_f=min_f_igr,
        max_f=max_f,
        delta_f=delta_f,
        jmak_name_list=test_transcriptome['ORF'].to_list(),
        heatmap_color=heatmap_color_ntrans,
        num_cpus=num_cpus,
        verbosity=verbosity,
        save_fig=save_fig,
        save_name='test_ntrans'
    )
    create_model(
        test_igr,
        1,
        'test intergenic regions',
        min_f=min_f_igr,
        max_f=max_f,
        delta_f=delta_f,
        jmak_name_list=None,
        heatmap_color=heatmap_color_igr,
        num_cpus=num_cpus,
        verbosity=verbosity,
        save_fig=save_fig,
        save_name='test_igr'
    )


if __name__ == '__main__':
    # test_main()
    main()

