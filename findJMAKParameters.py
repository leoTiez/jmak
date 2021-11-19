#!/usr/bin/env python3
import sys

from src.DataLoader import load_chrom_split, load_chrom_data, create_time_data
from src.Model import RegionModel
from src.Utils import argparse_jmak_param


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
        data.reshape(len(data), -1),
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


def test_main(args):
    param_chrom = load_chrom_split('parameter')
    param_data, _ = load_chrom_data(param_chrom, [])
    (param_trans, param_ntrans, param_igr, param_igr_start, param_igr_end, param_transcriptom) = param_data
    if args.do_each:
        create_model(
            param_trans[:, :, 0],
            1,
            name='Test parameter model Start',
            jmak_name_list=param_transcriptom['ORF'].to_list(),
            verbosity=args.verbosity,
            save_fig=args.save_fig,
            num_cpus=args.num_cpus,
            save_name='test_param_start'
        )
    else:
        create_model(
            param_trans,
            3,
            name='Test parameter model total',
            jmak_name_list=param_transcriptom['ORF'].to_list(),
            verbosity=args.verbosity,
            save_fig=args.save_fig,
            num_cpus=args.num_cpus,
            save_name='test_param_totla'
        )


def main(args):
    train_chrom = load_chrom_split('train')
    test_chrom = load_chrom_split('test')

    min_f_trans = args.min_f_trans
    min_f_igr = args.min_f_igr
    max_f = args.max_f
    delta_f = args.delta_f
    num_cpus = args.num_cpus
    heatmap_color_trans = 'Greens'
    heatmap_color_ntrans = 'Oranges'
    heatmap_color_igr = 'Blues'
    verbosity = args.verbosity
    save_fig = args.save_fig

    train_data, test_data = load_chrom_data(train_chrom_list=train_chrom, test_chrom_list=test_chrom)
    (train_trans, train_ntrans, train_igr, train_igr_start, train_igr_end, train_transcriptome) = train_data
    (test_trans, test_ntrans, test_igr, test_igr_start, test_igr_end, test_transcriptome) = test_data

    # Train
    # Transcripts
    if args.do_each:
        create_model(
            train_trans[:, :, 0],
            1,
            'Train genes start',
            min_f=min_f_trans,
            max_f=max_f,
            delta_f=delta_f,
            jmak_name_list=train_transcriptome['ORF'].to_list(),
            heatmap_color=heatmap_color_trans,
            num_cpus=num_cpus,
            verbosity=verbosity,
            save_fig=save_fig,
            save_name='train_trans_start'
        )
        create_model(
            train_trans[:, :, 1],
            1,
            'Train genes centre',
            min_f=min_f_trans,
            max_f=max_f,
            delta_f=delta_f,
            jmak_name_list=train_transcriptome['ORF'].to_list(),
            heatmap_color=heatmap_color_trans,
            num_cpus=num_cpus,
            verbosity=verbosity,
            save_fig=save_fig,
            save_name='train_trans_centre'
        )
        create_model(
            train_trans[:, :, 2],
            1,
            'Train genes end',
            min_f=min_f_trans,
            max_f=max_f,
            delta_f=delta_f,
            jmak_name_list=train_transcriptome['ORF'].to_list(),
            heatmap_color=heatmap_color_trans,
            num_cpus=num_cpus,
            verbosity=verbosity,
            save_fig=save_fig,
            save_name='train_trans_end'
        )
    else:
        create_model(
            train_trans,
            3,
            'Train genes total',
            min_f=min_f_trans,
            max_f=max_f,
            delta_f=delta_f,
            jmak_name_list=train_transcriptome['ORF'].to_list(),
            heatmap_color=heatmap_color_trans,
            num_cpus=num_cpus,
            verbosity=verbosity,
            save_fig=save_fig,
            save_name='train_trans_total'
        )
    # NTS
    if args.do_each:
        create_model(
            train_ntrans[:, :, 0],
            1,
            'Train NTS start',
            min_f=min_f_igr,
            max_f=max_f,
            delta_f=delta_f,
            jmak_name_list=train_transcriptome['ORF'].to_list(),
            heatmap_color=heatmap_color_ntrans,
            num_cpus=num_cpus,
            verbosity=verbosity,
            save_fig=save_fig,
            save_name='train_ntrans_start'
        )
        create_model(
            train_ntrans[:, :, 1],
            1,
            'Train NTS centre',
            min_f=min_f_igr,
            max_f=max_f,
            delta_f=delta_f,
            jmak_name_list=train_transcriptome['ORF'].to_list(),
            heatmap_color=heatmap_color_ntrans,
            num_cpus=num_cpus,
            verbosity=verbosity,
            save_fig=save_fig,
            save_name='train_ntrans_centre'
        )
        create_model(
            train_ntrans[:, :, 2],
            1,
            'Train NTS end',
            min_f=min_f_igr,
            max_f=max_f,
            delta_f=delta_f,
            jmak_name_list=train_transcriptome['ORF'].to_list(),
            heatmap_color=heatmap_color_ntrans,
            num_cpus=num_cpus,
            verbosity=verbosity,
            save_fig=save_fig,
            save_name='train_ntrans_end'
        )
    else:
        create_model(
            train_ntrans,
            3,
            'Train NTS total',
            min_f=min_f_igr,
            max_f=max_f,
            delta_f=delta_f,
            jmak_name_list=train_transcriptome['ORF'].to_list(),
            heatmap_color=heatmap_color_ntrans,
            num_cpus=num_cpus,
            verbosity=verbosity,
            save_fig=save_fig,
            save_name='train_ntrans_total'
        )
    # Intergenic
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
    # Transcripts
    if args.do_each:
        create_model(
            test_trans[:, :, 0],
            1,
            'Test genes start',
            min_f=min_f_trans,
            max_f=max_f,
            delta_f=delta_f,
            jmak_name_list=test_transcriptome['ORF'].to_list(),
            heatmap_color=heatmap_color_trans,
            num_cpus=num_cpus,
            verbosity=verbosity,
            save_fig=save_fig,
            save_name='test_trans_start'
        )
        create_model(
            test_trans[:, :, 1],
            1,
            'Test genes centre',
            min_f=min_f_trans,
            max_f=max_f,
            delta_f=delta_f,
            jmak_name_list=test_transcriptome['ORF'].to_list(),
            heatmap_color=heatmap_color_trans,
            num_cpus=num_cpus,
            verbosity=verbosity,
            save_fig=save_fig,
            save_name='test_centre'
        )
        create_model(
            test_trans[:, :, 2],
            1,
            'Test genes end',
            min_f=min_f_trans,
            max_f=max_f,
            delta_f=delta_f,
            jmak_name_list=test_transcriptome['ORF'].to_list(),
            heatmap_color=heatmap_color_trans,
            num_cpus=num_cpus,
            verbosity=verbosity,
            save_fig=save_fig,
            save_name='test_trans_end'
        )
    else:
        create_model(
            test_trans,
            3,
            'Test genes total',
            min_f=min_f_trans,
            max_f=max_f,
            delta_f=delta_f,
            jmak_name_list=test_transcriptome['ORF'].to_list(),
            heatmap_color=heatmap_color_trans,
            num_cpus=num_cpus,
            verbosity=verbosity,
            save_fig=save_fig,
            save_name='test_trans_total'
        )
    # NTS
    if args.do_each:
        create_model(
            test_ntrans[:, :, 0],
            1,
            'Test NTS start',
            min_f=min_f_igr,
            max_f=max_f,
            delta_f=delta_f,
            jmak_name_list=test_transcriptome['ORF'].to_list(),
            heatmap_color=heatmap_color_ntrans,
            num_cpus=num_cpus,
            verbosity=verbosity,
            save_fig=save_fig,
            save_name='test_ntrans_start'
        )
        create_model(
            test_ntrans[:, :, 1],
            1,
            'Test NTS centre',
            min_f=min_f_igr,
            max_f=max_f,
            delta_f=delta_f,
            jmak_name_list=test_transcriptome['ORF'].to_list(),
            heatmap_color=heatmap_color_ntrans,
            num_cpus=num_cpus,
            verbosity=verbosity,
            save_fig=save_fig,
            save_name='test_ntrans_centre'
        )
        create_model(
            test_ntrans[:, :, 2],
            1,
            'Test NTS end',
            min_f=min_f_igr,
            max_f=max_f,
            delta_f=delta_f,
            jmak_name_list=test_transcriptome['ORF'].to_list(),
            heatmap_color=heatmap_color_ntrans,
            num_cpus=num_cpus,
            verbosity=verbosity,
            save_fig=save_fig,
            save_name='test_ntrans_end'
        )
    else:
        create_model(
            test_ntrans,
            3,
            'Test NTS total',
            min_f=min_f_igr,
            max_f=max_f,
            delta_f=delta_f,
            jmak_name_list=test_transcriptome['ORF'].to_list(),
            heatmap_color=heatmap_color_ntrans,
            num_cpus=num_cpus,
            verbosity=verbosity,
            save_fig=save_fig,
            save_name='test_ntrans_total'
        )
    # Intergenic
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
    args = argparse_jmak_param(sys.argv[1:])
    # test_main(args)
    main(args)

