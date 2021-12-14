#!/usr/bin/env python3
import sys

from src.DataLoader import load_chrom_split, load_chrom_data, create_time_data
from src.Model import RegionModel
from src.UtilsMain import argparse_jmak_param


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
            param_ntrans[:, :, 0],
            1,
            name='Test nts model Start',
            jmak_name_list=list(map(
                lambda x: '%s %s' % (x[0], x[1]),
                zip(param_transcriptom['chr'].to_list(), param_transcriptom['ORF'].to_list())
            )),
            verbosity=args.verbosity,
            save_fig=args.save_fig,
            num_cpus=args.num_cpus,
            save_name='test_param_start'
        )
    else:
        create_model(
            param_igr,
            2,
            name='Test parameter model igr',
            jmak_name_list=list(map(
                lambda x: '%s %s' % (x[0][0], x[1]),
                zip(param_igr_start, range(len(param_igr_start)))
            )),
            verbosity=args.verbosity,
            save_fig=args.save_fig,
            num_cpus=args.num_cpus,
            save_name='test_param_total'
        )


def main(args):
    chrom_list = load_chrom_split()
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
    no_tcr = args.no_tcr
    used_transcriptoms = [True, False, False] if not no_tcr else [True, True, True]
    num_bins = 3 if not no_tcr else 1

    train_data = load_chrom_data(chrom_list=chrom_list, used_transcriptomes=used_transcriptoms, num_trans_bins=num_bins)
    (trans, ntrans, igr, igr_start, igr_end, transcriptome) = train_data

    # Transcripts
    if no_tcr:
        create_model(
            trans,
            1,
            'All Genes',
            min_f=min_f_trans,
            max_f=max_f,
            delta_f=delta_f,
            jmak_name_list=list(map(
                lambda x: '%s %s' % (x[0], x[1]),
                zip(transcriptome['chr'].to_list(), transcriptome['ORF'].to_list())
            )),
            heatmap_color=heatmap_color_trans,
            num_cpus=num_cpus,
            verbosity=verbosity,
            save_fig=save_fig,
            save_name='all_trans'
        )
        create_model(
            ntrans,
            1,
            'All NTS',
            min_f=min_f_trans,
            max_f=max_f,
            delta_f=delta_f,
            jmak_name_list=list(map(
                lambda x: '%s %s' % (x[0], x[1]),
                zip(transcriptome['chr'].to_list(), transcriptome['ORF'].to_list())
            )),
            heatmap_color=heatmap_color_trans,
            num_cpus=num_cpus,
            verbosity=verbosity,
            save_fig=save_fig,
            save_name='all_trans'
        )
        create_model(
            igr[:, :, 0],
            1,
            'IGR Strand +',
            min_f=min_f_igr,
            max_f=max_f,
            delta_f=delta_f,
            jmak_name_list=list(map(
                    lambda x: '%s %s' % (x[0][0], x[1]),
                    zip(igr_start, range(len(igr_start)))
                )),
            heatmap_color=heatmap_color_trans,
            num_cpus=num_cpus,
            verbosity=verbosity,
            save_fig=save_fig,
            save_name='igr_plus'
        )
        create_model(
            igr[:, :, 1],
            1,
            'IGR Strand -',
            min_f=min_f_igr,
            max_f=max_f,
            delta_f=delta_f,
            jmak_name_list=list(map(
                    lambda x: '%s %s' % (x[0][0], x[1]),
                    zip(igr_start, range(len(igr_start)))
                )),
            heatmap_color=heatmap_color_trans,
            num_cpus=num_cpus,
            verbosity=verbosity,
            save_fig=save_fig,
            save_name='igr_minus'
        )
    else:
        if args.do_each:
            create_model(
                trans[:, :, 0],
                1,
                'Genes start',
                min_f=min_f_trans,
                max_f=max_f,
                delta_f=delta_f,
                jmak_name_list=list(map(
                    lambda x: '%s %s' % (x[0], x[1]),
                    zip(transcriptome['chr'].to_list(), transcriptome['ORF'].to_list())
                )),
                heatmap_color=heatmap_color_trans,
                num_cpus=num_cpus,
                verbosity=verbosity,
                save_fig=save_fig,
                save_name='trans_start'
            )
            create_model(
                trans[:, :, 1],
                1,
                'Genes centre',
                min_f=min_f_trans,
                max_f=max_f,
                delta_f=delta_f,
                jmak_name_list=list(map(
                    lambda x: '%s %s' % (x[0], x[1]),
                    zip(transcriptome['chr'].to_list(), transcriptome['ORF'].to_list())
                )),
                heatmap_color=heatmap_color_trans,
                num_cpus=num_cpus,
                verbosity=verbosity,
                save_fig=save_fig,
                save_name='trans_centre'
            )
            create_model(
                trans[:, :, 2],
                1,
                'Genes end',
                min_f=min_f_trans,
                max_f=max_f,
                delta_f=delta_f,
                jmak_name_list=list(map(
                    lambda x: '%s %s' % (x[0], x[1]),
                    zip(transcriptome['chr'].to_list(), transcriptome['ORF'].to_list())
                )),
                heatmap_color=heatmap_color_trans,
                num_cpus=num_cpus,
                verbosity=verbosity,
                save_fig=save_fig,
                save_name='trans_end'
            )
        else:
            create_model(
                trans,
                3,
                'Genes total',
                min_f=min_f_trans,
                max_f=max_f,
                delta_f=delta_f,
                jmak_name_list=list(map(
                    lambda x: '%s %s' % (x[0], x[1]),
                    zip(transcriptome['chr'].to_list(), transcriptome['ORF'].to_list())
                )),
                heatmap_color=heatmap_color_trans,
                num_cpus=num_cpus,
                verbosity=verbosity,
                save_fig=save_fig,
                save_name='trans_total'
            )
        # NTS
        if args.do_each:
            create_model(
                ntrans[:, :, 0],
                1,
                'NTS start',
                min_f=min_f_igr,
                max_f=max_f,
                delta_f=delta_f,
                jmak_name_list=list(map(
                    lambda x: '%s %s' % (x[0], x[1]),
                    zip(transcriptome['chr'].to_list(), transcriptome['ORF'].to_list())
                )),
                heatmap_color=heatmap_color_ntrans,
                num_cpus=num_cpus,
                verbosity=verbosity,
                save_fig=save_fig,
                save_name='ntrans_start'
            )
            create_model(
                ntrans[:, :, 1],
                1,
                'NTS centre',
                min_f=min_f_igr,
                max_f=max_f,
                delta_f=delta_f,
                jmak_name_list=list(map(
                    lambda x: '%s %s' % (x[0], x[1]),
                    zip(transcriptome['chr'].to_list(), transcriptome['ORF'].to_list())
                )),
                heatmap_color=heatmap_color_ntrans,
                num_cpus=num_cpus,
                verbosity=verbosity,
                save_fig=save_fig,
                save_name='ntrans_centre'
            )
            create_model(
                ntrans[:, :, 2],
                1,
                'NTS end',
                min_f=min_f_igr,
                max_f=max_f,
                delta_f=delta_f,
                jmak_name_list=list(map(
                    lambda x: '%s %s' % (x[0], x[1]),
                    zip(transcriptome['chr'].to_list(), transcriptome['ORF'].to_list())
                )),
                heatmap_color=heatmap_color_ntrans,
                num_cpus=num_cpus,
                verbosity=verbosity,
                save_fig=save_fig,
                save_name='ntrans_end'
            )
        else:
            create_model(
                ntrans,
                3,
                'NTS total',
                min_f=min_f_igr,
                max_f=max_f,
                delta_f=delta_f,
                jmak_name_list=list(map(
                    lambda x: '%s %s' % (x[0], x[1]),
                    zip(transcriptome['chr'].to_list(), transcriptome['ORF'].to_list())
                )),
                heatmap_color=heatmap_color_ntrans,
                num_cpus=num_cpus,
                verbosity=verbosity,
                save_fig=save_fig,
                save_name='ntrans_total'
            )
        # Intergenic
        create_model(
            igr,
            2,
            'Intergenic regions',
            min_f=min_f_igr,
            max_f=max_f,
            delta_f=delta_f,
            jmak_name_list=list(map(
                    lambda x: '%s %s' % (x[0][0], x[1]),
                    zip(igr_start, range(len(igr_start)))
                )),
            heatmap_color=heatmap_color_igr,
            num_cpus=num_cpus,
            verbosity=verbosity,
            save_fig=save_fig,
            save_name='igr'
        )


if __name__ == '__main__':
    args = argparse_jmak_param(sys.argv[1:])
    if args.test:
        test_main(args)
    else:
        main(args)

