import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

from src.UtilsMain import argparse_errorplotter
from src.Utils import validate_dir


def main(args):
    array_folder = args.array_dir
    save_prefix = args.save_prefix
    max_iter = args.max_iter
    title_biotype = args.title_biotype
    save_fig = args.save_fig

    gene_s = np.empty(0)
    gene_c = np.empty(0)
    gene_e = np.empty(0)

    nts_s = np.empty(0)
    nts_c = np.empty(0)
    nts_e = np.empty(0)

    igr = np.empty(0)
    i = 0
    while True and i < max_iter:
        i += 1
        if 'total' in save_prefix:
            if not os.path.isfile('%s/%s%s_Train genes total.csv' % (array_folder, save_prefix, i)):
                break

            gs = np.loadtxt('%s/%s%s_Train genes total.csv' % (array_folder, save_prefix, i), delimiter=',')
            ntss = np.loadtxt('%s/%s%s_Train NTS total.csv' % (array_folder, save_prefix, i), delimiter=',')

            if 'size' not in title_biotype.lower():
                intergen = np.loadtxt(
                    '%s/%s%s_Train intergenic regions.csv' % (array_folder, save_prefix, i),
                    delimiter=','
                )
                igr = np.append(igr, intergen)

        else:
            if not os.path.isfile('%s/%s%s_Train genes start.csv' % (array_folder, save_prefix, i)):
                break
            gs = np.loadtxt('%s/%s%s_Train genes start.csv' % (array_folder, save_prefix, i), delimiter=',')
            gc = np.loadtxt('%s/%s%s_Train genes centre.csv' % (array_folder, save_prefix, i), delimiter=',')
            ge = np.loadtxt('%s/%s%s_Train genes end.csv' % (array_folder, save_prefix, i), delimiter=',')

            ntss = np.loadtxt('%s/%s%s_Train NTS start.csv' % (array_folder, save_prefix, i), delimiter=',')
            ntsc = np.loadtxt('%s/%s%s_Train NTS centre.csv' % (array_folder, save_prefix, i), delimiter=',')
            ntse = np.loadtxt('%s/%s%s_Train NTS end.csv' % (array_folder, save_prefix, i), delimiter=',')

            if 'size' not in title_biotype.lower():
                intergen = np.loadtxt(
                    '%s/%s%s_Train intergenic regions.csv' % (array_folder, save_prefix, i),
                    delimiter=','
                )
                igr = np.append(igr, intergen)

            gene_c = np.append(gene_c, gc)
            gene_e = np.append(gene_e, ge)
            nts_c = np.append(nts_c, ntsc)
            nts_e = np.append(nts_e, ntse)

        gene_s = np.append(gene_s, gs)
        nts_s = np.append(nts_s, ntss)

    rand_gene_s = np.empty(0)
    rand_gene_c = np.empty(0)
    rand_gene_e = np.empty(0)

    rand_nts_s = np.empty(0)
    rand_nts_c = np.empty(0)
    rand_nts_e = np.empty(0)

    rand_igr = np.empty(0)
    i = 0
    while True and i < max_iter:
        i += 1
        if 'total' in save_prefix:
            if not os.path.isfile('%s/%s_random%s_Train genes total.csv' % (array_folder, save_prefix, i)):
                break

            randg_s = np.loadtxt('%s/%s_random%s_Train genes total.csv' % (array_folder, save_prefix, i), delimiter=',')
            randnts_s = np.loadtxt('%s/%s_random%s_Train NTS total.csv' % (array_folder, save_prefix, i), delimiter=',')

            if 'size' not in title_biotype.lower():
                randigr = np.loadtxt(
                    '%s/%s_random%s_Train intergenic regions.csv' % (array_folder, save_prefix, i),
                    delimiter=','
                )
                rand_igr = np.append(rand_igr, randigr)
        else:
            if not os.path.isfile('%s/%s_random%s_Train genes start.csv' % (array_folder, save_prefix, i)):
                break
            randg_s = np.loadtxt('%s/%s_random%s_Train genes start.csv' % (array_folder, save_prefix, i), delimiter=',')
            randg_c = np.loadtxt('%s/%s_random%s_Train genes centre.csv' % (array_folder, save_prefix, i), delimiter=',')
            randg_e = np.loadtxt('%s/%s_random%s_Train genes end.csv' % (array_folder, save_prefix, i), delimiter=',')

            randnts_s = np.loadtxt('%s/%s_random%s_Train NTS start.csv' % (array_folder, save_prefix, i), delimiter=',')
            randnts_c = np.loadtxt('%s/%s_random%s_Train NTS centre.csv' % (array_folder, save_prefix, i), delimiter=',')
            randnts_e = np.loadtxt('%s/%s_random%s_Train NTS end.csv' % (array_folder, save_prefix, i), delimiter=',')

            if 'size' not in title_biotype.lower():
                randigr = np.loadtxt(
                    '%s/%s_random%s_Train intergenic regions.csv' % (array_folder, save_prefix, i),
                    delimiter=','
                )
                rand_igr = np.append(rand_igr, randigr)
            rand_gene_c = np.append(rand_gene_c, randg_c)
            rand_gene_e = np.append(rand_gene_e, randg_e)
            rand_nts_c = np.append(rand_nts_c, randnts_c)
            rand_nts_e = np.append(rand_nts_e, randnts_e)

        rand_gene_s = np.append(rand_gene_s, randg_s)
        rand_nts_s = np.append(rand_nts_s, randnts_s)

    palette = sns.color_palette()
    custom_lines = [Line2D([0], [0], color=palette[0], lw=4),
                    Line2D([0], [0], color=palette[-2], lw=4)]
    plt.figure(figsize=(8, 7))
    if 'size' not in title_biotype.lower():
        if 'total' in save_prefix:
            data = [
                gene_s, rand_gene_s,
                nts_s, rand_nts_s,
                igr, rand_igr
            ]
            palette = [
                palette[0], palette[-2],
                palette[0], palette[-2],
                palette[0], palette[-2]
            ]
            xticks = [.5, 2.5, 4.5]
            xtick_handles = ['Genes', 'NTS', 'IGR']
        else:
            data = [
                gene_s, rand_gene_s, gene_c, rand_gene_c, gene_e, rand_gene_e,
                nts_s, rand_nts_s, nts_c, rand_nts_c, nts_e, rand_nts_e,
                igr, rand_igr
            ]
            palette = [
                palette[0], palette[-2],
                palette[0], palette[-2],
                palette[0], palette[-2],
                palette[0], palette[-2],
                palette[0], palette[-2],
                palette[0], palette[-2],
                palette[0], palette[-2],

            ]
            xticks = [.5, 2.5, 4.5, 6.5, 8.5, 10.5, 12.5]
            xtick_handles = ['Gene s', 'Gene c', 'Gene e', 'NTS s', 'NTS c', 'NTS e', 'IGR']
    else:
        if 'total' in save_prefix:
            data = [
                gene_s, rand_gene_s,
                nts_s, rand_nts_s,
            ]
            palette = [
                palette[0], palette[-2],
                palette[0], palette[-2],
            ]
            xticks = [.5, 2.5]
            xtick_handles = ['Genes', 'NTS']
        else:
            data = [
                gene_s, rand_gene_s, gene_c, rand_gene_c, gene_e, rand_gene_e,
                nts_s, rand_nts_s, nts_c, rand_nts_c, nts_e
            ]
            palette = [
                palette[0], palette[-2],
                palette[0], palette[-2],
                palette[0], palette[-2],
                palette[0], palette[-2],
                palette[0], palette[-2],
                palette[0], palette[-2],
            ]
            xticks = [.5, 2.5, 4.5, 6.5, 8.5, 10.5]
            xtick_handles = ['Gene s', 'Gene c', 'Gene e', 'NTS s', 'NTS c', 'NTS e']
    sns.violinplot(data=data, palette=palette)
    plt.xticks(xticks, xtick_handles)
    plt.title('Error distribution %s' % title_biotype)
    plt.legend(custom_lines, ['Original', 'Random'])

    if save_fig:
        directory = validate_dir('figures/total_error_dist')
        plt.savefig('%s/compare_error_dist_%s.png' % (directory, save_prefix))
        plt.close('all')
    else:
        plt.show()


if __name__ == '__main__':
    arguments = argparse_errorplotter(sys.argv[1:])
    main(arguments)

