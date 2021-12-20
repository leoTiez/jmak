import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.stats import ttest_ind

from src.UtilsMain import argparse_errorplotter
from src.Utils import validate_dir


def trim_data(data, rm_percentile=5):
    lower, upper = np.percentile(data, [rm_percentile / 2., 100. - rm_percentile / 2.])
    return data[np.logical_and(data > lower, data < upper)]


def main(args):
    array_folder = args.array_dir
    save_prefix = args.save_prefix
    max_iter = args.max_iter
    title_biotype = args.title_biotype
    save_fig = args.save_fig
    rm_percentile = 20
    p_thresh = .0001

    gene_s = np.empty(0)
    gene_c = np.empty(0)
    gene_e = np.empty(0)

    nts_s = np.empty(0)
    nts_c = np.empty(0)
    nts_e = np.empty(0)

    igr = np.empty(0)
    igr_minus = np.empty(0)
    i = 0
    while True and i < max_iter:
        i += 1
        if 'total' in save_prefix or 'notcr' in save_prefix:
            if not os.path.isfile('%s/%s%s_test_Genes total_error.txt' % (array_folder, save_prefix, i)) \
                    and not os.path.isfile('%s/%s%s_test_All Genes_error.txt' % (array_folder, save_prefix, i)):
                continue

            if 'total' in save_prefix and not 'notcr':
                gs = np.loadtxt('%s/%s%s_test_Genes total_error.txt' % (array_folder, save_prefix, i), delimiter=',')
                ntss = np.loadtxt('%s/%s%s_test_NTS total_error.txt' % (array_folder, save_prefix, i), delimiter=',')
            else:
                gs = np.loadtxt('%s/%s%s_test_All Genes_error.txt' % (array_folder, save_prefix, i), delimiter=',')
                ntss = np.loadtxt('%s/%s%s_test_All NTS_error.txt' % (array_folder, save_prefix, i), delimiter=',')
            if 'size' not in save_prefix.lower() and 'netseq' not in save_prefix.lower():
                if 'total' in save_prefix and not 'notcr':
                    intergen = np.loadtxt(
                        '%s/%s%s_test_Intergenic regions_error.txt' % (array_folder, save_prefix, i),
                        delimiter=','
                    )
                else:
                    intergen = np.loadtxt(
                        '%s/%s%s_test_IGR Strand +_error.txt' % (array_folder, save_prefix, i),
                        delimiter=','
                    )
                    intergen_minus = np.loadtxt(
                        '%s/%s%s_test_IGR Strand -_error.txt' % (array_folder, save_prefix, i),
                        delimiter=','
                    )
                    igr_minus = np.append(igr_minus, intergen_minus)
                igr = np.append(igr, intergen)

        else:
            if not os.path.isfile('%s/%s%s_test_Genes start_error.txt' % (array_folder, save_prefix, i)):
                continue
            gs = np.loadtxt('%s/%s%s_test_Genes start_error.txt' % (array_folder, save_prefix, i), delimiter=',')
            gc = np.loadtxt('%s/%s%s_test_Genes centre_error.txt' % (array_folder, save_prefix, i), delimiter=',')
            ge = np.loadtxt('%s/%s%s_test_Genes end_error.txt' % (array_folder, save_prefix, i), delimiter=',')

            ntss = np.loadtxt('%s/%s%s_test_NTS start_error.txt' % (array_folder, save_prefix, i), delimiter=',')
            ntsc = np.loadtxt('%s/%s%s_test_NTS centre_error.txt' % (array_folder, save_prefix, i), delimiter=',')
            ntse = np.loadtxt('%s/%s%s_test_NTS end_error.txt' % (array_folder, save_prefix, i), delimiter=',')

            if 'size' not in save_prefix.lower() and 'netseq' not in save_prefix.lower():
                intergen = np.loadtxt(
                    '%s/%s%s_test_Intergenic regions_error.txt' % (array_folder, save_prefix, i),
                    delimiter=','
                )
                igr = np.append(igr, intergen)

            gene_c = np.append(gene_c, gc)
            gene_e = np.append(gene_e, ge)
            nts_c = np.append(nts_c, ntsc)
            nts_e = np.append(nts_e, ntse)

        gene_s = np.append(gene_s, gs)
        nts_s = np.append(nts_s, ntss)

    if 'gp' in save_prefix:
        gene_s = trim_data(gene_s, rm_percentile=rm_percentile)
        nts_s = trim_data(nts_s, rm_percentile=rm_percentile)
        if 'size' not in save_prefix.lower() and 'netseq' not in save_prefix.lower():
            igr = trim_data(igr, rm_percentile=rm_percentile)
        if 'each' in save_prefix:
            gene_c = trim_data(gene_c, rm_percentile=rm_percentile)
            gene_e = trim_data(gene_e, rm_percentile=rm_percentile)
            nts_c = trim_data(nts_c, rm_percentile=rm_percentile)
            nts_e = trim_data(nts_e, rm_percentile=rm_percentile)

    rand_gene_s = np.empty(0)
    rand_gene_c = np.empty(0)
    rand_gene_e = np.empty(0)

    rand_nts_s = np.empty(0)
    rand_nts_c = np.empty(0)
    rand_nts_e = np.empty(0)

    rand_igr = np.empty(0)
    rand_igr_minus = np.empty(0)
    i = 0
    while True and i < max_iter:
        i += 1
        if 'total' in save_prefix or 'notcr' in save_prefix:
            if not os.path.isfile('%s/%s%s_random_test_Genes total_error.txt' % (array_folder, save_prefix, i))\
                    and not os.path.isfile('%s/%s%s_random_test_All NTS_error.txt' % (array_folder, save_prefix, i)):
                continue

            if 'total' in save_prefix and not 'notcr':
                randg_s = np.loadtxt(
                    '%s/%s%s_random_test_Genes total_error.txt'
                    % (array_folder, save_prefix, i),
                    delimiter=','
                )
                randnts_s = np.loadtxt(
                    '%s/%s%s_random_test_NTS total_error.txt'
                    % (array_folder, save_prefix, i),
                    delimiter=','
                )
            else:
                randg_s = np.loadtxt(
                    '%s/%s%s_random_test_All Genes_error.txt'
                    % (array_folder, save_prefix, i),
                    delimiter=','
                )
                randnts_s = np.loadtxt(
                    '%s/%s%s_random_test_All NTS_error.txt'
                    % (array_folder, save_prefix, i),
                    delimiter=','
                )

            if 'size' not in save_prefix.lower() and 'netseq' not in save_prefix.lower():
                if 'total' in save_prefix and not 'notcr':
                    randigr = np.loadtxt(
                        '%s/%s%s_random_test_Intergenic regions_error.txt' % (array_folder, save_prefix, i),
                        delimiter=','
                    )
                else:
                    randigr = np.loadtxt(
                        '%s/%s%s_random_test_IGR Strand +_error.txt' % (array_folder, save_prefix, i),
                        delimiter=','
                    )
                    randigr_minus = np.loadtxt(
                        '%s/%s%s_random_test_IGR Strand -_error.txt' % (array_folder, save_prefix, i),
                        delimiter=','
                    )
                    rand_igr_minus = np.append(rand_igr_minus, randigr_minus)
                rand_igr = np.append(rand_igr, randigr)
        else:
            if not os.path.isfile('%s/%s%s_random_test_Genes start_error.txt' % (array_folder, save_prefix, i)):
                continue
            randg_s = np.loadtxt('%s/%s%s_random_test_Genes start_error.txt' % (array_folder, save_prefix, i), delimiter=',')
            randg_c = np.loadtxt('%s/%s%s_random_test_Genes centre_error.txt' % (array_folder, save_prefix, i), delimiter=',')
            randg_e = np.loadtxt('%s/%s%s_random_test_Genes end_error.txt' % (array_folder, save_prefix, i), delimiter=',')

            randnts_s = np.loadtxt('%s/%s%s_random_test_NTS start_error.txt' % (array_folder, save_prefix, i), delimiter=',')
            randnts_c = np.loadtxt('%s/%s%s_random_test_NTS centre_error.txt' % (array_folder, save_prefix, i), delimiter=',')
            randnts_e = np.loadtxt('%s/%s%s_random_test_NTS end_error.txt' % (array_folder, save_prefix, i), delimiter=',')

            if 'size' not in save_prefix.lower() and 'netseq' not in save_prefix.lower():
                randigr = np.loadtxt(
                    '%s/%s%s_random_test_Intergenic regions_error.txt' % (array_folder, save_prefix, i),
                    delimiter=','
                )
                rand_igr = np.append(rand_igr, randigr)
            rand_gene_c = np.append(rand_gene_c, randg_c)
            rand_gene_e = np.append(rand_gene_e, randg_e)
            rand_nts_c = np.append(rand_nts_c, randnts_c)
            rand_nts_e = np.append(rand_nts_e, randnts_e)

        rand_gene_s = np.append(rand_gene_s, randg_s)
        rand_nts_s = np.append(rand_nts_s, randnts_s)

    if 'gp' in save_prefix:
        rand_gene_s = trim_data(rand_gene_s, rm_percentile=rm_percentile)
        rand_nts_s = trim_data(rand_nts_s, rm_percentile=rm_percentile)
        if 'size' not in save_prefix.lower() and 'netseq' not in save_prefix.lower():
            rand_igr = trim_data(rand_igr, rm_percentile=rm_percentile)
        if 'each' in save_prefix:
            rand_gene_c = trim_data(rand_gene_c, rm_percentile=rm_percentile)
            rand_gene_e = trim_data(rand_gene_e, rm_percentile=rm_percentile)
            rand_nts_c = trim_data(rand_nts_c, rm_percentile=rm_percentile)
            rand_nts_e = trim_data(rand_nts_e, rm_percentile=rm_percentile)

    cpalette = sns.color_palette()
    custom_lines = [Line2D([0], [0], color=cpalette[3], lw=4),
                    Line2D([0], [0], color=cpalette[0], lw=4),
                    Line2D([0], [0], color=cpalette[-2], lw=4)]
    plt.figure(figsize=(8, 7))
    if 'size' not in save_prefix.lower() and 'netseq' not in save_prefix.lower():
        if 'total' in save_prefix and not 'notcr':
            data = [
                gene_s, rand_gene_s,
                nts_s, rand_nts_s,
                igr, rand_igr
            ]
            _, p_gene_s = ttest_ind(gene_s, rand_gene_s, alternative='less')
            _, p_nts_s = ttest_ind(nts_s, rand_nts_s, alternative='less')
            _, p_igr = ttest_ind(igr, rand_igr, alternative='less')
            p_values = [p_gene_s, p_nts_s, p_igr]
            palette = []
            for p in p_values:
                palette.extend([cpalette[3], cpalette[-2]] if p < p_thresh else [cpalette[0], cpalette[-2]])

            xticks = [.5, 2.5, 4.5]
            xtick_handles = ['Genes', 'NTS', 'IGR']
        elif 'notcr' in save_prefix:
            data = [
                gene_s, rand_gene_s,
                nts_s, rand_nts_s,
                igr, rand_igr,
                igr_minus, rand_igr_minus
            ]
            _, p_gene_s = ttest_ind(gene_s, rand_gene_s, alternative='less')
            _, p_nts_s = ttest_ind(nts_s, rand_nts_s, alternative='less')
            _, p_igr = ttest_ind(igr, rand_igr, alternative='less')
            _, p_igr_minus = ttest_ind(igr_minus, rand_igr_minus, alternative='less')
            p_values = [p_gene_s, p_nts_s, p_igr, p_igr_minus]
            palette = []
            for p in p_values:
                palette.extend([cpalette[3], cpalette[-2]] if p < p_thresh else [cpalette[0], cpalette[-2]])
            xticks = [.5, 2.5, 4.5, 6.5]
            xtick_handles = ['Genes', 'NTS', 'IGR +', 'IGR -']
        else:
            data = [
                gene_s, rand_gene_s, gene_c, rand_gene_c, gene_e, rand_gene_e,
                nts_s, rand_nts_s, nts_c, rand_nts_c, nts_e, rand_nts_e,
                igr, rand_igr
            ]
            _, p_gene_s = ttest_ind(gene_s, rand_gene_s, alternative='less')
            _, p_gene_c = ttest_ind(gene_c, rand_gene_c, alternative='less')
            _, p_gene_e = ttest_ind(gene_e, rand_gene_e, alternative='less')
            _, p_nts_s = ttest_ind(nts_s, rand_nts_s, alternative='less')
            _, p_nts_c = ttest_ind(nts_c, rand_nts_c, alternative='less')
            _, p_nts_e = ttest_ind(nts_e, rand_nts_e, alternative='less')
            _, p_igr = ttest_ind(igr, rand_igr, alternative='less')
            p_values = [p_gene_s, p_gene_c, p_gene_e, p_nts_s, p_nts_c, p_nts_e, p_igr]
            palette = []
            for p in p_values:
                palette.extend([cpalette[3], cpalette[-2]] if p < p_thresh else [cpalette[0], cpalette[-2]])
            xticks = [.5, 2.5, 4.5, 6.5, 8.5, 10.5, 12.5]
            xtick_handles = ['Gene s', 'Gene c', 'Gene e', 'NTS s', 'NTS c', 'NTS e', 'IGR']
    else:
        if 'total' in save_prefix or 'notcr' in save_prefix:
            data = [
                gene_s, rand_gene_s,
                nts_s, rand_nts_s,
            ]
            _, p_gene_s = ttest_ind(gene_s, rand_gene_s, alternative='less')
            _, p_nts_s = ttest_ind(nts_s, rand_nts_s, alternative='less')
            p_values = [p_gene_s, p_nts_s]
            palette = []
            for p in p_values:
                palette.extend([cpalette[3], cpalette[-2]] if p < p_thresh else [cpalette[0], cpalette[-2]])
            xticks = [.5, 2.5]
            xtick_handles = ['Genes', 'NTS']
        else:
            data = [
                gene_s, rand_gene_s, gene_c, rand_gene_c, gene_e, rand_gene_e,
                nts_s, rand_nts_s, nts_c, rand_nts_c, nts_e, rand_nts_e
            ]
            _, p_gene_s = ttest_ind(gene_s, rand_gene_s, alternative='less')
            _, p_gene_c = ttest_ind(gene_c, rand_gene_c, alternative='less')
            _, p_gene_e = ttest_ind(gene_e, rand_gene_e, alternative='less')
            _, p_nts_s = ttest_ind(nts_s, rand_nts_s, alternative='less')
            _, p_nts_c = ttest_ind(nts_c, rand_nts_c, alternative='less')
            _, p_nts_e = ttest_ind(nts_e, rand_nts_e, alternative='less')
            p_values = [p_gene_s, p_gene_c, p_gene_e, p_nts_s, p_nts_c, p_nts_e]
            palette = []
            for p in p_values:
                palette.extend([cpalette[3], cpalette[-2]] if p < p_thresh else [cpalette[0], cpalette[-2]])
            xticks = [.5, 2.5, 4.5, 6.5, 8.5, 10.5]
            xtick_handles = ['Gene s', 'Gene c', 'Gene e', 'NTS s', 'NTS c', 'NTS e']
    ax = sns.violinplot(data=data, palette=palette)
    plt.plot([-.5, np.max(xticks) + 1], [0.5, 0.5], color=cpalette[-2], ls='--', alpha=.5)
    plt.setp(ax.collections, alpha=.2)
    sns.stripplot(data=data, palette=palette)

    plt.xticks(xticks, xtick_handles)
    plt.title('Error distribution %s' % title_biotype)
    plt.legend(custom_lines, ['Significant True', 'True', 'Random'])
    if 'gp' not in save_prefix.lower():
        plt.ylim((0, 1))

    if save_fig:
        directory = validate_dir('figures/total_error_dist')
        plt.savefig('%s/compare_error_dist_%s.png' % (directory, save_prefix))
        plt.close('all')
    else:
        plt.show()


if __name__ == '__main__':
    arguments = argparse_errorplotter(sys.argv[1:])
    main(arguments)

