import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.stats import ttest_ind

from src.UtilsMain import argparse_errorplotter
from src.Utils import validate_dir


def main(args):
    array_folder = args.array_dir
    save_fig = args.save_fig
    p_thresh = args.pthresh

    xticks = [.5, 2.5, 4.5, 6.5, 8.5, 10.5, 12.5]
    cpalette = sns.color_palette()
    custom_lines = [Line2D([0], [0], color=cpalette[3], lw=4),
                    Line2D([0], [0], color=cpalette[0], lw=4),
                    Line2D([0], [0], color=cpalette[-2], lw=4)]

    bio_types = ['netseq', 'nucl', 'abf1', 'h2a', 'size', 'rel_meres']
    for setup in range(4):
        if setup > 2:
            mode_prefix = 'mode_'
        else:
            mode_prefix = ''
        if setup % 2:
            notcr_prefix = ''
            type_string = 'each'
        else:
            notcr_prefix = 'notcr_'
            type_string = 'total'

        for knn in [5, 10, 20, 50, 100]:
            for bt in bio_types:
                if bt == 'netseq':
                    bt_name = 'NET-seq'
                elif bt == 'nucl':
                    bt_name = 'Nucleosome density'
                elif bt == 'abf1':
                    bt_name = 'Abf1'
                elif bt == 'h2a':
                    bt_name = 'H2A.Z'
                elif bt == 'size':
                    bt_name = 'Size'
                elif bt == 'rel_meres':
                    bt_name = 'Relative meres'

                save_prefix = '%s/all_%s%s%s_knn%s_%s' % (array_folder, mode_prefix, notcr_prefix, bt, knn, type_string)
                filtered_files = glob.glob('%s*' % save_prefix)

                gene_s = np.empty(0)
                gene_c = np.empty(0)
                gene_e = np.empty(0)

                nts_s = np.empty(0)
                nts_c = np.empty(0)
                nts_e = np.empty(0)

                igr = np.empty(0)
                igr_minus = np.empty(0)

                rand_gene_s = np.empty(0)
                rand_gene_c = np.empty(0)
                rand_gene_e = np.empty(0)

                rand_nts_s = np.empty(0)
                rand_nts_c = np.empty(0)
                rand_nts_e = np.empty(0)

                rand_igr = np.empty(0)
                rand_igr_minus = np.empty(0)

                plt.figure(figsize=(8, 7))
                print(save_prefix)

                for ff in filtered_files:
                    values = np.loadtxt(ff)
                    if 'random' not in ff.lower():
                        if 'gene_start' in ff.lower():
                            gene_s = np.append(gene_s, values)
                        elif 'gene_centre' in ff.lower():
                            gene_c = np.append(gene_c, values)
                        elif 'gene_end' in ff.lower():
                            gene_e = np.append(gene_e, values)
                        elif 'nts_start' in ff.lower():
                            nts_s = np.append(nts_s, values)
                        elif 'nts_centre' in ff.lower():
                            nts_c = np.append(nts_c, values)
                        elif 'nts_end' in ff.lower():
                            nts_e = np.append(nts_e, values)
                        elif 'igr_minus' in ff.lower():
                            igr_minus = np.append(igr_minus, values)
                        elif 'igr' in ff.lower():
                            igr = np.append(igr, values)
                    else:
                        if 'gene_start' in ff.lower():
                            rand_gene_s = np.append(rand_gene_s, values)
                        elif 'gene_centre' in ff.lower():
                            rand_gene_c = np.append(rand_gene_c, values)
                        elif 'gene_end' in ff.lower():
                            rand_gene_e = np.append(rand_gene_e, values)
                        elif 'nts_start' in ff.lower():
                            rand_nts_s = np.append(rand_nts_s, values)
                        elif 'nts_centre' in ff.lower():
                            rand_nts_c = np.append(rand_nts_c, values)
                        elif 'nts_end' in ff.lower():
                            rand_nts_e = np.append(rand_nts_e, values)
                        elif 'igr_minus' in ff.lower():
                            rand_igr_minus = np.append(rand_igr_minus, values)
                        elif 'igr' in ff.lower():
                            rand_igr = np.append(rand_igr, values)
                data = []
                palette = []
                for d, rd in zip(
                        [gene_s, gene_c, gene_e, nts_s, nts_c, nts_e, igr, igr_minus],
                        [rand_gene_s, rand_gene_c, rand_gene_e, rand_nts_s, rand_nts_c, rand_nts_e, rand_igr, rand_igr_minus]
                ):
                    if len(d) > 0 and len(rd) > 0:
                        data.extend([d, rd])
                        _, p_value = ttest_ind(d, rd, alternative='less')
                        better_than_cf = np.sum(d < .5) / d.size >= .9
                        palette.extend([cpalette[3], cpalette[-2]] if p_value < p_thresh and better_than_cf
                                       else [cpalette[0], cpalette[-2]])

                if not data:
                    continue

                xticks_temp = xticks[:len(data) // 2]
                if type_string == 'each':
                    if bt in ['size', 'netseq']:
                        xtick_labels = ['TS S', 'TS C', 'TS E', 'NTS S', 'NTS C', 'NTS E']
                    else:
                        xtick_labels = ['TS S', 'TS C', 'TS E', 'NTS S', 'NTS C', 'NTS E', 'IGR']
                else:
                    if bt in ['size', 'netseq']:
                        xtick_labels = ['TS', 'NTS']
                    else:
                        xtick_labels = ['TS', 'NTS', 'IGR +', 'IGR -']

                ax = sns.violinplot(data=data, palette=palette)
                plt.plot([-.5, np.max(xticks_temp) + 1], [0.5, 0.5], color=cpalette[-2], ls='--', alpha=.5)
                plt.setp(ax.collections, alpha=.2)
                sns.stripplot(data=data, palette=palette)

                plt.xticks(xticks_temp, xtick_labels, fontsize=16)
                title_mode = 'Mode' if mode_prefix == 'mode_' else ''
                plt.title('%s %s | kNN%s' % (title_mode, bt_name, knn), fontsize=30)
                plt.legend(custom_lines, ['Significant True', 'True', 'Random'], fontsize=16)
                plt.ylim((0, 1))
                plt.yticks(fontsize=16)
                if save_fig:
                    directory = validate_dir('figures/total_error_dist')
                    save_prefix = '%s%s%s_knn%s_%s' % (mode_prefix, notcr_prefix, bt, knn, type_string)
                    plt.savefig('%s/compare_error_dist_%s.png' % (directory, save_prefix))
                    plt.close('all')
                else:
                    plt.show()


if __name__ == '__main__':
    arguments = argparse_errorplotter(sys.argv[1:])
    main(arguments)

