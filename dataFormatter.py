import sys
import os
import numpy as np
from src.DataLoader import trim_data
from src.UtilsMain import argparse_errorplotter


def main(args):
    array_folder = args.array_dir
    save_prefix = args.save_prefix
    max_iter = args.max_iter

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
                    and not os.path.isfile('%s/%s%s_test_All TS_error.txt' % (array_folder, save_prefix, i)):
                continue

            if 'total' in save_prefix and not 'notcr':
                gs = np.loadtxt('%s/%s%s_test_Genes total_error.txt' % (array_folder, save_prefix, i), delimiter=',')
                ntss = np.loadtxt('%s/%s%s_test_NTS total_error.txt' % (array_folder, save_prefix, i), delimiter=',')
            else:
                gs = np.loadtxt('%s/%s%s_test_All TS_error.txt' % (array_folder, save_prefix, i), delimiter=',')
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
            if not os.path.isfile('%s/%s%s_test_TCR region start (TS)_error.txt' % (array_folder, save_prefix, i)):
                continue
            gs = np.loadtxt('%s/%s%s_test_TCR region start (TS)_error.txt'
                            % (array_folder, save_prefix, i), delimiter=',')
            gc = np.loadtxt('%s/%s%s_test_TCR region centre (TS)_error.txt'
                            % (array_folder, save_prefix, i), delimiter=',')
            ge = np.loadtxt('%s/%s%s_test_TCR region end (TS)_error.txt'
                            % (array_folder, save_prefix, i), delimiter=',')

            ntss = np.loadtxt('%s/%s%s_test_TCR region start (NTS)_error.txt'
                              % (array_folder, save_prefix, i), delimiter=',')
            ntsc = np.loadtxt('%s/%s%s_test_TCR region centre (NTS)_error.txt'
                              % (array_folder, save_prefix, i), delimiter=',')
            ntse = np.loadtxt('%s/%s%s_test_TCR region end (NTS)_error.txt'
                              % (array_folder, save_prefix, i), delimiter=',')

            if 'size' not in save_prefix.lower() and 'netseq' not in save_prefix.lower():
                intergen = np.loadtxt(
                    '%s/%s%s_test_Non-TCR region_error.txt' % (array_folder, save_prefix, i),
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
    rand_igr_minus = np.empty(0)
    i = 0
    while True and i < max_iter:
        i += 1
        if 'total' in save_prefix or 'notcr' in save_prefix:
            if not os.path.isfile('%s/%s%s_random_test_Genes total_error.txt' % (array_folder, save_prefix, i)) \
                    and not os.path.isfile('%s/%s%s_random_test_All TS_error.txt' % (array_folder, save_prefix, i)):
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
                    '%s/%s%s_random_test_All TS_error.txt'
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
            if not os.path.isfile('%s/%s%s_random_test_TCR region start (TS)_error.txt' % (array_folder, save_prefix, i)):
                continue
            randg_s = np.loadtxt('%s/%s%s_random_test_TCR region start (TS)_error.txt' % (array_folder, save_prefix, i),
                                 delimiter=',')
            randg_c = np.loadtxt('%s/%s%s_random_test_TCR region centre (TS)_error.txt' % (array_folder, save_prefix, i),
                                 delimiter=',')
            randg_e = np.loadtxt('%s/%s%s_random_test_TCR region end (TS)_error.txt' % (array_folder, save_prefix, i),
                                 delimiter=',')

            randnts_s = np.loadtxt('%s/%s%s_random_test_TCR region start (NTS)_error.txt' % (array_folder, save_prefix, i),
                                   delimiter=',')
            randnts_c = np.loadtxt('%s/%s%s_random_test_TCR region centre (NTS)_error.txt' % (array_folder, save_prefix, i),
                                   delimiter=',')
            randnts_e = np.loadtxt('%s/%s%s_random_test_TCR region end (NTS)_error.txt' % (array_folder, save_prefix, i),
                                   delimiter=',')

            if 'size' not in save_prefix.lower() and 'netseq' not in save_prefix.lower():
                randigr = np.loadtxt(
                    '%s/%s%s_random_test_Non-TCR region_error.txt' % (array_folder, save_prefix, i),
                    delimiter=','
                )
                rand_igr = np.append(rand_igr, randigr)
            rand_gene_c = np.append(rand_gene_c, randg_c)
            rand_gene_e = np.append(rand_gene_e, randg_e)
            rand_nts_c = np.append(rand_nts_c, randnts_c)
            rand_nts_e = np.append(rand_nts_e, randnts_e)

        rand_gene_s = np.append(rand_gene_s, randg_s)
        rand_nts_s = np.append(rand_nts_s, randnts_s)

    np.savetxt('%s/all_%s_gene_start_error.txt' % (array_folder, save_prefix), gene_s, delimiter=',')
    if 'total' not in save_prefix.lower() and 'notcr' not in save_prefix.lower():
        np.savetxt('%s/all_%s_gene_centre_error.txt' % (array_folder, save_prefix), gene_c, delimiter=',')
        np.savetxt('%s/all_%s_gene_end_error.txt' % (array_folder, save_prefix), gene_e, delimiter=',')
    np.savetxt('%s/all_%s_random_gene_start_error.txt' % (array_folder, save_prefix), rand_gene_s, delimiter=',')
    if 'total' not in save_prefix.lower() and 'notcr' not in save_prefix.lower():
        np.savetxt('%s/all_%s_random_gene_centre_error.txt' % (array_folder, save_prefix), rand_gene_c, delimiter=',')
        np.savetxt('%s/all_%s_random_gene_end_error.txt' % (array_folder, save_prefix), rand_gene_e, delimiter=',')

    np.savetxt('%s/all_%s_nts_start_error.txt' % (array_folder, save_prefix), nts_s, delimiter=',')
    if 'total' not in save_prefix.lower() and 'notcr' not in save_prefix.lower():
        np.savetxt('%s/all_%s_nts_centre_error.txt' % (array_folder, save_prefix), nts_c, delimiter=',')
        np.savetxt('%s/all_%s_nts_end_error.txt' % (array_folder, save_prefix), nts_e, delimiter=',')
    np.savetxt('%s/all_%s_random_nts_start_error.txt' % (array_folder, save_prefix), rand_nts_s, delimiter=',')
    if 'total' not in save_prefix.lower() and 'notcr' not in save_prefix.lower():
        np.savetxt('%s/all_%s_random_nts_centre_error.txt' % (array_folder, save_prefix), rand_nts_c, delimiter=',')
        np.savetxt('%s/all_%s_random_nts_end_error.txt' % (array_folder, save_prefix), rand_nts_e, delimiter=',')

    if 'size' not in save_prefix.lower() and 'netseq' not in save_prefix.lower():
        np.savetxt('%s/all_%s_igr_error.txt' % (array_folder, save_prefix), igr, delimiter=',')
        np.savetxt('%s/all_%s_random_igr_error.txt' % (array_folder, save_prefix), rand_igr, delimiter=',')
        if 'total' in save_prefix.lower() and 'notcr' in save_prefix.lower():
            np.savetxt('%s/all_%s_igr_minus_error.txt' % (array_folder, save_prefix), igr_minus, delimiter=',')
            np.savetxt('%s/all_%s_random_igr_minus_error.txt'
                       % (array_folder, save_prefix), rand_igr_minus, delimiter=',')


if __name__ == '__main__':
    main(argparse_errorplotter(sys.argv[1:]))

