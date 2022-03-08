import sys
import numpy as np
import pandas as pd
import warnings

from src.Model import RegionModel
from src.UtilsMain import argparse_kjmapredict_param
from src.DataLoader import transform_path, num_pydim_pos, create_time_data

from datahandler import reader


def load_data(bw_list, chrom, start, end, ref_genome_path):
    def get_repair(cpd_data):
        rr_60m = (cpd_data[0] - cpd_data[1]) / cpd_data[0]
        rr_20m = (cpd_data[2] - cpd_data[3]) / cpd_data[2]
        rr_120m = (cpd_data[2] - cpd_data[4]) / cpd_data[2]

        rr_20m = np.maximum(0, rr_20m)
        rr_60m = np.maximum(rr_20m, rr_60m)
        rr_120m = np.minimum(np.maximum(rr_60m, rr_120m), 1)
        return rr_20m, rr_60m, rr_120m

    for num, path in enumerate(bw_list):
        bw_list[num] = transform_path(path)

    ref_genome_path = transform_path(ref_genome_path)
    ref_genome = reader.load_fast(ref_genome_path, is_abs_path=True, is_fastq=False)
    denom_m, _ = num_pydim_pos(chrom, np.asarray([start, end]), '-', ref_genome)
    denom_p, _ = num_pydim_pos(chrom, np.asarray([start, end]), '+', ref_genome)
    denom_m = 2 * denom_m
    denom_p = 2 * denom_p

    if denom_m == 0:
        warnings.warn('Negative strand does not possess any pyrimidine dimers in given coordinates')
    if denom_p == 0:
        warnings.warn('Positive strand does not possess any pyrimidine dimers in given coordinates')

    data_values = []
    for num, bw_path in enumerate(bw_list):
        bw = reader.load_big_file(bw_path, rel_path='', is_abs_path=True)
        v = np.nansum(bw.values(chrom, start, end))
        if 'minus' in bw_path.lower():
            v /= denom_m
        elif 'plus' in bw_path.lower():
            v /= denom_p
        data_values.append(v)

    neg_v = get_repair(data_values[::2])
    pos_v = get_repair(data_values[1::2])
    return neg_v, pos_v


def main(args):
    chrom = args.chrom
    start = args.start
    end = args.end
    pred_t = np.asarray(args.t)
    min_f = args.min_f
    max_f = args.max_f
    delta_f = args.delta_f
    save_fig = args.save_fig
    save_prefix = args.save_prefix
    num_cpus = args.num_cpus
    verbosity = args.verbosity

    bw_list = [
        'data/seq/0h_A1_minus.bw',
        'data/seq/0h_A1_plus.bw',
        'data/seq/1h_A1_minus.bw',
        'data/seq/1h_A1_plus.bw',
        'data/seq/0h_A2_minus.bw',
        'data/seq/0h_A2_plus.bw',
        'data/seq/20m_A2_minus.bw',
        'data/seq/20m_A2_plus.bw',
        'data/seq/2h_A2_minus.bw',
        'data/seq/2h_A2_plus.bw'
    ]
    ref_genome_path = 'data/ref/SacCer3.fa'

    neg_v, pos_v = load_data(bw_list, chrom, start, end, ref_genome_path)

    region_name = '%s:%s-%s' % (chrom, start, end)
    region_model = RegionModel(
        create_time_data(1, 2),
        np.asarray([neg_v, pos_v]).reshape(2, -1),
        name=region_name
    )

    region_model.fit_models(
        min_f=min_f,
        max_f=max_f,
        delta_f=delta_f,
        names=['%s -' % region_name, '%s +' % region_name],
        verbosity=verbosity,
        num_cpus=num_cpus,
        save_fig=save_fig,
        save_prefix=save_prefix
    )

    neg_mod, pos_mod = region_model.models
    print('### Negative strand:\tm: %.3f\t1/tau: %.3f\ttheta: %.3f' % (neg_mod.m, neg_mod.beta, neg_mod.max_frac))
    print('### Positive strand:\tm: %.3f\t1/tau: %.3f\ttheta: %.3f' % (pos_mod.m, pos_mod.beta, pos_mod.max_frac))

    frac_neg = neg_mod.repair_fraction(pred_t)
    der_neg = neg_mod.repair_derivative(pred_t)

    frac_pos = pos_mod.repair_fraction(pred_t)
    der_pos = pos_mod.repair_derivative(pred_t)

    print('### Save repair predictions to file %s%s.tsv' % (save_prefix, region_name))

    df = pd.DataFrame(
        [
            frac_neg,
            der_neg,
            frac_pos,
            der_pos
        ],
        columns=pred_t.tolist()
    )
    df['name'] = ['Repair fraction -', 'Repair derivative -', 'Repair fraction +', 'Repair derivative +']
    df.to_csv('%s_%s.tsv' % (save_prefix, region_name), sep='\t')


if __name__ == '__main__':
    args = argparse_kjmapredict_param(sys.argv[1:])
    main(args)


