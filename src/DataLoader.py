import os
import re
import numpy as np
import pandas as pd
from datahandler import reader


PY_DIM_POS = ['TT', 'CT', 'TC', 'CC']
PY_DIM_NEG = ['AA', 'GA', 'AG', 'GG']


def trim_data(data, rm_percentile=5, only_upper=True, return_mask=True):
    if only_upper:
        lower, upper = np.percentile(data, [0, 100. - rm_percentile])
    else:
        lower, upper = np.percentile(data, [rm_percentile / 2., 100. - rm_percentile / 2.])
    mask = np.logical_and(data > lower, data < upper)
    if return_mask:
        return mask
    else:
        return data[mask]


def transform_path(path_string):
    if path_string[:5] != '/home' and path_string[0] != '~':
        path_prefix = os.getcwd()
        path = '%s/%s' % (path_prefix, path_string)
    else:
        if path_string[0] == '~':
            path = os.path.expanduser(path_string)
        else:
            path = path_string

    return os.path.abspath(path)


def load_chrom_split(path='chrom.data', return_all=True, data_type=''):
    path = transform_path(path)
    cs_df = pd.read_csv(path, sep='\t')
    if return_all:
        return cs_df[cs_df['chr'] != 'chrM']['chr'].to_list()
    else:
        cs_df_type = cs_df[cs_df['data_type'] == data_type]
        return cs_df_type['chr'].to_list()


def load_meres(identifier='centromeres', path=None):
    if identifier == 'centromeres':
        path = path if path is not None else 'data/ref/centromeres.bed'
        centromeres_path = transform_path(path)
        meres = pd.read_csv(
            centromeres_path,
            sep='\t',
            names=['chr', 'start', 'end', 'name', 'score', 'strand']
        )
    elif identifier == 'telomeres':
        path = path if path is not None else 'data/ref/telomeres.bed'
        telomeres_path = transform_path(path)
        meres = pd.read_csv(
            telomeres_path,
            sep='\t',
            names=['chr', 'start', 'end', 'name', 'score', 'strand']
        )
    else:
        raise ValueError('Meres identifier not understood.')

    c = pd.DataFrame(meres['chr'])
    c['pos'] = (meres['start'] + meres['end']) / 2.
    return c


def load_transcriptome(
        high_t_path='data/ref/trans_high.txt',
        medium_t_path='data/ref/trans_medium.txt',
        low_t_path='data/ref/trans_low.txt',
):
    def remove_dupl(df):
        df = df.drop(columns=['Unnamed: 0', 'rr'])
        return df.drop_duplicates()

    high_t_path = transform_path(high_t_path)
    high_t = remove_dupl(pd.read_csv(high_t_path))

    medium_t_path = transform_path(medium_t_path)
    medium_t = remove_dupl(pd.read_csv(medium_t_path))

    low_t_path = transform_path(low_t_path)
    low_t = remove_dupl(pd.read_csv(low_t_path))
    return high_t, medium_t, low_t


def num_pydim_pos(chrom, bins, direct, dna_seq):
    chrom_seq = list(filter(lambda x: x.id == chrom, dna_seq))[0]
    combinations = PY_DIM_POS if direct == '+' else PY_DIM_NEG
    seq = chrom_seq.seq
    num_pydim = []
    num_overlap = []
    for i in range(len(bins) - 1):
        ind = set()
        for c in combinations:
            c_ind = set([m.start() for m in re.finditer(c, str(seq[bins[i]:bins[i+1]]))])
            ind = ind.union(c_ind)
        num_overlap.append(len(list(ind.intersection(np.asarray(list(ind)) + 1))))
        num_pydim.append(len(list(ind)))

    return np.asarray(num_pydim), np.asarray(num_overlap)


def normalise_data(trans, chrom, data, ref_genome, num_bins=3):
    def get_data(data, start, end, read, num_bins_=3):
        bins = np.round(np.linspace(start, end, num_bins_ + 1)).astype('int')
        if read == '+':
            denom, _ = num_pydim_pos(chrom, bins, '+', ref_genome)
        else:
            denom, _ = num_pydim_pos(chrom, bins, '-', ref_genome)
        denom = 2 * denom

        if np.any(denom == 0):
            return [-1] * num_bins_, [-1] * num_bins_, [-1] * num_bins_

        add_last_value = False
        if bins[-1] == len(data[0]):
            bins[-1] -= 1
            add_last_value = True
        if read == '+':
            trans_mask_p[start:end] = True
            bin_sum0h_1 = np.add.reduceat(data[1], bins)
            bin_sum0h_2 = np.add.reduceat(data[5], bins)
            bin_sum20m = np.add.reduceat(data[7], bins)
            bin_sum1h = np.add.reduceat(data[3], bins)
            bin_sum2h = np.add.reduceat(data[9], bins)
        else:
            trans_mask_m[start:end] = True
            bin_sum0h_1 = np.add.reduceat(data[0], bins)
            bin_sum0h_2 = np.add.reduceat(data[4], bins)
            bin_sum20m = np.add.reduceat(data[6], bins)
            bin_sum1h = np.add.reduceat(data[2], bins)
            bin_sum2h = np.add.reduceat(data[8], bins)

        if add_last_value:
            bin_sum0h_1[-2] += bin_sum0h_1[-1]
            bin_sum0h_2[-2] += bin_sum0h_2[-1]
            bin_sum20m[-2] += bin_sum20m[-1]
            bin_sum1h[-2] += bin_sum1h[-1]
            bin_sum2h[-2] += bin_sum2h[-1]

        bin_sum0h_1 = bin_sum0h_1[:-1]
        bin_sum0h_2 = bin_sum0h_2[:-1]
        bin_sum20m = bin_sum20m[:-1]
        bin_sum1h = bin_sum1h[:-1]
        bin_sum2h = bin_sum2h[:-1]

        cpd_0h_1 = np.nan_to_num(bin_sum0h_1 / denom)
        cpd_0h_2 = np.nan_to_num(bin_sum0h_2 / denom)
        cpd_20m = np.nan_to_num(bin_sum20m / denom)
        cpd_1h = np.nan_to_num(bin_sum1h / denom)
        cpd_2h = np.nan_to_num(bin_sum2h / denom)

        if read == '+':
            cpd_0h_1 = np.flip(cpd_0h_1)
            cpd_0h_2 = np.flip(cpd_0h_2)
            cpd_20m = np.flip(cpd_20m)
            cpd_1h = np.flip(cpd_1h)
            cpd_2h = np.flip(cpd_2h)

        rel_20m = (cpd_0h_2 - cpd_20m) / cpd_0h_2
        rel_1h = (cpd_0h_1 - cpd_1h) / cpd_0h_1
        rel_2h = (cpd_0h_2 - cpd_2h) / cpd_0h_2

        # Enforce progressing repair. Negative repair is not possible
        rel_20m = np.maximum(0, rel_20m)
        rel_1h = np.maximum(rel_20m, rel_1h)
        rel_2h = np.minimum(np.maximum(rel_1h, rel_2h), 1)

        return rel_20m, rel_1h, rel_2h

    def get_start_end_intergenic(intergenic_mask, borders):
        if intergenic_mask[0] == 0:
            igr_start = borders[::2]
            igr_end = borders[1::2]
        else:
            igr_start = borders[1::2]
            igr_end = borders[::2]
            igr_start = np.insert(igr_start, 0, 0)
            igr_end = np.append(igr_end, intergenic_mask.size - 1)
        # igr_start, igr_end = igr_start + 1, igr_end + 1
        return igr_start, igr_end
    trans_norm_20, trans_norm_60, trans_norm_120 = [], [], []
    non_trans_norm_20, non_trans_norm_60, non_trans_norm_120 = [], [], []
    trans_mask_p = np.zeros(data[0].size).astype('bool')
    trans_mask_m = np.zeros(data[0].size).astype('bool')
    for num, i in enumerate(trans.iterrows()):
        i = i[1]
        # zero-based array vs one-based notation
        start = i['start'] - 1
        end = i['end'] - 1
        non_read, read = '+', '-'
        if start > end:
            end_temp = start
            start = end
            end = end_temp
            non_read, read = '-', '+'

        rel_20, rel_60, rel_120 = get_data(data, start, end, read, num_bins_=num_bins)
        non_rel_20, non_rel_60, non_rel_120 = get_data(data, start, end, non_read, num_bins_=num_bins)
        trans_norm_20.append(rel_20)
        trans_norm_60.append(rel_60)
        trans_norm_120.append(rel_120)
        non_trans_norm_20.append(non_rel_20)
        non_trans_norm_60.append(non_rel_60)
        non_trans_norm_120.append(non_rel_120)

    trans_mask_ensemle = np.logical_or(trans_mask_m, trans_mask_p)
    borders_ensemble = np.where(~trans_mask_ensemle != np.roll(~trans_mask_ensemle, shift=-1))[0]

    ig_norm_20_ens, ig_norm_60_ens, ig_norm_120_ens = [], [], []
    start_igr_ensemble, end_igr_ensemble = get_start_end_intergenic(~trans_mask_ensemle, borders_ensemble)
    for s, e in zip(start_igr_ensemble, end_igr_ensemble):
        rel_20_m, rel_60_m, rel_120_m = get_data(data, s, e, '-', num_bins_=1)
        rel_20_p, rel_60_p, rel_120_p = get_data(data, s, e, '+', num_bins_=1)
        ig_norm_20_ens.append(rel_20_p)
        ig_norm_60_ens.append(rel_60_p)
        ig_norm_120_ens.append(rel_120_p)
        ig_norm_20_ens.append(rel_20_m)
        ig_norm_60_ens.append(rel_60_m)
        ig_norm_120_ens.append(rel_120_m)

    return (
               np.asarray(trans_norm_20), np.asarray(trans_norm_60), np.asarray(trans_norm_120),
               np.asarray(non_trans_norm_20), np.asarray(non_trans_norm_60), np.asarray(non_trans_norm_120),
               np.asarray(ig_norm_20_ens), np.asarray(ig_norm_60_ens), np.asarray(ig_norm_120_ens),
           ), (start_igr_ensemble, end_igr_ensemble)


def load_chrom_data(
        chrom_list=['chrIII'],
        bw_list=[
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
        ],
        transcriptome_path_list=[
            'data/ref/trans_high.txt',
            'data/ref/trans_medium.txt',
            'data/ref/trans_low.txt',
        ],
        used_transcriptomes=[True, False, False],
        num_trans_bins=3,
        ref_genome_path='data/ref/SacCer3.fa',
        shuffle_data=False,
):
    def get_data(chrom_list):
        trans, non_trans, igr = [], [], []
        start_igr_list, end_igr_list = [], []
        data_transcriptome = pd.DataFrame(columns=ht.columns)

        for chrom in chrom_list:
            chrom_data = [np.nan_to_num(d.values(chrom, 0, d.chroms(chrom))) for d in bw_objs]
            transcriptome_chrom = transcriptome[transcriptome['chr'] == chrom]
            (
                (t_20, t_60, t_120, nt_20, nt_60, nt_120, igr_20, igr_60, igr_120),
                (start_igr, end_igr)
            ) = normalise_data(transcriptome_chrom, chrom, chrom_data, ref_genome, num_bins=num_trans_bins)
            trans.extend([t_20, t_60, t_120])
            non_trans.extend([nt_20, nt_60, nt_120])
            igr.extend([igr_20, igr_60, igr_120])
            start_igr_list.extend([(chrom, int(s)) for s in start_igr])
            end_igr_list.extend([(chrom, int(e)) for e in end_igr])
            data_transcriptome = data_transcriptome.append(transcriptome_chrom)

        trans_20 = [i for k in trans[::3] for i in k]
        trans_60 = [i for k in trans[1::3] for i in k]
        trans_120 = [i for k in trans[2::3] for i in k]
        trans = np.asarray([trans_20, trans_60, trans_120])

        ntrans_20 = [i for k in non_trans[::3] for i in k]
        ntrans_60 = [i for k in non_trans[1::3] for i in k]
        ntrans_120 = [i for k in non_trans[2::3] for i in k]
        non_trans = np.asarray([ntrans_20, ntrans_60, ntrans_120])

        igr_20 = [i for k in igr[::3] for i in k]
        igr_60 = [i for k in igr[1::3] for i in k]
        igr_120 = [i for k in igr[2::3] for i in k]
        # Pair plus and minus into one data point
        igr = np.asarray([igr_20, igr_60, igr_120]).reshape(3, -1, 2)

        data_transcriptome = data_transcriptome.reset_index(drop=True)
        shuffle_idx_trans = np.arange(len(data_transcriptome.index))
        shuffle_idx_igr = np.arange(len(start_igr_list))
        if shuffle_data:
            np.random.shuffle(shuffle_idx_trans)
            np.random.shuffle(shuffle_idx_igr)

        return (
            trans.transpose(1, 0, 2)[shuffle_idx_trans],
            non_trans.transpose(1, 0, 2)[shuffle_idx_trans],
            igr.transpose(1, 0, 2)[shuffle_idx_igr],
            np.asarray(start_igr_list)[shuffle_idx_igr].tolist(),
            np.asarray(end_igr_list)[shuffle_idx_igr].tolist(),
            data_transcriptome.reindex(shuffle_idx_trans)
        )

    # Load data
    for num, path in enumerate(bw_list):
        bw_list[num] = transform_path(path)
    for num, path in enumerate(transcriptome_path_list):
        transcriptome_path_list[num] = transform_path(path)
    ref_genome_path = transform_path(ref_genome_path)

    ht, mt, lt = load_transcriptome(*transcriptome_path_list)
    transcriptome = pd.DataFrame(columns=ht.columns)
    for is_transcriptome, t in zip(used_transcriptomes, [ht, mt, lt]):
        if is_transcriptome:
            transcriptome = transcriptome.append(t)
    transcriptome = transcriptome.drop_duplicates()
    ref_genome = reader.load_fast(ref_genome_path, is_abs_path=True, is_fastq=False)
    bw_objs = []
    for bw in bw_list:
        bw_objs.append(reader.load_big_file(bw, rel_path='', is_abs_path=True))

    return get_data(chrom_list)


def create_time_data(num_pos, num_datapoints):
    return np.tile(np.concatenate((
        np.ones(num_pos) * 20,
        np.ones(num_pos) * 60,
        np.ones(num_pos) * 120
    ), axis=None), num_datapoints).reshape(num_datapoints, -1)


def load_bio_data(
        chrom_starti_tuple_list,
        chrom_endi_tuple_list,
        bw_paths,
        shuffle_data=False,
        use_directionality=False,
        use_sum=False
):
    def get_data():
        num_data_lists = len(bw_paths) if not use_directionality else len(bw_paths) // 2
        data_list = [[] for _ in range(num_data_lists)]
        old_chrom = ''
        for (chrom_s, start), (chrom_e, end) in zip(chrom_starti_tuple_list, chrom_endi_tuple_list):
            start, end = int(start), int(end)
            if chrom_s != chrom_e:
                raise ValueError('Chromosomes of start and end index do not match')
            if old_chrom != chrom_s:
                all_chrom_data = [np.nan_to_num(d.values(chrom_s, 0, d.chroms(chrom_s))) for d in bw_objs]
                old_chrom = chrom_s
            read = '+'
            if start > end:
                end_temp = start
                start = end
                end = end_temp
                read = '-'
            if use_directionality:
                if read == '-':
                    chrom_data = all_chrom_data[::2]
                else:
                    chrom_data = all_chrom_data[1::2]
            else:
                chrom_data = all_chrom_data

            fun = np.nanmean if not use_sum else np.nansum
            for num, cd in enumerate(chrom_data):
                data_list[num].append(fun(cd[start:end]))

        shuffle_idx = np.arange(len(data_list[0]))
        if shuffle_data:
            np.random.shuffle(shuffle_idx)

        return [np.asarray(d)[shuffle_idx] for d in data_list]

    for num, path in enumerate(bw_paths):
        bw_paths[num] = transform_path(path)

    bw_objs = []
    for bw in bw_paths:
        bw_objs.append(reader.load_big_file(bw, rel_path='', is_abs_path=True))

    # Prepare train data
    return get_data()

