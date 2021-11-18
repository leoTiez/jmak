import os
import re
import numpy as np
import pandas as pd
from datahandler import reader


PY_DIM_POS = ['TT', 'CT', 'TC', 'CC']
PY_DIM_NEG = ['AA', 'GA', 'AG', 'GG']


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


def load_chrom_split(data_type='train', path='chrom.data'):
    path = transform_path(path)
    cs_df = pd.read_csv(path, sep='\t')
    cs_df_type = cs_df[cs_df['data_type'] == data_type]
    return cs_df_type['chr'].to_list()


def load_centromeres(centromeres_path='data/ref/centromeres.bed'):
    centromeres_path = transform_path(centromeres_path)
    centromeres = pd.read_csv(
        centromeres_path,
        sep='\t',
        names=['chr', 'start', 'end', 'name', 'score', 'strand']
    )
    c = pd.DataFrame(centromeres['chr'])
    c['pos'] = (centromeres['start'] + centromeres['end']) / 2.
    return c


def load_transcriptome(
        high_t_path='data/ref/trans_high.txt',
        medium_t_path='data/ref/trans_medium.txt',
        low_t_path='data/ref/trans_low.txt',
):
    def remove_dupl(df):
        df = df.drop(columns=['Unnamed: 0', 'rr', 'pos'])
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
        rel_2h = np.maximum(rel_1h, rel_2h)
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
        igr_start, igr_end = igr_start + 1, igr_end + 1
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
        train_chrom_list=['chrIII'],
        test_chrom_list=['chrII'],
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
        ref_genome_path='data/ref/SacCer3.fa',
        seed=42
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
            ) = normalise_data(transcriptome_chrom, chrom, chrom_data, ref_genome)
            trans.extend([t_20, t_60, t_120])
            non_trans.extend([nt_20, nt_60, nt_120])
            igr.extend([igr_20, igr_60, igr_120])
            start_igr_list.extend([(chrom, s) for s in start_igr])
            end_igr_list.extend([(chrom, e) for e in end_igr])
            data_transcriptome = data_transcriptome.append(transcriptome_chrom)

        data_transcriptome = data_transcriptome.reset_index(drop=True)
        shuffle_idx_trans = np.arange(len(data_transcriptome.index))
        shuffle_idx_igr = np.arange(len(start_igr_list))
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(shuffle_idx_trans)
        np.random.shuffle(shuffle_idx_igr)
        return (
            np.asarray(trans).transpose(1, 0, 2)[shuffle_idx_trans].reshape(len(shuffle_idx_trans), -1),
            np.asarray(non_trans).transpose(1, 0, 2)[shuffle_idx_trans].reshape(len(shuffle_idx_trans), -1),
            np.asarray(igr).transpose(1, 0, 2)[shuffle_idx_igr].reshape(len(shuffle_idx_igr), -1),
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
    ref_genome = reader.load_fast(ref_genome_path, is_abs_path=True, is_fastq=False)
    bw_objs = []
    for bw in bw_list:
        bw_objs.append(reader.load_big_file(bw, rel_path='', is_abs_path=True))

    # Prepare train data
    train_data = get_data(train_chrom_list)

    # Prepare test data
    if test_chrom_list:
        test_data = get_data(test_chrom_list)
    else:
        test_data = None

    return train_data, test_data


def create_time_data(num_pos, num_datapoints):
    return np.tile(np.concatenate((
        np.ones(num_pos) * 20,
        np.ones(num_pos) * 60,
        np.ones(num_pos) * 120
    ), axis=None), num_datapoints).reshape(num_datapoints, -1)

