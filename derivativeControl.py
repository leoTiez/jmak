#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dcor import distance_correlation

from src.DataLoader import load_chrom_data, load_chrom_split
from src.Utils import validate_dir
from src.UtilsMain import argparse_derivative, create_models


def transform_repair_data(ts_data, nts_data, igr_data, no_tcr, do_each):
    if no_tcr:
        transform_d = [
            ts_data.reshape(ts_data.shape[0], ts_data.shape[1]),
            nts_data.reshape(nts_data.shape[0], nts_data.shape[1]),
            igr_data[:, :, 0],
            igr_data[:, :, 1]
        ]
    elif do_each:
        transform_d = [
            ts_data[:, :, 0], ts_data[:, :, 1], ts_data[:, :, 2],
            nts_data[:, :, 0], nts_data[:, :, 1], nts_data[:, :, 2],
            np.mean(igr_data, axis=2)
        ]
    else:
        raise ValueError(
            'Choose either the do_each flag or the no_tcr flag.'
            ' Other setups are not supported by derivativeControl'
        )

    return transform_d


def plot_correlation(
        xr,
        comparison,
        corr_5m,
        corr_20m,
        corr_60m,
        corr_total,
        is_cpd=False,
        save_fig=False,
        plt_dc=True,
        save_prefix=''
):
    plt.figure(figsize=(10, 7))
    plt.scatter(
        xr[0],
        comparison[0],
        color='blue',
        marker='.',
        alpha=.3
    )
    plt.scatter(
        xr[1],
        comparison[1],
        color='orange',
        marker='.',
        alpha=.3
    )
    plt.scatter(
        xr[2],
        comparison[2],
        color='green',
        marker='.',
        alpha=.3
    )

    if plt_dc:
        lgd = plt.legend([
            '5m, DC: %.3f' % corr_5m,
            '20m, DC: %.3f' % corr_20m,
            '60m, DC: %.3f' % corr_60m,
            ], loc='center left', bbox_to_anchor=(1, 0.5), fontsize=24)
        title_str = 'XR vs %s\nTotal correlation: %.3f' % ('relative repair' if is_cpd else 'KJMA', corr_total)
    else:
        lgd = plt.legend(['5m', '20m', '60m'], fontsize=24)
        title_str = 'XR vs %s' % ('relative repair' if is_cpd else 'KJMA')
        print(
            '5m, DC: %.3f\t' % corr_5m,
            '20m, DC: %.3f\t' % corr_20m,
            '60m, DC: %.3f\t' % corr_60m,
            'Total correlation: %.3f' % corr_total
        )

    for lh in lgd.legendHandles:
        lh.set_alpha(1)
    plt.title(title_str, fontsize=32)
    plt.xlabel('XR', fontsize=24)
    plt.ylabel('Relative repair' if is_cpd else r'$\sqrt{d f(t)}$', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    if save_fig:
        directory = validate_dir('figures/derivative')
        plt.savefig('%s/%s_xr_vs_%s_total.png' % (directory, save_prefix, 'relative_repair' if is_cpd else 'kjma'))
        plt.close('all')
    else:
        plt.show()


def main(args):
    do_each = args.do_each
    save_fig = args.save_fig
    save_prefix = args.save_prefix
    no_tcr = args.no_tcr
    verbosity = args.verbosity
    plt_dc = args.plt_dc
    no_5m = False
    used_transcriptoms = [True, False, False] if not no_tcr else [True, True, True]
    num_bins = 3 if not no_tcr else 1

    xr_bw_paths = [
        'data/seq/xr_5m_neg.bw',
        'data/seq/xr_5m_pos.bw',
        'data/seq/xr_20m_neg.bw',
        'data/seq/xr_20m_pos.bw',
        'data/seq/xr_60m_neg.bw',
        'data/seq/xr_60m_pos.bw'
    ]

    if verbosity > 0:
        print('Load CPD')
    chrom_list = load_chrom_split()

    data = load_chrom_data(chrom_list=chrom_list, used_transcriptomes=used_transcriptoms, num_trans_bins=num_bins)
    (cpd_trans, cpd_ntrans, cpd_igr, cpd_start_igr, _, _) = data
    (xr_trans, xr_ntrans, xr_igr, xr_start_igr, _, _) = load_chrom_data(
        chrom_list=chrom_list,
        used_transcriptomes=used_transcriptoms,
        bw_list=xr_bw_paths,
        num_trans_bins=num_bins,
        is_xr=True
    )
    region_model_list = create_models(
        data,
        do_each=do_each,
        no_tcr=no_tcr,
        verbosity=verbosity,
        save_fig=save_fig,
        plot_derivative=True,
        save_prefix=save_prefix
    )

    xr_data = transform_repair_data(xr_trans, xr_ntrans, xr_igr, no_tcr=no_tcr, do_each=do_each)
    cpd_data = transform_repair_data(cpd_trans, cpd_ntrans, cpd_igr, no_tcr=no_tcr, do_each=do_each)

    region_t_deviance = []
    pred_der_5m, pred_der_20m, pred_der_60m = [], [], []
    for region_model, xr in zip(region_model_list, xr_data):
        m_values = np.asarray(list(region_model.get_model_parameter('m', do_filter=False)))
        filter_mask = ~np.isnan(m_values)
        t_deviance = []

        for r, xr_r, do_consider in zip(region_model.models, xr, filter_mask):
            if do_consider:
                rr_5 = r.repair_derivative(5)
                rr_20 = r.repair_derivative(20)
                rr_60 = r.repair_derivative(60)
                pred_der_5m.append(rr_5)
                pred_der_20m.append(rr_20)
                pred_der_60m.append(rr_60)

                xr_r = xr_r.reshape(-1)
                pred_der = np.asarray([rr_5, rr_20, rr_60])
                peak_pos_der = np.argmax(pred_der)
                peak_pos_xr = np.argmax(xr_r)
                peak_diff = peak_pos_xr - peak_pos_der
                if peak_diff > 0:
                    t_deviance.append(
                        np.sum((xr_r[peak_diff:] / np.max(xr_r)
                                - pred_der[:-peak_diff] / np.max(pred_der))**2
                               ) + peak_diff)
                elif peak_diff < 0:
                    peak_diff = np.abs(peak_diff)
                    t_deviance.append(
                        np.sum((xr_r[:-peak_diff] / np.max(xr_r)
                                - pred_der[peak_diff:] / np.max(pred_der))**2
                               ) + peak_diff)
                else:
                    t_deviance.append(np.sum((xr_r / np.max(xr_r) - pred_der / np.max(pred_der))**2))

            else:
                pred_der_5m.append(np.nan)
                pred_der_20m.append(np.nan)
                pred_der_60m.append(np.nan)

        t_deviance = np.asarray(t_deviance)
        region_t_deviance.append(t_deviance[~np.isnan(t_deviance)] / 2.)

    # Reduce increasing variance in derivative
    pred_der_5m = np.sqrt(pred_der_5m)
    pred_der_20m = np.sqrt(pred_der_20m)
    pred_der_60m = np.sqrt(pred_der_60m)
    mask = np.logical_and(~np.isnan(pred_der_5m), np.logical_and(~np.isnan(pred_der_20m), ~np.isnan(pred_der_60m)))
    mask_der = np.logical_and(pred_der_5m > 0.025, np.logical_and(pred_der_20m > 0.025, pred_der_60m > 0.025))
    mask = np.logical_and(mask, mask_der)
    total_xr = np.concatenate(xr_data)[mask, :].reshape(np.sum(mask), -1)
    total_cpd = np.concatenate(cpd_data)[mask, :].reshape(np.sum(mask), -1)
    total_cpd = np.vstack([
        total_cpd[:, 0],
        total_cpd[:, 1] - total_cpd[:, 0],
        total_cpd[:, 2] - total_cpd[:, 1]
    ])
    total_pred = np.vstack([
            pred_der_5m[mask],
            pred_der_20m[mask],
            pred_der_60m[mask]
        ])
    brown_corr_5m = distance_correlation(total_xr[:, 0], total_pred[0])
    brown_corr_20m = distance_correlation(total_xr[:, 1], total_pred[1])
    brown_corr_60m = distance_correlation(total_xr[:, 2], total_pred[2])
    brown_corr_cpd_5m = distance_correlation(total_xr[:, 0], total_cpd[0])
    brown_corr_cpd_20m = distance_correlation(total_xr[:, 1], total_cpd[1])
    brown_corr_cpd_60m = distance_correlation(total_xr[:, 2], total_cpd[2])
    if no_5m:
        brown_corr_total = distance_correlation(total_xr[:, 1:].reshape(-1), total_pred[1:].T.reshape(-1))
        brown_corr_cpd_total = distance_correlation(total_xr[:, 1:].reshape(-1), total_cpd[1:].T.reshape(-1))
    else:
        brown_corr_total = distance_correlation(total_xr.reshape(-1), total_pred.T.reshape(-1))
        brown_corr_cpd_total = distance_correlation(total_xr.reshape(-1), total_cpd.T.reshape(-1))

    plot_correlation(
        total_xr.T,
        total_pred,
        brown_corr_5m,
        brown_corr_20m,
        brown_corr_60m,
        brown_corr_total,
        is_cpd=False,
        plt_dc=plt_dc,
        save_fig=save_fig,
        save_prefix=save_prefix
    )
    plot_correlation(
        total_xr.T,
        total_cpd,
        brown_corr_cpd_5m,
        brown_corr_cpd_20m,
        brown_corr_cpd_60m,
        brown_corr_cpd_total,
        is_cpd=True,
        save_fig=save_fig,
        plt_dc=plt_dc,
        save_prefix=save_prefix
    )

    plt.figure(figsize=(8, 10))
    sns.violinplot(data=region_t_deviance)
    plt.xticks(
        np.arange(len(region_model_list)),
        [region.name for region in region_model_list],
        fontsize=24,
        rotation=45
    )
    plt.yticks(fontsize=24)
    plt.title('Temporal deviance distribution', fontsize=32)
    plt.tight_layout()
    if save_fig:
        directory = validate_dir('figures/derivative')
        plt.savefig('%s/%s_temporal_evolution_total.png' % (directory, save_prefix))
        plt.close('all')
    else:
        plt.show()


if __name__ == '__main__':
    args = argparse_derivative(sys.argv[1:])
    main(args)
