#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression, LinearRegression

from src.DataLoader import load_chrom_split, load_chrom_data, create_time_data
from src.UtilsMain import argparse_model_comp_param, create_models
from src.Model import Hill


def sample_data(repair, n_dp=1000):
    return np.stack([np.random.binomial(1, p=r, size=n_dp) for r in repair]).T


def plot_error_dist(kjma_score, hill_score, log_score, lin_score, name='', save_fig=True, save_prefix=''):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.gca()
    ax.violinplot([kjma_score, hill_score, log_score, lin_score], showmedians=True)
    for num, med in enumerate(
            [np.median(kjma_score), np.median(hill_score), np.median(log_score), np.median(lin_score)]):
        plt.text(num + 1.05, med + med * .2, '%.5f' % med, fontsize=16)
    ax.set_xticks(np.arange(1, 5))
    ax.set_xticklabels(['KJMA', 'Hill', 'Logistic', 'Linear'], fontsize=18)
    ax.set_ylabel('MSE', fontsize=18)
    ax.tick_params(axis='y', labelsize=16)

    fig.suptitle('MSE %s' % name, fontsize=32)
    plt.tight_layout()
    if save_fig:
        Path('figures/comp_models').mkdir(parents=True, exist_ok=True)
        plt.savefig('figures/comp_models/%s_%s_error.png' % (save_prefix, name))
        plt.close('all')
    else:
        plt.show()


def main(args):
    chrom_list = load_chrom_split()
    save_prefix = args.save_prefix
    verbosity = args.verbosity
    save_fig = args.save_fig
    no_tcr = args.no_tcr
    do_each = args.do_each
    min_m = args.min_m
    max_m = args.max_m
    min_beta = args.min_beta
    max_beta = args.max_beta
    used_transcriptoms = [True, False, False] if not no_tcr else [True, True, True]
    num_bins = 3 if not no_tcr else 1

    train_data = load_chrom_data(chrom_list=chrom_list, used_transcriptomes=used_transcriptoms, num_trans_bins=num_bins)
    (trans, ntrans, igr, igr_start, igr_end, transcriptome) = train_data
    if do_each and not no_tcr:
        iterator = [trans[:, :, 0], trans[:, :, 1], trans[:, :, 2],
                    ntrans[:, :, 0], ntrans[:, :, 1], ntrans[:, :, 2], igr]
        num_pos = 1
    elif not do_each and not no_tcr:
        iterator = [trans, ntrans, igr]
        num_pos = 3
    else:
        iterator = [trans, ntrans, igr[:, :, 0], igr[:, :, 1]]
        num_pos = 1

    kjma_model_list = create_models(
        train_data,
        do_each=do_each,
        no_tcr=no_tcr,
        verbosity=verbosity,
        save_fig=save_fig,
        save_prefix=save_prefix
    )
    tot_kjma_score = []
    tot_log_score = []
    tot_lin_score = []
    tot_hill_score = []
    for reg_kjma_m, data_list in zip(kjma_model_list, iterator):
        m_shape = np.asarray(list(reg_kjma_m.get_model_parameter('m')))
        beta = np.asarray(list(reg_kjma_m.get_model_parameter('beta')))
        mask = np.ones_like(m_shape, dtype='bool')
        if min_m > 0:
            mask[m_shape < min_m] = False
        if max_m > 0:
            mask[m_shape > max_m] = False
        if min_beta > 0:
            mask[beta < min_beta] = False
        if max_beta > 0:
            mask[beta > max_beta] = False

        kjma_score = np.asarray(reg_kjma_m.get_all_scores())
        mask = np.logical_and(mask, ~np.isnan(kjma_score))
        kjma_score, kjma_pred_l = kjma_score[mask], []
        log_score, log_pred_l = [], []
        lin_score, lin_pred_l = [], []
        hill_score, hill_pred_l = [], []

        for d, kjma, is_used in zip(data_list, reg_kjma_m.models, mask):
            if not is_used:
                continue
            kjma_pred_l.append(kjma.repair_fraction_over_time(140))
            # Logisitic regression model
            try:
                # Create single probability array
                if len(d.shape) > 1:
                    d = np.mean(d, axis=1)
                    if np.any(np.isnan(d)) or np.any(d < 0) or np.any(d > 1):
                        raise ValueError('Could not create probabilities.')
                # Sample data for logistic regression (class labels for 0/damage and 1/repair)
                sd = sample_data(d)

                log_model = LogisticRegression(penalty='l2').fit(
                    create_time_data(num_pos, len(sd)).reshape(-1, 1),
                    sd.ravel()
                )
                log_pred = log_model.predict_proba([[20], [60], [120]])[:, 1] * kjma.max_frac
                if not np.any(np.isnan(log_pred)):
                    log_score.append(np.sum((log_pred - d)**2) / len(d))
                    log_pred_l.append(log_model.predict_proba(np.arange(140).reshape(-1, 1))[:, 1] * kjma.max_frac)
            except ValueError:
                print('Repair probability creates only single class. Skip logistic regression.')

            # Hill equation model
            hill_model = Hill(create_time_data(num_pos, 1), d, kjma.max_frac, name=kjma.name)
            hill_model.estimate_parameters()
            try:
                hill_score.append(hill_model.score())
                hill_pred_l.append(hill_model.repair_fraction(np.arange(140)))
            except ValueError:
                print('Parameters could not be found. Skip Hill estimation.')

            # Linear regression model
            lin_model = LinearRegression().fit(
                create_time_data(num_pos, 1).reshape(-1, 1),
                d.ravel()
            )
            lin_pred = lin_model.predict([[20], [60], [120]])
            lin_score.append(np.sum((lin_pred - d)**2) / len(d))
            lin_pred_l.append(lin_model.predict(np.arange(140).reshape(-1, 1)))

            if verbosity > 2:
                # Plot different model predictions for area
                plt.figure(figsize=(8, 7))
                plt.scatter([20, 60, 120], d, c='black')
                plt.plot(kjma_pred_l[-1], label='KJMA', linestyle='--')
                if not np.any(np.isnan(log_pred)):
                    plt.plot(log_pred_l[-1], label='Logistic', linestyle='--')
                plt.plot(lin_pred_l[-1], label='Linear', linestyle='--')
                if not hill_model.not_found:
                    plt.plot(hill_pred_l[-1], label='Hill', linestyle='--')
                plt.legend(fontsize=18)
                plt.xlabel(r'Time $t$ (min)', fontsize=18)
                plt.ylabel(r'$f(t)$', fontsize=18)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.ylim((0, 1))
                plt.title('%s' % kjma.name, fontsize=32)
                if save_fig:
                    Path('figures/comp_models').mkdir(parents=True, exist_ok=True)
                    plt.savefig('figures/comp_models/%s_%s.png' % (save_prefix, kjma.name))
                    plt.close('all')
                else:
                    plt.show()

        # Compute baum welch
        _, kjma_log_p = mannwhitneyu(kjma_score, log_score, nan_policy='omit', alternative='two-sided')
        _, kjma_hill_p = mannwhitneyu(kjma_score, hill_score, nan_policy='omit', alternative='two-sided')
        _, kjma_lin_p = mannwhitneyu(kjma_score, lin_score, nan_policy='omit', alternative='two-sided')

        print('Mann-Whitney U p-values for %s wrt KJMA model:\n\tLog:\t%.4f\n\tHill:\t%.4f\n\tLin:\t%.4f'
              % (reg_kjma_m.name, kjma_log_p, kjma_hill_p, kjma_lin_p))

        # plot average prediction over time
        plt.figure(figsize=(8, 7))
        t_time = np.arange(140)
        plt.plot(t_time, np.mean(kjma_pred_l, axis=0), label='KJMA', linestyle='--', color='tab:blue')
        plt.fill_between(
            t_time,
            np.mean(kjma_pred_l, axis=0) - np.std(kjma_pred_l, axis=0),
            np.mean(kjma_pred_l, axis=0) + np.std(kjma_pred_l, axis=0),
            color='tab:blue',
            alpha=.1
        )
        plt.plot(t_time, np.mean(log_pred_l, axis=0), label='Logistic', linestyle='--', color='tab:orange')
        plt.fill_between(
            t_time,
            np.mean(log_pred_l, axis=0) - np.std(log_pred_l, axis=0),
            np.mean(log_pred_l, axis=0) + np.std(log_pred_l, axis=0),
            color='tab:orange',
            alpha=.1
        )
        plt.plot(t_time, np.mean(lin_pred_l, axis=0), label='Linear', linestyle='--', color='tab:green')
        plt.fill_between(
            t_time,
            np.mean(lin_pred_l, axis=0) - np.std(lin_pred_l, axis=0),
            np.mean(lin_pred_l, axis=0) + np.std(lin_pred_l, axis=0),
            color='tab:green',
            alpha=.1
        )
        plt.plot(t_time, np.mean(hill_pred_l, axis=0), label='Hill', linestyle='--', color='tab:red')
        plt.fill_between(
            t_time,
            np.mean(hill_pred_l, axis=0) - np.std(hill_pred_l, axis=0),
            np.mean(hill_pred_l, axis=0) + np.std(hill_pred_l, axis=0),
            color='tab:red',
            alpha=.1
        )
        plt.legend(fontsize=16)
        plt.xlabel(r'Time $t$ (min)', fontsize=18)
        plt.ylabel(r'$f(t)$', fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.title('Average Prediction\n%s' % reg_kjma_m.name, fontsize=32)
        plt.tight_layout()
        if save_fig:
            Path('figures/comp_models').mkdir(parents=True, exist_ok=True)
            plt.savefig('figures/comp_models/%s_%s_avg_prediction.png' % (save_prefix, reg_kjma_m.name))
            plt.close('all')
        else:
            plt.show()

        tot_log_score.extend(log_score)
        tot_hill_score.extend(hill_score)
        tot_lin_score.extend(lin_score)
        tot_kjma_score.extend(kjma_score)

        # Violin plot for region error
        plot_error_dist(
            kjma_score,
            hill_score,
            log_score,
            lin_score,
            name=reg_kjma_m.name,
            save_fig=save_fig,
            save_prefix=save_prefix
        )

    # Compute Baum-Welch for all regions
    # Compute baum welch
    _, tot_kjma_log_p = mannwhitneyu(tot_kjma_score, tot_log_score, nan_policy='omit', alternative='less')
    _, tot_kjma_hill_p = mannwhitneyu(tot_kjma_score, tot_hill_score, nan_policy='omit', alternative='less')
    _, tot_kjma_lin_p = mannwhitneyu(tot_kjma_score, tot_lin_score, nan_policy='omit', alternative='less')

    print('Mann-Whitney U p-values for all errors wrt KJMA model:\n\tLog:\t%.4f\n\tHill:\t%.4f\n\tLin:\t%.4f'
          % (tot_kjma_log_p, tot_kjma_hill_p, tot_kjma_lin_p))
    # Violin plot for total error
    plot_error_dist(
        tot_kjma_score,
        tot_hill_score,
        tot_log_score,
        tot_lin_score,
        name='total',
        save_fig=save_fig,
        save_prefix=save_prefix
    )


if __name__ == '__main__':
    main(argparse_model_comp_param(sys.argv[1:]))



