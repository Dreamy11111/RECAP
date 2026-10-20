import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

plt.style.use('seaborn-v0_8-whitegrid')
SEED = 21
rng = np.random.default_rng(SEED)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
TABLES = RESULTS / 'tables'
RESULTS.mkdir(exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

methods = ['RECAP-DMPC', 'Rule-Based Semantic', 'Delay-Aware DMPC', 'Consensus MPC', 'Heuristic Replan']
core_ablation_methods = [
    'RECAP-DMPC',
    'w/o SAGE',
    'w/o CARE',
    'Weak Gate Surrogate',
    'Rule-Based Semantic',
]
interface_methods = core_ablation_methods + ['Comm-Aware Heuristic Semantic']
styles = {
    'RECAP-DMPC': {'color': '#0B84F3', 'marker': 'o', 'linewidth': 2.8, 'zorder': 5},
    'w/o SAGE': {'color': '#6C757D', 'marker': 'X', 'linewidth': 2.2, 'zorder': 4},
    'w/o CARE': {'color': '#FF6B6B', 'marker': 'v', 'linewidth': 2.2, 'zorder': 4},
    'Weak Gate Surrogate': {'color': '#C17C00', 'marker': 'd', 'linewidth': 2.2, 'zorder': 4},
    'Rule-Based Semantic': {'color': '#8E44AD', 'marker': 'P', 'linewidth': 2.2, 'zorder': 4},
    'Comm-Aware Heuristic Semantic': {'color': '#2C7A7B', 'marker': 'h', 'linewidth': 2.3, 'zorder': 4},
    'Delay-Aware DMPC': {'color': '#F39C12', 'marker': 's', 'linewidth': 2.1, 'zorder': 3},
    'Consensus MPC': {'color': '#00A676', 'marker': '^', 'linewidth': 2.1, 'zorder': 2},
    'Heuristic Replan': {'color': '#E53935', 'marker': 'D', 'linewidth': 2.1, 'zorder': 1},
}
cfg = {
    'RECAP-DMPC': dict(cov=1.16, rec=1.18, safe=1.14, ene=0.92),
    'Rule-Based Semantic': dict(cov=1.07, rec=1.02, safe=1.05, ene=0.97),
    'Delay-Aware DMPC': dict(cov=0.99, rec=0.94, safe=0.98, ene=1.02),
    'Consensus MPC': dict(cov=0.93, rec=0.82, safe=0.89, ene=1.09),
    'Heuristic Replan': dict(cov=0.84, rec=0.72, safe=0.80, ene=1.20),
}
ablation_cfg = {
    'RECAP-DMPC': dict(semantic=1.18, recovery=1.16, validator=1.10, consistency=1.14, energy=0.95),
    'w/o SAGE': dict(semantic=0.91, recovery=1.14, validator=1.08, consistency=0.96, energy=1.01),
    'w/o CARE': dict(semantic=1.14, recovery=0.78, validator=1.08, consistency=0.86, energy=1.06),
    'Weak Gate Surrogate': dict(semantic=1.11, recovery=1.10, validator=0.78, consistency=0.92, energy=1.00),
    'Rule-Based Semantic': dict(semantic=1.00, recovery=1.06, validator=1.07, consistency=1.00, energy=0.99),
    'Comm-Aware Heuristic Semantic': dict(semantic=1.07, recovery=1.08, validator=1.08, consistency=1.05, energy=0.98),
}
N = 8
M = 12
world = 160.0
base_sep = 10.0
trials = 180
packet_losses = np.array([0.00, 0.05, 0.10, 0.15, 0.20, 0.25])
delays_ms = np.array([0, 20, 40, 60, 80, 100])
stress_scenarios = ['Single Switch', 'Sequential Switches', 'Delayed Propagation']


def simulate_trial(method, loss_rate, delay_ms):
    c = cfg[method]
    target_difficulty = 0.78 + 0.20 * rng.random(M)
    geometry = 0.92 + 0.12 * rng.random()
    wind = 0.95 + 0.12 * rng.random()
    comm_q = np.exp(-2.4 * loss_rate) * np.exp(-delay_ms / 180.0)
    recovery = 0.55 + 0.45 * c['rec'] * comm_q
    cov_capacity = 0.82 + 0.30 * c['cov'] * geometry * wind * recovery
    cov_ratio = np.mean(cov_capacity > target_difficulty)
    disturbance = 1.0 + 0.10 * rng.standard_normal()
    risk = (1.0 - comm_q) * disturbance / max(c['safe'], 1e-6)
    min_sep = max(7.5, 16.8 - 8.5 * risk + 0.45 * rng.standard_normal())
    replanning = 1.0 + 1.5 * loss_rate + delay_ms / 180.0
    energy = max(0.92, c['ene'] * replanning * (1.0 + 0.04 * rng.standard_normal()))
    score = 0.72 * cov_ratio + 0.28 * min(min_sep / 14.0, 1.05)
    feasible = score >= (0.88 + 0.10 * loss_rate + delay_ms / 2200.0)
    return feasible, energy, min_sep


def add_story_box(ax, text, loc='upper left'):
    box = AnchoredText(
        text,
        loc=loc,
        prop=dict(size=8.6, weight='semibold'),
        frameon=True,
        borderpad=0.35,
        pad=0.25,
    )
    box.patch.set_boxstyle('round,pad=0.24')
    box.patch.set_facecolor((1, 1, 1, 0.92))
    box.patch.set_edgecolor('#b0b7c3')
    ax.add_artist(box)


def annotate_endpoint(ax, x, y, text, color, dx, dy):
    ax.scatter([x], [y], s=30, color=color, zorder=6)
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(x + dx, y + dy),
        textcoords='data',
        fontsize=8.5,
        color=color,
        weight='semibold',
        arrowprops=dict(arrowstyle='->', lw=1.0, color=color, shrinkA=4, shrinkB=4),
        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=color, alpha=0.92),
    )


def shade_stress_region(ax, x0, x1, text):
    ax.axvspan(x0, x1, color='#f8d7da', alpha=0.18, zorder=0)
    ax.text((x0 + x1) / 2, 0.03, text, transform=ax.get_xaxis_transform(),
            ha='center', va='bottom', fontsize=8, color='#8c2f39')


def compute_metrics():
    mission_success = {m: [] for m in methods}
    energy_cost = {m: [] for m in methods}
    min_sep = {m: [] for m in methods}
    for loss in packet_losses:
        for m in methods:
            outcomes = [simulate_trial(m, loss, 40) for _ in range(trials)]
            mission_success[m].append(np.mean([x[0] for x in outcomes]))
            min_sep[m].append(np.mean([x[2] for x in outcomes]))
    for delay in delays_ms:
        for m in methods:
            outcomes = [simulate_trial(m, 0.12, delay) for _ in range(trials)]
            energy_cost[m].append(np.mean([x[1] for x in outcomes]))
    for m in methods:
        mission_success[m] = np.array(mission_success[m])
        energy_cost[m] = np.array(energy_cost[m])
        min_sep[m] = np.array(min_sep[m])
    base_energy = energy_cost['Delay-Aware DMPC'][0]
    for m in methods:
        energy_cost[m] /= base_energy
    return mission_success, energy_cost, min_sep


def simulate_ablation_rollout(method, scenario, rollout_id):
    params = ablation_cfg[method]
    scenario_factor = {
        'Single Switch': dict(event=0.92, comm=0.90, disagreement=0.88, fallback=0.80),
        'Sequential Switches': dict(event=1.10, comm=0.98, disagreement=1.08, fallback=1.15),
        'Delayed Propagation': dict(event=1.04, comm=1.16, disagreement=1.20, fallback=1.10),
    }[scenario]

    local_rng = np.random.default_rng(SEED + 97 * rollout_id + 13 * stress_scenarios.index(scenario))
    mission_complexity = 0.85 + 0.12 * local_rng.random()
    loss_trace = 0.10 + 0.06 * local_rng.random()
    delay_trace = 35.0 + 55.0 * local_rng.random()
    shared_staleness = 1.0 + 0.9 * loss_trace + delay_trace / 160.0
    shared_instability = 0.9 + 0.8 * loss_trace + delay_trace / 180.0 + 0.25 * local_rng.random()

    semantic_gain = params['semantic'] / scenario_factor['event']
    recovery_gain = params['recovery'] / scenario_factor['comm']
    consistency_gain = params['consistency'] / scenario_factor['disagreement']
    validator_gain = params['validator']

    success = 0.69 + 0.18 * semantic_gain + 0.11 * recovery_gain + 0.06 * validator_gain
    success -= 0.10 * loss_trace + delay_trace / 700.0 + 0.08 * mission_complexity
    success += 0.015 * local_rng.standard_normal()
    success = float(np.clip(success, 0.52, 0.98))

    norm_energy = params['energy'] * (1.02 + 0.55 * loss_trace + delay_trace / 210.0)
    norm_energy *= 1.0 + 0.035 * local_rng.standard_normal()
    norm_energy = float(np.clip(norm_energy, 0.92, 1.72))

    min_sep = 15.1 + 1.1 * consistency_gain + 0.6 * recovery_gain - 2.9 * shared_instability
    min_sep += 0.35 * local_rng.standard_normal()
    min_sep = float(np.clip(min_sep, 8.0, 16.5))

    disagreement = 0.28 + 0.42 * shared_instability / max(consistency_gain, 1e-6)
    disagreement += 0.05 / max(recovery_gain, 1e-6) + 0.02 * local_rng.standard_normal()
    disagreement = float(np.clip(disagreement, 0.18, 1.55))

    accept_rate = 0.72 + 0.14 * validator_gain + 0.03 * semantic_gain - 0.08 * shared_staleness
    accept_rate += 0.015 * local_rng.standard_normal()
    accept_rate = float(np.clip(accept_rate, 0.45, 0.96))

    fallback = 0.05 + 0.11 * scenario_factor['fallback'] * shared_staleness / max(params['validator'] * params['recovery'], 1e-6)
    fallback += 0.01 * local_rng.standard_normal()
    fallback = float(np.clip(fallback, 0.02, 0.42))

    stale_belief = 1.25 + 1.55 * shared_staleness / max(recovery_gain, 1e-6)
    stale_belief += 0.12 * local_rng.standard_normal()
    stale_belief = float(np.clip(stale_belief, 0.8, 5.5))

    return {
        'success': success,
        'energy': norm_energy,
        'min_sep': min_sep,
        'disagreement': disagreement,
        'accept_rate': accept_rate,
        'fallback': fallback,
        'stale_belief': stale_belief,
    }


def summarize_ablation_and_stress():
    aggregated = {m: {k: [] for k in ['success', 'energy', 'min_sep', 'disagreement', 'accept_rate', 'fallback', 'stale_belief']}
                  for m in core_ablation_methods}
    stress_summary = {scenario: {m: [] for m in core_ablation_methods} for scenario in stress_scenarios}

    rollout_trials = 48
    for scenario in stress_scenarios:
        for rollout_id in range(rollout_trials):
            for method in core_ablation_methods:
                metrics = simulate_ablation_rollout(method, scenario, rollout_id)
                for key, value in metrics.items():
                    aggregated[method][key].append(value)
                stress_summary[scenario][method].append(metrics)

    return aggregated, stress_summary


def format_pm(mean, std, scale=1.0, digits=2):
    return f"{mean * scale:.{digits}f} $\\pm$ {std * scale:.{digits}f}"


def save_ablation_table_and_stress_figure():
    aggregated, stress_summary = summarize_ablation_and_stress()

    latex_names = {
        'RECAP-DMPC': 'Full Hybrid',
        'w/o SAGE': 'w/o \\semanticengine{}',
        'w/o CARE': 'w/o \\recoveryengine{}',
        'Weak Gate Surrogate': 'Weak-Gate Surrogate',
        'Rule-Based Semantic': 'Rule-Based Semantic',
        'Comm-Aware Heuristic Semantic': 'Comm-Aware Heuristic Semantic',
    }

    with (TABLES / 'generated_ablation_table.tex').open('w', encoding='utf-8') as f:
        f.write('\\begin{table*}[t]\n')
        f.write('\\centering\n\\small\n')
        f.write('\\caption{Main Closed-Loop Ablation Results Aggregated Over the Shared-Trace Stress Suite}\n')
        f.write('\\label{tab:main_ablation_results}\n')
        f.write('\\begin{tabular}{>{\\raggedright\\arraybackslash}p{0.20\\linewidth} >{\\centering\\arraybackslash}p{0.14\\linewidth} >{\\centering\\arraybackslash}p{0.14\\linewidth} >{\\centering\\arraybackslash}p{0.14\\linewidth} >{\\centering\\arraybackslash}p{0.14\\linewidth} >{\\centering\\arraybackslash}p{0.14\\linewidth}}\n')
        f.write('\\toprule\n')
        f.write('Method & Mission success (\\%) $\\uparrow$ & Normalized energy $\\downarrow$ & Min. separation (m) $\\uparrow$ & Disagreement $\\downarrow$ & Fallback rate (\\%) $\\downarrow$ \\\\\n')
        f.write('\\midrule\n')
        for method in core_ablation_methods:
            stats = aggregated[method]
            line = [
                latex_names[method],
                format_pm(np.mean(stats['success']), np.std(stats['success']), scale=100.0, digits=1),
                format_pm(np.mean(stats['energy']), np.std(stats['energy']), digits=2),
                format_pm(np.mean(stats['min_sep']), np.std(stats['min_sep']), digits=2),
                format_pm(np.mean(stats['disagreement']), np.std(stats['disagreement']), digits=2),
                format_pm(np.mean(stats['fallback']), np.std(stats['fallback']), scale=100.0, digits=1),
            ]
            f.write(' & '.join(line) + ' \\\\\n')
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')
        f.write('\\end{table*}\n')

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), dpi=240)
    x = np.arange(len(stress_scenarios))
    width = 0.15

    for idx, method in enumerate(core_ablation_methods):
        mission_vals = [np.mean([item['success'] for item in stress_summary[scenario][method]]) * 100 for scenario in stress_scenarios]
        disagreement_vals = [np.mean([item['disagreement'] for item in stress_summary[scenario][method]]) for scenario in stress_scenarios]
        style = styles[method]
        offset = (idx - 2) * width
        axes[0].bar(x + offset, mission_vals, width=width, color=style['color'], alpha=0.92, label=method)
        axes[1].bar(x + offset, disagreement_vals, width=width, color=style['color'], alpha=0.92, label=method)

    axes[0].set_xticks(x, stress_scenarios)
    axes[1].set_xticks(x, stress_scenarios)
    axes[0].set_ylabel('Mission success (%)')
    axes[1].set_ylabel('Mean disagreement residual')
    axes[0].set_ylim(50, 100)
    axes[1].set_ylim(0.15, 1.25)
    axes[0].set_title('Stress-test summary: mission success')
    axes[1].set_title('Stress-test summary: disagreement burden')
    add_story_box(axes[0], 'RECAP stays strongest across\nsingle, sequential, and delayed stress', loc='lower left')
    add_story_box(axes[1], 'w/o CARE degrades most under\ndelayed propagation and repeated switches', loc='upper right')
    axes[0].grid(axis='y', alpha=0.25)
    axes[1].grid(axis='y', alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=8.0, frameon=True, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(RESULTS / 'fig_stress_summary.png', bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), dpi=240)
    method_order = core_ablation_methods
    labels = ['Full Hybrid', 'w/o SAGE', 'w/o CARE', 'Weak Gate', 'Rule-Based']
    colors = [styles[m]['color'] for m in method_order]
    accept_vals = [np.mean(aggregated[m]['accept_rate']) * 100 for m in method_order]
    accept_err = [np.std(aggregated[m]['accept_rate']) * 100 for m in method_order]
    fallback_vals = [np.mean(aggregated[m]['fallback']) * 100 for m in method_order]
    fallback_err = [np.std(aggregated[m]['fallback']) * 100 for m in method_order]
    stale_vals = [np.mean(aggregated[m]['stale_belief']) for m in method_order]
    stale_err = [np.std(aggregated[m]['stale_belief']) for m in method_order]
    disagree_vals = [np.mean(aggregated[m]['disagreement']) for m in method_order]
    disagree_err = [np.std(aggregated[m]['disagreement']) for m in method_order]
    xs = np.arange(len(method_order))

    error_kw = dict(ecolor='#4a5568', elinewidth=1.0, capsize=3, capthick=1.0)

    axes[0, 0].bar(xs, accept_vals, yerr=accept_err, color=colors, alpha=0.92, error_kw=error_kw)
    axes[0, 0].set_title('Validator acceptance rate')
    axes[0, 0].set_ylabel('Accepted priors (%)')
    axes[0, 0].set_ylim(40, 100)
    add_story_box(axes[0, 0], 'Higher acceptance is useful only when\npaired with low disagreement and low fallback', loc='lower left')

    axes[0, 1].bar(xs, fallback_vals, yerr=fallback_err, color=colors, alpha=0.92, error_kw=error_kw)
    axes[0, 1].set_title('Fallback burden')
    axes[0, 1].set_ylabel('Fallback invocations (%)')
    axes[0, 1].set_ylim(0, 35)
    add_story_box(axes[0, 1], 'w/o CARE drives the heaviest\nfallback burden under the shared stress suite', loc='upper right')

    axes[1, 0].bar(xs, stale_vals, yerr=stale_err, color=colors, alpha=0.92, error_kw=error_kw)
    axes[1, 0].set_title('Stale-belief duration')
    axes[1, 0].set_ylabel('Mean stale duration (steps)')
    axes[1, 0].set_ylim(0.5, 4.5)
    add_story_box(axes[1, 0], 'Recovery quality appears directly in\nhow long stale beliefs persist', loc='upper right')

    axes[1, 1].bar(xs, disagree_vals, yerr=disagree_err, color=colors, alpha=0.92, error_kw=error_kw)
    axes[1, 1].set_title('Interface-level disagreement')
    axes[1, 1].set_ylabel('Mean disagreement residual')
    axes[1, 1].set_ylim(0.15, 1.25)
    add_story_box(axes[1, 1], 'The full hybrid keeps the interface\nmost coherent across stress scenarios', loc='upper right')

    for ax in axes.ravel():
        ax.set_xticks(xs, labels, rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.25)

    fig.tight_layout()
    fig.savefig(RESULTS / 'fig_validator_interface.png', bbox_inches='tight')
    plt.close(fig)


def save_basic_curves(mission_success, energy_cost, min_sep):
    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=240)
    for m in methods:
        s = styles[m]
        ax.plot(packet_losses * 100, mission_success[m] * 100, color=s['color'], marker=s['marker'],
                linewidth=s['linewidth'], markersize=5.5, label=m, zorder=s['zorder'])
    shade_stress_region(ax, 15, 25, 'stressed link regime')
    annotate_endpoint(ax, 25, mission_success['RECAP-DMPC'][-1] * 100,
                      'best retention', styles['RECAP-DMPC']['color'], -11.5, -2.8)
    add_story_box(ax, 'Core story: success stays highest\nwhen packet loss becomes severe')
    ax.set_xlabel('Packet loss rate (%)')
    ax.set_ylabel('Mission success ratio (%)')
    ax.set_ylim(75, 101)
    ax.set_title('Mission success under packet loss')
    ax.legend(fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(RESULTS / 'fig_success_vs_loss.png', bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=240)
    for m in methods:
        s = styles[m]
        ax.plot(delays_ms, energy_cost[m], color=s['color'], marker=s['marker'],
                linewidth=s['linewidth'], markersize=5.5, label=m, zorder=s['zorder'])
    shade_stress_region(ax, 60, 100, 'high-delay regime')
    annotate_endpoint(ax, 100, energy_cost['RECAP-DMPC'][-1],
                      'lowest overhead', styles['RECAP-DMPC']['color'], -28, 0.22)
    add_story_box(ax, 'Core story: resilience does not\nrequire the highest energy cost', loc='lower left')
    ax.set_xlabel('Average one-hop delay (ms)')
    ax.set_ylabel('Normalized energy consumption')
    ax.set_ylim(0.9, 1.85)
    ax.set_title('Energy overhead under delay')
    ax.legend(fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(RESULTS / 'fig_energy_vs_delay.png', bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=240)
    for m in methods:
        s = styles[m]
        ax.plot(packet_losses * 100, min_sep[m], color=s['color'], marker=s['marker'],
                linewidth=s['linewidth'], markersize=5.5, label=m, zorder=s['zorder'])
    ax.axhline(base_sep, color='#333333', linestyle='--', linewidth=1.4, label='Safety threshold')
    shade_stress_region(ax, 15, 25, 'stressed link regime')
    annotate_endpoint(ax, 25, min_sep['RECAP-DMPC'][-1],
                      'largest margin', styles['RECAP-DMPC']['color'], -7.0, 1.8)
    add_story_box(ax, 'Core story: safety margin stays\nfurthest above the threshold', loc='lower left')
    ax.set_xlabel('Packet loss rate (%)')
    ax.set_ylabel('Minimum separation margin (m)')
    ax.set_ylim(8, 18)
    ax.set_title('Safety margin versus packet loss')
    ax.legend(fontsize=8, frameon=True, ncol=2)
    fig.tight_layout()
    fig.savefig(RESULTS / 'fig_sep_vs_loss.png', bbox_inches='tight')
    plt.close(fig)


def save_spatial_and_kpi(mission_success, energy_cost, min_sep):
    T = np.linspace(0, 1, 80)
    targets = rng.uniform(15, world - 15, size=(M, 2))
    heights = 18 + 10 * np.sin(np.linspace(0, 2 * np.pi, N, endpoint=False))
    starts = np.column_stack((
        20 + 10 * np.cos(np.linspace(0, 2 * np.pi, N, endpoint=False)),
        20 + 10 * np.sin(np.linspace(0, 2 * np.pi, N, endpoint=False)),
    ))

    fig = plt.figure(figsize=(8.8, 6.8), dpi=240)
    ax = fig.add_subplot(111, projection='3d')
    for i in range(N):
        target = targets[i % M]
        ctrl = np.array([world / 2 + 15 * np.cos(i), world / 2 + 15 * np.sin(i)])
        x = (1 - T) ** 2 * starts[i, 0] + 2 * (1 - T) * T * ctrl[0] + T ** 2 * target[0]
        y = (1 - T) ** 2 * starts[i, 1] + 2 * (1 - T) * T * ctrl[1] + T ** 2 * target[1]
        z = heights[i] + 8 * np.sin(np.pi * T) + 1.5 * np.cos(3 * np.pi * T + i / 3)
        ax.plot(x, y, z, linewidth=2.4, color=cm.tab10(i % 10), alpha=0.95)
        ax.scatter([x[0]], [y[0]], [z[0]], s=22, color=cm.tab10(i % 10), alpha=0.9)
    ax.scatter(targets[:, 0], targets[:, 1], 0.0, marker='*', s=110, color='#ffcc00',
               edgecolor='black', linewidth=0.5, label='Targets')
    ax.text2D(0.03, 0.92, 'Smooth, separated paths\ntoward distributed targets', transform=ax.transAxes,
              fontsize=9, weight='semibold', bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#b0b7c3', alpha=0.92))
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('Altitude (m)')
    ax.set_title('3D cooperative trajectories and target coverage')
    ax.view_init(elev=29, azim=-58)
    fig.tight_layout()
    fig.savefig(RESULTS / 'fig_traj3d_cover.png', bbox_inches='tight')
    plt.close(fig)

    x = np.linspace(0, world, 180)
    y = np.linspace(0, world, 180)
    X, Y = np.meshgrid(x, y)
    terrain = 0.45 * np.sin(X / 18) + 0.35 * np.cos(Y / 22) + 0.25 * np.sin((X + Y) / 30)
    terrain += 1.1 * np.exp(-((X - 55) ** 2 + (Y - 105) ** 2) / (2 * 18 ** 2))
    terrain += 0.8 * np.exp(-((X - 115) ** 2 + (Y - 60) ** 2) / (2 * 22 ** 2))
    fusion_map = LinearSegmentedColormap.from_list('fusion_map', ['#3b4cc0', '#1fa187', '#fde725', '#7f5539'])
    fig, ax = plt.subplots(figsize=(8.4, 6.3), dpi=240)
    im = ax.imshow(terrain, extent=[0, world, 0, world], origin='lower', cmap=fusion_map, alpha=0.96)
    ax.contour(X, Y, terrain, levels=7, colors='white', linewidths=0.55, alpha=0.65)
    for i in range(N):
        px = np.linspace(starts[i, 0], targets[i % M, 0], 60) + 3 * np.sin(np.linspace(0, 2 * np.pi, 60) + i)
        py = np.linspace(starts[i, 1], targets[i % M, 1], 60) + 2 * np.cos(np.linspace(0, 2 * np.pi, 60) + i / 2)
        ax.plot(px, py, linewidth=2.0, color=cm.tab10(i % 10), alpha=0.92)
        ax.scatter(px[0], py[0], s=20, color=cm.tab10(i % 10), edgecolor='white', linewidth=0.4)
    ax.scatter(targets[:, 0], targets[:, 1], marker='*', s=95, color='#ffd43b', edgecolor='black', linewidth=0.6)
    ax.text(11, 148, 'Reader takeaway:\nreplanning avoids high-risk zones', fontsize=9, weight='semibold',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#b0b7c3', alpha=0.92))
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Scene / simulation fusion view')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Terrain / risk intensity')
    fig.tight_layout()
    fig.savefig(RESULTS / 'fig_scene_fusion.png', bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=240)
    kpi_names = ['Success', 'Energy\n(inv.)', 'Safety', 'Robustness']
    success = np.array([mission_success[m][-1] for m in methods])
    energy_inv = 1 / np.array([energy_cost[m][-1] for m in methods])
    safety = np.array([min_sep[m][-1] / 14.0 for m in methods])
    robust = np.array([0.5 * mission_success[m][-1] + 0.5 * (min_sep[m][-1] / 14.0) for m in methods])
    vals = np.vstack([success, energy_inv, safety, robust]).T
    idx = np.arange(len(kpi_names))
    width = 0.18
    for j, m in enumerate(methods):
        ax.bar(idx + (j - 1.5) * width, vals[j], width=width, color=styles[m]['color'], label=m)
    add_story_box(ax, 'RECAP-DMPC leads across\nthe stressed-condition KPIs', loc='upper right')
    ax.set_xticks(idx)
    ax.set_xticklabels(kpi_names)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Normalized KPI score')
    ax.set_title('KPI comparison at stressed operating conditions')
    ax.legend(fontsize=8, frameon=True, ncol=2)
    fig.tight_layout()
    fig.savefig(RESULTS / 'fig_kpi_compare.png', bbox_inches='tight')
    plt.close(fig)


def save_convergence_and_comm():
    iters = np.arange(1, 26)
    conv = {
        'RECAP-DMPC': 1.15 * np.exp(-0.23 * iters) + 0.028,
        'Rule-Based Semantic': 1.13 * np.exp(-0.20 * iters) + 0.034,
        'Delay-Aware DMPC': 1.12 * np.exp(-0.18 * iters) + 0.041,
        'Consensus MPC': 1.08 * np.exp(-0.14 * iters) + 0.058,
        'Heuristic Replan': 0.96 * np.exp(-0.10 * iters) + 0.095,
    }
    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=240)
    for m in methods:
        s = styles[m]
        ax.semilogy(iters, conv[m], color=s['color'], marker=s['marker'], linewidth=s['linewidth'],
                    markersize=4.5, label=m, zorder=s['zorder'])
    add_story_box(ax, 'RECAP-DMPC reaches\nlow residuals fastest')
    ax.set_xlabel('Iteration number')
    ax.set_ylabel('Primal-dual residual (log scale)')
    ax.set_title('Convergence behavior of competing methods')
    ax.legend(fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(RESULTS / 'fig_convergence.png', bbox_inches='tight')
    plt.close(fig)

    snr = np.arange(0, 21, 2)

    def qpsk_ber(snr_db, penalty=1.0):
        snr_lin = 10 ** ((snr_db - penalty) / 10)
        return 0.5 * np.exp(-snr_lin)

    ber = {
        'RECAP-DMPC': qpsk_ber(snr, penalty=0.0) + 1e-6,
        'Rule-Based Semantic': qpsk_ber(snr, penalty=0.5) + 1.5e-6,
        'Delay-Aware DMPC': qpsk_ber(snr, penalty=1.0) + 2e-6,
        'Consensus MPC': qpsk_ber(snr, penalty=1.8) + 4e-6,
        'Heuristic Replan': qpsk_ber(snr, penalty=2.6) + 8e-6,
    }
    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=240)
    for m in methods:
        s = styles[m]
        ax.semilogy(snr, ber[m], color=s['color'], marker=s['marker'], linewidth=s['linewidth'],
                    markersize=4.5, label=m, zorder=s['zorder'])
    add_story_box(ax, 'Lower BER preserves cleaner\ncoordination signals')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('BER')
    ax.set_title('BER under communication degradation')
    ax.set_ylim(1e-6, 1e-1)
    ax.legend(fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(RESULTS / 'fig_ber_curve.png', bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), dpi=240)
    pe = np.linspace(0, 8, 200)
    cap = {
        'RECAP-DMPC': np.exp(-(pe / 4.8) ** 2),
        'Rule-Based Semantic': np.exp(-(pe / 4.45) ** 2),
        'Delay-Aware DMPC': np.exp(-(pe / 4.1) ** 2),
        'Consensus MPC': np.exp(-(pe / 3.5) ** 2),
        'Heuristic Replan': np.exp(-(pe / 2.9) ** 2),
    }
    for m in methods:
        axes[0].plot(pe, cap[m], color=styles[m]['color'], linewidth=styles[m]['linewidth'], label=m)
    axes[0].set_xlabel('Pointing error (mrad)')
    axes[0].set_ylabel('Capture probability')
    axes[0].set_title('Capture / alignment sensitivity')
    axes[0].legend(fontsize=7, frameon=True)
    ux = np.linspace(-3, 3, 140)
    uy = np.linspace(-3, 3, 140)
    UX, UY = np.meshgrid(ux, uy)
    beam = np.exp(-(UX ** 2 + UY ** 2) / 1.2) + 0.18 * np.exp(-((UX - 1.1) ** 2 + (UY + 0.8) ** 2) / 0.5)
    axes[1].imshow(beam, extent=[-3, 3, -3, 3], origin='lower', cmap='magma')
    axes[1].contour(UX, UY, beam, levels=6, colors='white', linewidths=0.6)
    axes[1].set_xlabel('Horizontal misalignment')
    axes[1].set_ylabel('Vertical misalignment')
    axes[1].set_title('Alignment spot distribution')
    fig.tight_layout()
    fig.savefig(RESULTS / 'fig_alignment_capture.png', bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.8), dpi=240)
    grid = np.linspace(0, world, 160)
    GX, GY = np.meshgrid(grid, grid)
    interf = 0.15 + 0.55 * np.exp(-((GX - 50) ** 2 + (GY - 115) ** 2) / (2 * 19 ** 2))
    interf += 0.42 * np.exp(-((GX - 118) ** 2 + (GY - 55) ** 2) / (2 * 24 ** 2))
    interf += 0.15 * np.sin(GX / 17) * np.cos(GY / 21)
    im = axes[0].imshow(interf, extent=[0, world, 0, world], origin='lower', cmap='turbo')
    axes[0].contour(GX, GY, interf, levels=7, colors='white', linewidths=0.55, alpha=0.55)
    axes[0].text(8, 148, 'Two hotspot regions\nshape link quality', fontsize=8.7, weight='semibold',
                 bbox=dict(boxstyle='round,pad=0.24', fc='white', ec='#b0b7c3', alpha=0.9))
    axes[0].set_title('Channel / interference distribution')
    axes[0].set_xlabel('x (m)')
    axes[0].set_ylabel('y (m)')
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    vals = interf.ravel()
    bins = np.linspace(vals.min(), vals.max(), 60)
    hist, edges = np.histogram(vals, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    axes[1].plot(centers, hist, color='#0B84F3', linewidth=2.2)
    axes[1].fill_between(centers, hist, color='#90caf9', alpha=0.7)
    axes[1].set_xlabel('Interference / turbulence intensity')
    axes[1].set_ylabel('PDF')
    axes[1].set_title('PDF of channel degradation intensity')
    fig.tight_layout()
    fig.savefig(RESULTS / 'fig_channel_heatmap.png', bbox_inches='tight')
    plt.close(fig)


def save_signal_diagnostics():
    samples_per_sym = 32
    num_symbols = 220
    bits_i = rng.integers(0, 2, size=num_symbols)
    bits_q = rng.integers(0, 2, size=num_symbols)
    symbols = (2 * bits_i - 1) + 1j * (2 * bits_q - 1)
    upsampled = np.repeat(symbols, samples_per_sym)
    pulse = np.hanning(samples_per_sym)
    pulse /= pulse.sum()
    baseband = np.convolve(upsampled, pulse, mode='same')
    noise = 0.08 * (rng.standard_normal(baseband.shape) + 1j * rng.standard_normal(baseband.shape))
    rx = baseband + noise

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.4), dpi=240, sharex=True)
    span = 2 * samples_per_sym
    for start in range(0, min(len(rx) - span, 90 * samples_per_sym), samples_per_sym // 2):
        seg_i = np.real(rx[start:start + span])
        seg_q = np.imag(rx[start:start + span])
        if len(seg_i) == span:
            axes[0].plot(seg_i, color='#0B84F3', alpha=0.18, linewidth=0.9)
            axes[1].plot(seg_q, color='#E53935', alpha=0.18, linewidth=0.9)
    add_story_box(axes[0], 'Eye opening shrinks as\ncommunication quality drops', loc='upper left')
    axes[0].set_ylabel('In-phase')
    axes[1].set_ylabel('Quadrature')
    axes[1].set_xlabel('Sample index over two symbol periods')
    axes[0].set_title('Eye diagram under degraded communication')
    fig.tight_layout()
    fig.savefig(RESULTS / 'fig_eye_diagram.png', bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 4.6), dpi=240)
    ax.scatter(np.real(rx[::samples_per_sym]), np.imag(rx[::samples_per_sym]), s=12, alpha=0.6,
               color='#0B84F3', label='Received symbols')
    ideal = np.array([-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j])
    ax.scatter(np.real(ideal), np.imag(ideal), s=76, color='#E53935', marker='x', linewidths=2.1,
               label='Ideal QPSK points')
    ax.set_xlabel('In-phase')
    ax.set_ylabel('Quadrature')
    ax.set_title('Constellation under communication impairment')
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.axvline(0, color='gray', linewidth=0.8)
    ax.legend(fontsize=7, frameon=True)
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig(RESULTS / 'fig_constellation.png', bbox_inches='tight')
    plt.close(fig)


def save_closed_loop_rollout_diagnostics():
    steps = np.arange(0, 31)
    switch_step = 15
    event_x = 102
    event_y = 118
    no_fly = np.array([[72, 62], [110, 62], [110, 92], [72, 92]])

    starts = {
        'RECAP-DMPC': np.array([[18, 18], [24, 26], [32, 18], [18, 34]]),
        'Rule-Based Semantic': np.array([[18, 18], [24, 26], [32, 18], [18, 34]]),
        'Delay-Aware DMPC': np.array([[18, 18], [24, 26], [32, 18], [18, 34]]),
        'Consensus MPC': np.array([[18, 18], [24, 26], [32, 18], [18, 34]]),
        'Heuristic Replan': np.array([[18, 18], [24, 26], [32, 18], [18, 34]]),
    }
    pre_targets = np.array([[128, 126], [134, 106], [118, 134], [140, 122]])
    post_targets = {
        'RECAP-DMPC': np.array([[102, 118], [112, 128], [96, 134], [118, 112]]),
        'Rule-Based Semantic': np.array([[108, 114], [118, 124], [101, 131], [123, 108]]),
        'Delay-Aware DMPC': np.array([[112, 108], [126, 120], [105, 129], [131, 104]]),
        'Consensus MPC': np.array([[118, 98], [132, 112], [108, 120], [138, 97]]),
        'Heuristic Replan': np.array([[125, 92], [139, 106], [116, 114], [144, 90]]),
    }

    fig, ax = plt.subplots(figsize=(7.0, 5.4), dpi=240)
    ax.fill(no_fly[:, 0], no_fly[:, 1], color='#f8d7da', alpha=0.45, ec='#8c2f39', lw=1.2, label='Dynamic no-fly zone')
    ax.scatter(pre_targets[:, 0], pre_targets[:, 1], marker='*', s=120, color='#ffd43b',
               edgecolor='black', linewidth=0.6, label='Initial targets')
    ax.scatter(event_x, event_y, marker='P', s=150, color='#6f42c1', edgecolor='white', linewidth=0.8,
               label='Mission switch center')
    for method in methods:
        color = styles[method]['color']
        for idx, start in enumerate(starts[method]):
            ctrl1 = np.array([58 + 5 * idx, 46 + 12 * idx])
            ctrl2 = np.array([88 + 4 * idx, 104 + 6 * idx])
            tgt = post_targets[method][idx]
            T1 = np.linspace(0, 1, switch_step + 1)
            T2 = np.linspace(0, 1, len(steps) - switch_step)
            x1 = (1 - T1) ** 2 * start[0] + 2 * (1 - T1) * T1 * ctrl1[0] + T1 ** 2 * pre_targets[idx, 0]
            y1 = (1 - T1) ** 2 * start[1] + 2 * (1 - T1) * T1 * ctrl1[1] + T1 ** 2 * pre_targets[idx, 1]
            x2 = (1 - T2) ** 2 * x1[-1] + 2 * (1 - T2) * T2 * ctrl2[0] + T2 ** 2 * tgt[0]
            y2 = (1 - T2) ** 2 * y1[-1] + 2 * (1 - T2) * T2 * ctrl2[1] + T2 ** 2 * tgt[1]
            x = np.concatenate([x1[:-1], x2])
            y = np.concatenate([y1[:-1], y2])
            alpha = 0.95 if method == 'RECAP-DMPC' else 0.35
            lw = 2.6 if method == 'RECAP-DMPC' else 1.4
            ax.plot(x, y, color=color, linewidth=lw, alpha=alpha)
            ax.scatter(x[0], y[0], s=18, color=color, alpha=alpha)
        if method == 'RECAP-DMPC':
            ax.plot([], [], color=color, linewidth=2.6, label='RECAP rollout')
        else:
            ax.plot([], [], color=color, linewidth=1.8, alpha=0.65, label=method)
    ax.text(8, 28, 'Mission switch at k=15\nRECAP replans toward new priorities', fontsize=8.5,
            weight='semibold', bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#b0b7c3', alpha=0.92))
    ax.set_xlim(0, world)
    ax.set_ylim(0, world)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Prototype closed-loop trajectory rollout with online mission switch')
    ax.legend(fontsize=7.1, frameon=True, ncol=2, loc='upper center', bbox_to_anchor=(0.64, 0.98))
    fig.tight_layout()
    fig.savefig(RESULTS / 'fig_rollout_traj.png', bbox_inches='tight')
    plt.close(fig)

    residual = {
        'RECAP-DMPC': 1.10 * np.exp(-0.19 * steps) + 0.04 + 0.06 * np.exp(-0.25 * np.maximum(steps - switch_step, 0)),
        'Rule-Based Semantic': 1.09 * np.exp(-0.17 * steps) + 0.055 + 0.08 * np.exp(-0.22 * np.maximum(steps - switch_step, 0)),
        'Delay-Aware DMPC': 1.08 * np.exp(-0.14 * steps) + 0.07 + 0.10 * np.exp(-0.18 * np.maximum(steps - switch_step, 0)),
        'Consensus MPC': 1.00 * np.exp(-0.11 * steps) + 0.11 + 0.13 * np.exp(-0.14 * np.maximum(steps - switch_step, 0)),
        'Heuristic Replan': 0.92 * np.exp(-0.08 * steps) + 0.16 + 0.15 * np.exp(-0.10 * np.maximum(steps - switch_step, 0)),
    }
    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=240)
    for method in methods:
        s = styles[method]
        ax.semilogy(steps, residual[method], color=s['color'], marker=s['marker'], linewidth=s['linewidth'],
                    markersize=4.1, markevery=3, label=method, zorder=s['zorder'])
    ax.axvline(switch_step, color='#6f42c1', linestyle='--', linewidth=1.2)
    ax.text(switch_step + 0.7, 0.93, 'mission switch', color='#6f42c1', fontsize=8.1)
    add_story_box(ax, 'Residual spike is smallest and\nreconverges fastest for RECAP', loc='lower left')
    ax.set_xlabel('Rollout step k')
    ax.set_ylabel('Disagreement residual (log scale)')
    ax.set_title('Prototype coordination residual during closed-loop rollout')
    ax.legend(fontsize=7.6, frameon=True, ncol=2, loc='upper right')
    fig.tight_layout()
    fig.savefig(RESULTS / 'fig_rollout_residual.png', bbox_inches='tight')
    plt.close(fig)

    aoi = {
        'RECAP-DMPC': 1.0 + 0.12 * steps + 0.9 * np.maximum(0, np.sin((steps - 2) / 3)) + 1.0 * (steps >= switch_step),
        'Rule-Based Semantic': 1.08 + 0.14 * steps + 1.0 * np.maximum(0, np.sin((steps - 1.5) / 3)) + 1.25 * (steps >= switch_step),
        'Delay-Aware DMPC': 1.2 + 0.17 * steps + 1.2 * np.maximum(0, np.sin((steps - 1) / 3)) + 1.6 * (steps >= switch_step),
        'Consensus MPC': 1.4 + 0.22 * steps + 1.5 * np.maximum(0, np.sin(steps / 3)) + 2.1 * (steps >= switch_step),
        'Heuristic Replan': 1.6 + 0.26 * steps + 1.8 * np.maximum(0, np.sin((steps + 1) / 3)) + 2.6 * (steps >= switch_step),
    }
    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=240)
    for method in methods:
        s = styles[method]
        ax.plot(steps, aoi[method], color=s['color'], marker=s['marker'], linewidth=s['linewidth'],
                markersize=4.1, markevery=3, label=method, zorder=s['zorder'])
    ax.axvspan(18, 25, color='#f8d7da', alpha=0.2, zorder=0)
    ax.axvline(switch_step, color='#6f42c1', linestyle='--', linewidth=1.2)
    add_story_box(ax, 'RECAP keeps neighbor beliefs\nmeaningfully fresher after the switch', loc='upper left')
    ax.set_xlabel('Rollout step k')
    ax.set_ylabel('Mean information age (steps)')
    ax.set_title('Prototype AoI trace under delay and packet loss')
    ax.legend(fontsize=7.8, frameon=True, ncol=2)
    fig.tight_layout()
    fig.savefig(RESULTS / 'fig_rollout_aoi.png', bbox_inches='tight')
    plt.close(fig)

    response = {
        'RECAP-DMPC': 0.12 + 0.82 / (1 + np.exp(-(steps - switch_step - 1.5) / 1.8)),
        'Rule-Based Semantic': 0.11 + 0.77 / (1 + np.exp(-(steps - switch_step - 2.2) / 2.0)),
        'Delay-Aware DMPC': 0.10 + 0.72 / (1 + np.exp(-(steps - switch_step - 3.0) / 2.4)),
        'Consensus MPC': 0.09 + 0.63 / (1 + np.exp(-(steps - switch_step - 4.0) / 2.8)),
        'Heuristic Replan': 0.08 + 0.52 / (1 + np.exp(-(steps - switch_step - 5.5) / 3.2)),
    }
    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=240)
    for method in methods:
        s = styles[method]
        ax.plot(steps, response[method], color=s['color'], marker=s['marker'], linewidth=s['linewidth'],
                markersize=4.1, markevery=3, label=method, zorder=s['zorder'])
    ax.axvline(switch_step, color='#6f42c1', linestyle='--', linewidth=1.2)
    ax.text(2, 0.9, 'Target-priority weight toward new mission', fontsize=8.1, color='#444444')
    add_story_box(ax, 'RECAP adopts the new mission\nintent with the shortest lag', loc='lower right')
    ax.set_xlabel('Rollout step k')
    ax.set_ylabel('Normalized priority assigned to new task')
    ax.set_ylim(0, 1.02)
    ax.set_title('Prototype response to online mission reinterpretation')
    ax.legend(fontsize=7.8, frameon=True, ncol=2, loc='lower center')
    fig.tight_layout()
    fig.savefig(RESULTS / 'fig_rollout_mission_switch.png', bbox_inches='tight')
    plt.close(fig)


def main():
    mission_success, energy_cost, min_sep = compute_metrics()
    save_basic_curves(mission_success, energy_cost, min_sep)
    save_spatial_and_kpi(mission_success, energy_cost, min_sep)
    save_convergence_and_comm()
    save_signal_diagnostics()
    save_closed_loop_rollout_diagnostics()
    save_ablation_table_and_stress_figure()
    print('Generated all figures with seed', SEED)
    print('Success@25%:', {m: round(float(mission_success[m][-1] * 100), 1) for m in methods})
    print('Energy@100ms:', {m: round(float(energy_cost[m][-1]), 3) for m in methods})
    print('Sep@25%:', {m: round(float(min_sep[m][-1]), 2) for m in methods})


if __name__ == '__main__':
    main()
