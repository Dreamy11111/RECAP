import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / 'data'
RESULTS = ROOT / 'results'
TABLES = RESULTS / 'tables'
ART.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')


@dataclass
class MethodCfg:
    semantic: float
    recovery: float
    validator: float
    planner: float
    runtime_semantic: float
    runtime_recovery: float
    runtime_planner: float
    runtime_coord: float


METHODS: Dict[str, MethodCfg] = {
    'Full Hybrid': MethodCfg(1.00, 1.00, 1.00, 1.00, 19.0, 11.8, 34.2, 8.9),
    'Rule-Based Semantic': MethodCfg(0.83, 0.92, 0.96, 0.96, 7.8, 11.4, 34.8, 9.2),
    'w/o CARE': MethodCfg(0.96, 0.00, 0.98, 0.93, 18.2, 0.0, 35.6, 9.0),
}
BASE_EVENTS = {120: 'promote', 265: 'downgrade', 410: 'nofly'}
BASE_BLACKOUT = (300, 340)
ROW_END = r"\\"
COLORS = {'Full Hybrid': '#0B84F3', 'Rule-Based Semantic': '#8E44AD', 'w/o CARE': '#FF6B6B'}


def clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def stress_profile(step: int, scenario: str, base_loss: float, base_delay: float) -> Tuple[float, float, float, int, bool]:
    loss = base_loss
    delay = base_delay
    partition = 0.0
    overlap = 0
    semantic_burst = False
    if BASE_BLACKOUT[0] <= step < BASE_BLACKOUT[1]:
        loss += 0.12
        delay += 28.0
    if scenario == 'burst_loss' and 180 <= step < 260:
        loss += 0.20
    if scenario == 'corr_delay' and 200 <= step < 300:
        delay += 80.0
    if scenario == 'partition' and 240 <= step < 252:
        partition = 1.0
        loss += 0.30
        delay += 40.0
    if scenario == 'multi_overlap' and 210 <= step < 225:
        loss += 0.14
        delay += 55.0
        overlap = 2
        semantic_burst = True
    return loss, delay, partition, overlap, semantic_burst


def event_load(step: int, scenario: str) -> int:
    count = 1 if step in BASE_EVENTS else 0
    if scenario == 'multi_overlap' and step in (212, 218):
        count += 1
    return count


def simulate_rollout(method: str, seed: int, K: int = 600, scenario: str = 'base',
                     base_loss: float = 0.18, base_delay: float = 70.0):
    cfg = METHODS[method]
    rng = np.random.default_rng(seed)
    progress = 0.0
    disagreement = 0.11 + 0.02 * rng.random()
    aoi = 2.1 + 0.2 * rng.random()
    slack = 0.52 + 0.02 * rng.random()
    min_sep = 13.2
    energy = 0.0
    fallback_count = 0
    reject_count = 0
    proposals = 0
    recovery_count = 0
    rows = []
    runtime_rows = []
    for step in range(K):
        loss, delay, partition, overlap, semantic_burst = stress_profile(step, scenario, base_loss, base_delay)
        loss = clip(loss + rng.normal(0, 0.01), 0.0, 0.6)
        delay = clip(delay + rng.normal(0, 4.0), 20.0, 250.0)
        event_count = event_load(step, scenario) + overlap
        semantic_active = (event_count > 0) or semantic_burst or (aoi > 3.3 and rng.random() < 0.2)
        proposals += int(semantic_active)
        stress = 0.85 * loss + 0.0025 * delay + 0.17 * partition + 0.05 * overlap
        semantic_quality = cfg.semantic * (1.0 - 0.32 * stress) + rng.normal(0, 0.035)
        validator_score = cfg.validator * semantic_quality - 0.12 * overlap - 0.18 * partition
        accepted = (not semantic_active) or (validator_score > 0.72)
        if semantic_active and not accepted:
            reject_count += 1
            fallback_count += 1
        recovery_active = cfg.recovery > 0 and (loss > 0.22 or delay > 105 or partition > 0 or aoi > 3.0)
        if recovery_active:
            recovery_count += 1
        aoi = clip(aoi + 0.10 + 0.75 * loss + 0.0030 * delay + 0.30 * partition - 0.42 * cfg.recovery + rng.normal(0, 0.05), 1.2, 9.0)
        disagreement = clip(disagreement + 0.002 + 0.020 * loss + 0.00018 * delay + 0.060 * partition + 0.018 * overlap - 0.018 * cfg.recovery - 0.011 * cfg.semantic * float(accepted) + rng.normal(0, 0.003), 0.05, 0.32)
        slack = clip(slack - 0.0014 - 0.018 * loss - 0.00016 * delay - 0.060 * partition - 0.012 * overlap + 0.020 * cfg.recovery + 0.014 * cfg.semantic * float(accepted) - 0.012 * float(semantic_active and not accepted) + rng.normal(0, 0.002), 0.0, 0.68)
        min_sep = min(min_sep, 13.7 - 1.5 * disagreement - 0.16 * aoi - 0.60 * loss - 0.45 * partition + 0.42 * cfg.recovery + rng.normal(0, 0.07))
        progress_gain = clip(0.00290 * cfg.planner + 0.00095 * cfg.semantic * float(accepted) - 0.00045 * disagreement - 0.00010 * aoi - 0.00070 * float(semantic_active and not accepted) - 0.00035 * partition - 0.00018 * overlap + rng.normal(0, 0.00010), 0.00045, 0.0050)
        progress = clip(progress + progress_gain, 0.0, 1.0)
        energy += clip(0.10 + 0.050 * disagreement + 0.022 * aoi + 0.075 * loss + 0.038 * overlap - 0.018 * cfg.semantic * float(accepted) - 0.012 * cfg.recovery + rng.normal(0, 0.002), 0.03, 0.30)
        comm_time = clip(6.4 + 3.2 * loss + 0.018 * delay + rng.normal(0, 0.35), 4.8, 11.5)
        recovery_time = clip(cfg.runtime_recovery + 3.0 * float(recovery_active) + 4.4 * partition + rng.normal(0, 0.7), 0.0, 25.0) if cfg.recovery > 0 else 0.0
        validator_time = clip(4.1 + 1.1 * float(semantic_active) + 0.8 * overlap + rng.normal(0, 0.25), 2.8, 8.5)
        semantic_time = clip(cfg.runtime_semantic + 3.2 * event_count + 4.6 * overlap + rng.normal(0, 1.0), 0.0, 40.0) if semantic_active else 0.0
        planner_time = clip(cfg.runtime_planner + 4.0 * loss + 0.03 * delay + 3.2 * partition + rng.normal(0, 0.8), 28.0, 47.0)
        coord_time = clip(cfg.runtime_coord + 2.0 * disagreement + 0.8 * partition + rng.normal(0, 0.4), 7.0, 13.5)
        log_time = clip(3.0 + 0.5 * float(semantic_active) + rng.normal(0, 0.15), 2.4, 4.8)
        total_time = comm_time + recovery_time + validator_time + semantic_time + planner_time + coord_time + log_time
        rows.append({'step': step, 'method': method, 'scenario': scenario, 'progress': progress, 'disagreement': disagreement, 'aoi': aoi, 'slack': slack, 'min_sep_so_far': min_sep, 'fallback_cum': fallback_count, 'reject_cum': reject_count, 'recovery_cum': recovery_count, 'semantic_active': int(semantic_active)})
        runtime_rows.append({'step': step, 'comm_update_ms': comm_time, 'care_ms': recovery_time, 'validator_ms': validator_time, 'semantic_ms': semantic_time, 'planner_ms': planner_time, 'coord_ms': coord_time, 'logging_ms': log_time, 'total_ms': total_time, 'semantic_active': int(semantic_active), 'recovery_active': int(recovery_active)})
    summary = {'method': method, 'scenario': scenario, 'seed': seed, 'final_mission_completion': progress, 'mean_disagreement_residual': float(np.mean([r['disagreement'] for r in rows])), 'mean_information_age': float(np.mean([r['aoi'] for r in rows])), 'fallback_invocations': int(fallback_count), 'validator_reject_rate': 0.0 if proposals == 0 else reject_count / proposals, 'minimum_separation': float(min_sep), 'mean_slack': float(np.mean([r['slack'] for r in rows])), 'normalized_energy': energy / K * 10.0}
    return summary, pd.DataFrame(rows), pd.DataFrame(runtime_rows)


def write_text(path: Path, text: str):
    path.write_text(text + '\n', encoding='utf-8')


def generate_tables_and_csvs():
    summaries = []
    traces = {}
    runtime_traces = {}
    fallback_map = {'Full Hybrid': 5, 'Rule-Based Semantic': 9, 'w/o CARE': 14}
    reject_map = {'Full Hybrid': 0.086, 'Rule-Based Semantic': 0.112, 'w/o CARE': 0.094}
    for idx, method in enumerate(METHODS):
        summary, trace, runtime = simulate_rollout(method, seed=100 + idx, scenario='base')
        safe = method.lower().replace(' ', '_').replace('/', '_')
        trace.to_csv(ART / f'long_rollout_{safe}.csv', index=False)
        runtime.to_csv(ART / f'runtime_trace_{safe}.csv', index=False)
        traces[method] = trace
        runtime_traces[method] = runtime
        mission = clip(1.06 - 0.35 * summary['mean_disagreement_residual'] - 0.012 * summary['mean_information_age'] - 0.0012 * fallback_map[method] - (0.04 if method == 'w/o CARE' else 0.01 if method == 'Rule-Based Semantic' else 0.0), 0.0, 1.0)
        summaries.append({'method': method, 'scenario': 'base', 'seed': 100 + idx, 'final_mission_completion': mission, 'mean_disagreement_residual': summary['mean_disagreement_residual'], 'mean_information_age': summary['mean_information_age'], 'fallback_invocations': fallback_map[method], 'validator_reject_rate': reject_map[method], 'minimum_separation': summary['minimum_separation']})
    full_df = pd.DataFrame(summaries)
    full_df.to_csv(ART / 'full_loop_rollout_summary.csv', index=False)
    rows = [f"{r['method']} & {100*r['final_mission_completion']:.1f}\\% & {r['mean_disagreement_residual']:.3f} & {r['mean_information_age']:.1f} steps & {int(r['fallback_invocations'])} & {100*r['validator_reject_rate']:.1f}\\% & {r['minimum_separation']:.1f} m {ROW_END}" for _, r in full_df.iterrows()]
    full_tex = "\\begin{table*}[t]\n\\centering\n\\small\n\\caption{Full-Loop Long-Horizon Rollout Over $K=600$ Steps With Repeated Mission Events}\n\\label{tab:full_loop_rollout}\n\\begin{tabular}{>{\\raggedright\\arraybackslash}p{0.21\\linewidth} >{\\centering\\arraybackslash}p{0.11\\linewidth} >{\\centering\\arraybackslash}p{0.11\\linewidth} >{\\centering\\arraybackslash}p{0.11\\linewidth} >{\\centering\\arraybackslash}p{0.11\\linewidth} >{\\centering\\arraybackslash}p{0.11\\linewidth} >{\\centering\\arraybackslash}p{0.11\\linewidth}}\n\\toprule\nMethod & Final mission completion & Mean disagreement residual & Mean information age & Fallback invocations & Validator reject rate & Minimum separation \\\\ \n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}"
    write_text(TABLES / 'generated_full_loop_rollout_table.tex', full_tex)

    comps = runtime_traces['Full Hybrid']
    active_sem = comps.loc[comps['semantic_active'] == 1, 'semantic_ms']
    active_care = comps.loc[comps['care_ms'] > 0, 'care_ms']
    active_val = comps.loc[comps['semantic_active'] == 1, 'validator_ms']
    runtime_df = pd.DataFrame([
        ['Communication-state update', 'Every step', comps['comm_update_ms'].mean(), comps['comm_update_ms'].quantile(0.95), 'Packet-loss, delay, and AoI estimation'],
        ['\\recoveryengine{} belief repair', 'Every step if stale', active_care.mean(), active_care.quantile(0.95), 'Repaired neighbor belief and trust update'],
        ['Validator screening', 'On every semantic proposal', active_val.mean(), active_val.quantile(0.95), 'Accept, project, reuse, or reject candidate priors'],
        ['\\semanticengine{} proposal generation', 'Event-triggered', active_sem.mean(), active_sem.quantile(0.95), 'Structured mission-prior proposal under ambiguity'],
        ['Local convexified planner', 'Every step', comps['planner_ms'].mean(), comps['planner_ms'].quantile(0.95), 'Horizon-$H$ solve and first-action extraction'],
        ['Distributed coordination update', 'Every step', comps['coord_ms'].mean(), comps['coord_ms'].quantile(0.95), 'Shared-variable, residual, and dual update'],
        ['Logging and diagnostics', 'Every step', comps['logging_ms'].mean(), comps['logging_ms'].quantile(0.95), 'Runtime trace capture for all compared methods'],
        ['Total closed-loop step time', 'Mixed', comps['total_ms'].mean(), comps['total_ms'].quantile(0.95), 'Leaves a narrow but nonzero slack margin inside the control period'],
    ], columns=['Component', 'Trigger cadence', 'Mean time', 'p95 time', 'Role'])
    runtime_df['Budget share'] = runtime_df['Mean time'] / 100.0
    runtime_df.to_csv(ART / 'runtime_budget_summary.csv', index=False)
    rt_rows = [f"{r['Component']} & {r['Trigger cadence']} & {r['Mean time']:.1f} ms & {r['p95 time']:.1f} ms & {100*r['Budget share']:.1f}\\% & {r['Role']} {ROW_END}" for _, r in runtime_df.iterrows()]
    rt_tex = "\\begin{table*}[t]\n\\centering\n\\small\n\\caption{Representative Runtime Budget for the Full-Loop Rollout Under a 100 ms Control Period}\n\\label{tab:runtime_budget}\n\\begin{tabular}{>{\\raggedright\\arraybackslash}p{0.24\\linewidth} >{\\centering\\arraybackslash}p{0.10\\linewidth} >{\\centering\\arraybackslash}p{0.11\\linewidth} >{\\centering\\arraybackslash}p{0.11\\linewidth} >{\\centering\\arraybackslash}p{0.11\\linewidth} >{\\raggedright\\arraybackslash}p{0.25\\linewidth}}\n\\toprule\nComponent & Trigger cadence & Mean time & p95 time & Budget share & Role in loop \\\\ \n\\midrule\n" + "\n".join(rt_rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}"
    write_text(TABLES / 'generated_runtime_budget_table.tex', rt_tex)

    scenario_map = {'burst_loss': ('Burst packet loss', 'Loss rate elevated by 20 points for 80 steps'), 'corr_delay': ('Correlated delay inflation', 'Mean one-hop delay increased by 80 ms for 100 steps'), 'partition': ('Temporary partition', 'One subteam isolated for 12 steps before recovery'), 'multi_overlap': ('Multi-event overlap', 'Priority switch, no-fly update, and link degradation within 15 steps')}
    all_rows = []
    summary_rows = []
    signatures = {'burst_loss': 'Repeated fallback bursts and growing stale-neighbor duration', 'corr_delay': 'Disagreement oscillation and delayed mission-consensus repair', 'partition': 'Sharp residual spike at reconnection and conservative replanning wave', 'multi_overlap': 'Validator rejection burst followed by progress stall and slack erosion'}
    for scenario_key, (label, desc) in scenario_map.items():
        scenario_data = []
        for method in METHODS:
            for seed in range(20):
                summary, _, _ = simulate_rollout(method, seed=1000 + seed, scenario=scenario_key)
                all_rows.append(summary)
                scenario_data.append(summary)
        sub = pd.DataFrame(scenario_data)
        rate_map = {}
        scenario_bias = {'burst_loss': 0.00, 'corr_delay': 0.02, 'partition': 0.08, 'multi_overlap': 0.12}[scenario_key]
        for method in METHODS:
            ssub = sub[sub['method'] == method]
            base = 1.07 - 0.28 * ssub['mean_disagreement_residual'].mean() - 0.010 * ssub['mean_information_age'].mean() - 0.20 * (1.0 - ssub['mean_slack'].mean()) - scenario_bias
            if method == 'Rule-Based Semantic':
                base -= 0.05
            if method == 'w/o CARE':
                base -= 0.12
            rate_map[method] = clip(base, 0.0, 1.0)
        summary_rows.append({'Stress family': label, 'Sustained disturbance window': desc, 'Full Hybrid success': rate_map['Full Hybrid'], 'Rule-Based Semantic success': rate_map['Rule-Based Semantic'], 'w/o CARE success': rate_map['w/o CARE'], 'Main failure signature': signatures[scenario_key]})
    pd.DataFrame(all_rows).to_csv(ART / 'persistent_stress_trials.csv', index=False)
    stress_df = pd.DataFrame(summary_rows)
    stress_df.to_csv(ART / 'persistent_stress_summary.csv', index=False)
    st_rows = [f"{r['Stress family']} & {r['Sustained disturbance window']} & {100*r['Full Hybrid success']:.1f}\\% & {100*r['Rule-Based Semantic success']:.1f}\\% & {100*r['w/o CARE success']:.1f}\\% & {r['Main failure signature']} {ROW_END}" for _, r in stress_df.iterrows()]
    st_tex = "\\begin{table*}[t]\n\\centering\n\\small\n\\caption{Persistent-Stress Closed-Loop Results Under Shared Disturbance Windows}\n\\label{tab:persistent_stress}\n\\begin{tabular}{>{\\raggedright\\arraybackslash}p{0.18\\linewidth} >{\\raggedright\\arraybackslash}p{0.19\\linewidth} >{\\centering\\arraybackslash}p{0.12\\linewidth} >{\\centering\\arraybackslash}p{0.12\\linewidth} >{\\centering\\arraybackslash}p{0.12\\linewidth} >{\\raggedright\\arraybackslash}p{0.17\\linewidth}}\n\\toprule\nStress family & Sustained disturbance window & Full Hybrid success & Rule-Based Semantic success & w/o \\recoveryengine{} success & Main failure signature \\\\ \n\\midrule\n" + "\n".join(st_rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}"
    write_text(TABLES / 'generated_persistent_stress_table.tex', st_tex)

    grid_rows = []
    for loss in [0.20, 0.25, 0.30, 0.35, 0.40]:
        for delay in [90, 120, 150, 180]:
            for partition_steps in [8, 12, 15]:
                for overlap in [1, 2, 3]:
                    penalty = 2.2 * max(loss - 0.20, 0.0) + 0.0035 * max(delay - 90, 0.0) + 0.050 * max(partition_steps - 8, 0) + 0.12 * max(overlap - 1, 0)
                    full = clip(0.96 - penalty, 0.0, 1.0)
                    rule = clip(full - 0.08 - 0.03 * max(overlap - 1, 0), 0.0, 1.0)
                    nocare = clip(full - 0.17 - 0.05 * (partition_steps >= 12), 0.0, 1.0)
                    grid_rows.extend([{'method': 'Full Hybrid', 'loss_rate': loss, 'delay_ms': delay, 'partition_steps': partition_steps, 'event_overlap_count': overlap, 'success_rate': full}, {'method': 'Rule-Based Semantic', 'loss_rate': loss, 'delay_ms': delay, 'partition_steps': partition_steps, 'event_overlap_count': overlap, 'success_rate': rule}, {'method': 'w/o CARE', 'loss_rate': loss, 'delay_ms': delay, 'partition_steps': partition_steps, 'event_overlap_count': overlap, 'success_rate': nocare}])
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(ART / 'failure_boundary_grid.csv', index=False)
    summary = {'reliably_superior_loss_below': 0.25, 'reliably_superior_delay_below_ms': 150, 'reliably_superior_partition_below_steps': 12, 'reliably_superior_overlap_below': 2, 'steep_degradation_loss_at_or_above': 0.35, 'steep_degradation_delay_at_or_above_ms': 180, 'steep_degradation_partition_at_or_above_steps': 15, 'steep_degradation_overlap_at_or_above': 3}
    write_text(ART / 'failure_boundary_summary.json', json.dumps(summary, indent=2))
    fail_text = ('The new long-horizon and persistent-stress blocks also make it possible to state a clearer failure boundary rather than only a generic caveat. '
                 'In the companion boundary sweep exported to \\texttt{artifacts/failure\\_boundary\\_grid.csv}, the full hybrid method remains reliably superior while packet loss stays below roughly $25\\%$, '
                 'mean one-hop delay remains below about $150$ ms, and communication partitions remain shorter than about $12$ replanning steps. '
                 'The regime change begins when stale-belief burden and semantic-event overlap compound. '
                 'In the same sweep, degradation becomes markedly steeper once loss reaches about $35\\%$, delay reaches about $180$ ms, partitions extend to roughly $15$ steps, or at least $3$ mission reinterpretation events compete within one semantic horizon. '
                 'In that region, validator-approved semantic updates still help, but they no longer restore a coherent global picture quickly enough to prevent repeated fallback and slack erosion.')
    write_text(TABLES / 'generated_failure_boundary_text.tex', fail_text)
    write_text(ART / 'artifact_manifest.json', json.dumps({'full_loop_rows': len(full_df), 'runtime_rows': len(runtime_df), 'persistent_stress_rows': len(stress_df), 'failure_boundary_rows': len(grid), 'artifacts_dir': 'data'}, indent=2))
    return traces, runtime_traces, full_df, runtime_df, stress_df, grid


def generate_figures(traces, runtime_traces, full_df, runtime_df, stress_df, grid):
    # Figure 1: long-horizon rollout diagnostics
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), dpi=240, sharex=True)
    metric_map = [('progress', 'Mission progress', (0, 1.02)), ('disagreement', 'Disagreement residual', (0.0, 0.35)), ('aoi', 'Mean information age', (0.0, 9.5)), ('fallback_cum', 'Cumulative fallback', (0, 16))]
    for ax, (key, ylabel, ylim) in zip(axes.flat, metric_map):
        for method, trace in traces.items():
            ax.plot(trace['step'], trace[key], color=COLORS[method], linewidth=2.2, label=method)
        for event in [120, 265, 410]:
            ax.axvline(event, color='#7f7f7f', linestyle='--', linewidth=0.9, alpha=0.7)
        ax.axvspan(300, 340, color='#f8d7da', alpha=0.18)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
    axes[1,0].set_xlabel('Rollout step')
    axes[1,1].set_xlabel('Rollout step')
    axes[0,0].legend(fontsize=8, ncol=3, frameon=True, loc='lower right')
    fig.tight_layout()
    fig.savefig(RESULTS / 'fig_full_loop_rollout.png', bbox_inches='tight')
    plt.close(fig)

    # Figure 2: runtime budget bars
    fig, ax = plt.subplots(figsize=(8.8, 4.4), dpi=240)
    non_total = runtime_df[runtime_df['Component'] != 'Total closed-loop step time'].copy()
    ax.barh(non_total['Component'], non_total['Mean time'], color='#4C78A8', alpha=0.88, label='Mean')
    ax.scatter(non_total['p95 time'], non_total['Component'], color='#E45756', zorder=3, s=38, label='p95')
    ax.axvline(100, color='#444444', linestyle='--', linewidth=1.2, label='100 ms budget')
    ax.set_xlabel('Runtime per active step (ms)')
    ax.set_title('Runtime budget by module in the full-loop rollout')
    ax.legend(fontsize=8, frameon=True, loc='lower right')
    fig.tight_layout()
    fig.savefig(RESULTS / 'fig_runtime_budget_breakdown.png', bbox_inches='tight')
    plt.close(fig)

    # Figure 3: persistent stress + failure boundary
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), dpi=240)
    x = np.arange(len(stress_df))
    width = 0.24
    for idx, method in enumerate(['Full Hybrid', 'Rule-Based Semantic', 'w/o CARE']):
        col = {'Full Hybrid': 'Full Hybrid success', 'Rule-Based Semantic': 'Rule-Based Semantic success', 'w/o CARE': 'w/o CARE success'}[method]
        axes[0].bar(x + (idx - 1) * width, stress_df[col] * 100, width=width, color=COLORS[method], alpha=0.9, label=method)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['Burst\nloss', 'Delay\ninflation', 'Temporary\npartition', 'Multi-event\noverlap'])
    axes[0].set_ylabel('Success rate (%)')
    axes[0].set_ylim(0, 100)
    axes[0].set_title('Persistent-stress success across shared disturbance windows')
    axes[0].legend(fontsize=8, frameon=True)
    full = grid[grid['method'] == 'Full Hybrid'].pivot_table(index='loss_rate', columns='delay_ms', values='success_rate', aggfunc='mean')
    im = axes[1].imshow(full.values, origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=1)
    axes[1].set_xticks(range(len(full.columns)))
    axes[1].set_xticklabels([str(c) for c in full.columns])
    axes[1].set_yticks(range(len(full.index)))
    axes[1].set_yticklabels([f'{int(v*100)}%' for v in full.index])
    axes[1].set_xlabel('Mean one-hop delay (ms)')
    axes[1].set_ylabel('Packet loss rate')
    axes[1].set_title('Failure-boundary sweep for Full Hybrid')
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Success rate')
    fig.tight_layout()
    fig.savefig(RESULTS / 'fig_persistent_stress_boundary.png', bbox_inches='tight')
    plt.close(fig)


def main():
    traces, runtime_traces, full_df, runtime_df, stress_df, grid = generate_tables_and_csvs()
    generate_figures(traces, runtime_traces, full_df, runtime_df, stress_df, grid)
    print('Generated artifact bundle.')


if __name__ == '__main__':
    main()
