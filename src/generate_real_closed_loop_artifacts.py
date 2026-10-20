import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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

DT = 0.25
HORIZON = 6
STEPS = 180
U_MAX = 1.7
D_MIN = 1.10
SENSE_RADIUS = 6.0
CARE_STALE_STEPS = 2
LOSS_PROB = 0.14
MAX_DELAY = 2
NOFLY_CENTER = np.array([6.0, 0.0])
NOFLY_RADIUS = 1.7

COLORS = {
    'Full Hybrid': '#0B84F3',
    'Rule-Based Semantic': '#8E44AD',
    'w/o CARE': '#E45756',
}
ROW_END = r"\\"


@dataclass
class MethodCfg:
    semantic_mode: str
    care_enabled: bool
    validator_strength: float


METHODS: Dict[str, MethodCfg] = {
    'Full Hybrid': MethodCfg('contextual', True, 1.0),
    'Rule-Based Semantic': MethodCfg('rule', True, 0.92),
    'w/o CARE': MethodCfg('contextual', False, 1.0),
}


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec))


def unit(vec: np.ndarray) -> np.ndarray:
    nrm = norm(vec)
    return vec / nrm if nrm > 1e-9 else np.zeros_like(vec)


def build_goals(n_agents: int) -> np.ndarray:
    xs = np.linspace(8.0, 11.5, n_agents)
    ys = np.linspace(-3.2, 3.2, n_agents)
    return np.column_stack([xs, ys])


def initial_positions(n_agents: int, rng: np.random.Generator) -> np.ndarray:
    xs = np.linspace(-6.5, -3.5, n_agents)
    ys = np.linspace(-3.0, 3.0, n_agents)
    pts = np.column_stack([xs, ys])
    return pts + rng.normal(0.0, 0.22, size=pts.shape)


def comm_health_from_ages(ages: np.ndarray) -> float:
    return float(np.exp(-0.45 * np.mean(ages)))


def semantic_proposal(method: str, weights: np.ndarray, step: int, comm_health: float, n_agents: int) -> np.ndarray:
    proposal = weights.copy()
    if step == 45:
        if METHODS[method].semantic_mode == 'contextual':
            proposal[-1] += 0.34 + 0.10 * (1.0 - comm_health)
            proposal[0] -= 0.10
            proposal[1] -= 0.08 if n_agents > 3 else 0.04
        else:
            proposal[-1] += 0.22
            proposal[0] -= 0.06
    if step == 95:
        if METHODS[method].semantic_mode == 'contextual':
            proposal[0] += 0.16
            proposal[-1] -= 0.10
        else:
            proposal[0] += 0.10
            proposal[-1] -= 0.04
    proposal = np.maximum(proposal, 0.02)
    proposal /= proposal.sum()
    return proposal


def validate_proposal(method: str, prev_weights: np.ndarray, proposal: np.ndarray, comm_health: float) -> Tuple[np.ndarray, str, float]:
    delta = float(np.linalg.norm(proposal - prev_weights, ord=1))
    confidence = METHODS[method].validator_strength * comm_health
    if delta < 0.08:
        return proposal, 'accept', confidence
    if confidence > 0.62:
        return proposal, 'accept', confidence
    if confidence > 0.45:
        projected = 0.65 * prev_weights + 0.35 * proposal
        projected /= projected.sum()
        return projected, 'project', confidence
    return prev_weights.copy(), 'reuse', confidence


def assign_targets(positions: np.ndarray, goals: np.ndarray, weights: np.ndarray) -> List[int]:
    n_agents = len(positions)
    remaining = set(range(n_agents))
    assignment = [-1] * n_agents
    order = list(np.argsort(-weights))
    for goal_idx in order:
        best_agent = max(remaining, key=lambda idx: 1.5 * weights[goal_idx] - 0.12 * norm(goals[goal_idx] - positions[idx]))
        assignment[best_agent] = goal_idx
        remaining.remove(best_agent)
    if remaining:
        for idx in remaining:
            assignment[idx] = idx
    return assignment


def project_nofly(candidate_u: np.ndarray, pos: np.ndarray) -> np.ndarray:
    next_pos = pos + DT * candidate_u
    delta = next_pos - NOFLY_CENTER
    d = norm(delta)
    if d < NOFLY_RADIUS + 0.18:
        repel = unit(delta) * (NOFLY_RADIUS + 0.18 - d + 0.05) / DT
        candidate_u = candidate_u + repel
    return candidate_u


def planner_step(agent_idx: int, pos: np.ndarray, vel: np.ndarray, goal: np.ndarray, beliefs: np.ndarray, prev_u: np.ndarray) -> np.ndarray:
    goal_term = 1.25 * (goal - pos)
    neighbor_term = np.zeros(2)
    for j, belief in enumerate(beliefs):
        if j == agent_idx:
            continue
        diff = pos - belief[:2]
        d = norm(diff)
        if d < SENSE_RADIUS:
            neighbor_term += 0.55 * unit(diff) * clip((D_MIN + 0.8 - d), 0.0, 1.6)
    nominal = (goal_term + neighbor_term + 0.25 * prev_u + 0.10 * vel) / (1.25 + 0.25 + 0.10)
    nominal = project_nofly(nominal, pos)
    speed = norm(nominal)
    if speed > U_MAX:
        nominal = nominal / speed * U_MAX
    return nominal


def step_dynamics(pos: np.ndarray, vel: np.ndarray, control: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    new_vel = 0.52 * vel + 0.48 * control
    new_pos = pos + DT * new_vel
    return new_pos, new_vel


def run_closed_loop(method: str, n_agents: int, seed: int) -> Tuple[dict, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    goals = build_goals(n_agents)
    pos = initial_positions(n_agents, rng)
    vel = np.zeros((n_agents, 2))
    beliefs = np.hstack([pos.copy(), vel.copy()])
    last_rx_step = np.zeros((n_agents, n_agents), dtype=int)
    queued_msgs: List[List[Tuple[int, np.ndarray]]] = [[] for _ in range(STEPS + MAX_DELAY + 3)]
    prev_controls = np.zeros((n_agents, 2))
    weights = np.ones(n_agents) / n_agents
    completed = np.zeros(n_agents, dtype=bool)
    validator_counts = {'accept': 0, 'project': 0, 'reuse': 0}
    min_sep = 1e9
    care_triggers = 0
    stale_events = 0
    progress_curve = []
    rows = []

    for step in range(STEPS):
        for due_sender, packet in queued_msgs[step]:
            beliefs[due_sender] = packet
            for recv in range(n_agents):
                if recv != due_sender:
                    last_rx_step[recv, due_sender] = step

        ages = np.array([step - last_rx_step[i, j] for i in range(n_agents) for j in range(n_agents) if i != j], dtype=float)
        comm_health = comm_health_from_ages(ages)

        proposed = semantic_proposal(method, weights, step, comm_health, n_agents)
        semantic_active = step in (45, 95)
        validator_action = 'none'
        validator_conf = comm_health
        if semantic_active:
            weights, validator_action, validator_conf = validate_proposal(method, weights, proposed, comm_health)
            validator_counts[validator_action] += 1

        for i in range(n_agents):
            for j in range(n_agents):
                if i == j:
                    beliefs[j, :2] = pos[j]
                    beliefs[j, 2:] = vel[j]
                    continue
                age = step - last_rx_step[i, j]
                if METHODS[method].care_enabled and age >= CARE_STALE_STEPS:
                    beliefs[j, :2] = beliefs[j, :2] + DT * beliefs[j, 2:]
                    care_triggers += 1
                    stale_events += 1

        target_choices = assign_targets(pos, goals, weights)

        controls = np.zeros_like(pos)
        for i in range(n_agents):
            controls[i] = planner_step(i, pos[i], vel[i], goals[target_choices[i]], beliefs.copy(), prev_controls[i])

        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                pred_i = pos[i] + DT * controls[i]
                pred_j = beliefs[j, :2] + DT * beliefs[j, 2:] if METHODS[method].care_enabled else beliefs[j, :2]
                diff = pred_i - pred_j
                d = norm(diff)
                if d < D_MIN + 0.22:
                    corr = 0.95 * unit(diff if d > 1e-6 else pos[i] - pos[j] + np.array([0.1, 0.0])) * (D_MIN + 0.22 - d) / DT
                    controls[i] += corr
                    controls[j] -= corr
        speeds = np.linalg.norm(controls, axis=1)
        for i in range(n_agents):
            if speeds[i] > U_MAX:
                controls[i] = controls[i] / speeds[i] * U_MAX

        for i in range(n_agents):
            pos[i], vel[i] = step_dynamics(pos[i], vel[i], controls[i])
        prev_controls = controls.copy()

        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                min_sep = min(min_sep, norm(pos[i] - pos[j]))

        for goal_idx in range(n_agents):
            if any(norm(pos[i] - goals[goal_idx]) < 0.7 for i in range(n_agents)):
                completed[goal_idx] = True
        progress = float(completed.sum() / n_agents)
        progress_curve.append(progress)

        for sender in range(n_agents):
            packet = np.hstack([pos[sender], vel[sender]])
            for recv in range(n_agents):
                if sender == recv:
                    continue
                if rng.random() < LOSS_PROB:
                    continue
                delay = int(rng.integers(0, MAX_DELAY + 1))
                queued_msgs[step + delay].append((sender, packet.copy()))

        rows.append({
            'step': step,
            'method': method,
            'n_agents': n_agents,
            'seed': seed,
            'mission_completion': progress,
            'min_sep_so_far': min_sep,
            'mean_info_age': float(np.mean(ages)) if len(ages) else 0.0,
            'validator_action': validator_action,
            'validator_confidence': validator_conf,
            'care_trigger_count_cum': care_triggers,
            'stale_event_count_cum': stale_events,
            'semantic_trigger': int(semantic_active),
            'goal_weight_shift': float(np.linalg.norm(proposed - weights, ord=1)) if semantic_active else 0.0,
            'mean_speed': float(np.mean(np.linalg.norm(vel, axis=1))),
            'disagreement_proxy': float(np.mean([norm(pos[i] - beliefs[i, :2]) for i in range(n_agents)])),
        })

    log_df = pd.DataFrame(rows)
    success = float(progress_curve[-1] >= 0.95 and min_sep >= D_MIN - 0.05)
    summary = {
        'method': method,
        'n_agents': n_agents,
        'seed': seed,
        'success': success,
        'final_completion': progress_curve[-1],
        'min_separation': min_sep,
        'mean_info_age': float(log_df['mean_info_age'].mean()),
        'mean_disagreement': float(log_df['disagreement_proxy'].mean()),
        'validator_accept_rate': validator_counts['accept'] / max(sum(validator_counts.values()), 1),
        'validator_project_rate': validator_counts['project'] / max(sum(validator_counts.values()), 1),
        'validator_reuse_rate': validator_counts['reuse'] / max(sum(validator_counts.values()), 1),
        'care_triggers': care_triggers,
        'stale_events': stale_events,
    }
    return summary, log_df


def to_pm(series: pd.Series, scale: float = 1.0, digits: int = 1) -> str:
    return f"{series.mean() * scale:.{digits}f} $\\pm$ {series.std(ddof=0) * scale:.{digits}f}"


def build_outputs():
    summaries = []
    representative_logs = {}
    for n_agents in (3, 5):
        for method in METHODS:
            for seed in range(12):
                summary, log_df = run_closed_loop(method, n_agents=n_agents, seed=900 + 37 * n_agents + seed)
                summaries.append(summary)
                if seed == 0:
                    stem = f"real_closed_loop_{method.lower().replace(' ', '_').replace('/', '_')}_{n_agents}a"
                    log_df.to_csv(ART / f"{stem}.csv", index=False)
                    representative_logs[(method, n_agents)] = log_df
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(ART / 'real_closed_loop_summary.csv', index=False)

    rep = representative_logs[('Full Hybrid', 5)].copy()
    injection_log = rep[['step', 'validator_action', 'validator_confidence', 'care_trigger_count_cum', 'stale_event_count_cum', 'semantic_trigger', 'mission_completion', 'mean_info_age', 'disagreement_proxy']]
    injection_log.to_csv(ART / 'real_care_validator_injection_log.csv', index=False)

    # table
    rows = []
    for n_agents in (3, 5):
        for method in METHODS:
            sub = summary_df[(summary_df['n_agents'] == n_agents) & (summary_df['method'] == method)]
            rows.append(
                f"{n_agents} & {method} & {to_pm(sub['success'], scale=100.0, digits=1)} & {to_pm(sub['final_completion'], scale=100.0, digits=1)} & {to_pm(sub['min_separation'], digits=2)} & {to_pm(sub['mean_info_age'], digits=2)} & {to_pm(sub['mean_disagreement'], digits=3)} {ROW_END}"
            )
    table = (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\caption{Minimum Real Closed-Loop Multi-Seed Results for 3-Agent and 5-Agent Teams}\n"
        "\\label{tab:real_closed_loop}\n"
        "\\setlength{\\tabcolsep}{3.8pt}\n"
        "\\begin{tabular}{>{\\centering\\arraybackslash}p{0.07\\linewidth} >{\\raggedright\\arraybackslash}p{0.18\\linewidth} >{\\centering\\arraybackslash}p{0.13\\linewidth} >{\\centering\\arraybackslash}p{0.13\\linewidth} >{\\centering\\arraybackslash}p{0.12\\linewidth} >{\\centering\\arraybackslash}p{0.12\\linewidth} >{\\centering\\arraybackslash}p{0.12\\linewidth}}\n"
        "\\toprule\n"
        "Agents & Method & Success (\\%) & Final completion (\\%) & Min. sep. (m) & Mean info age & Mean disagreement \\\\ \n"
        "\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}"
    )
    (TABLES / 'generated_real_closed_loop_table.tex').write_text(table + "\n", encoding='utf-8')

    # figure
    fig, axes = plt.subplots(2, 1, figsize=(9.8, 6.4), dpi=240, sharex=True)
    for method in METHODS:
        log_df = representative_logs[(method, 5)]
        axes[0].plot(log_df['step'], log_df['mission_completion'], linewidth=2.2, color=COLORS[method], label=method)
        axes[1].plot(log_df['step'], log_df['mean_info_age'], linewidth=2.0, color=COLORS[method], label=method)
    for ax in axes:
        for step in (45, 95):
            ax.axvline(step, linestyle='--', color='#777777', linewidth=0.9)
        ax.axvspan(78, 102, color='#f8d7da', alpha=0.15)
    axes[0].set_ylabel('Mission completion')
    axes[1].set_ylabel('Mean info age')
    axes[1].set_xlabel('Closed-loop step')
    axes[0].legend(ncol=3, fontsize=8, frameon=True, loc='lower right')
    fig.tight_layout()
    fig.savefig(RESULTS / 'fig_real_closed_loop_summary.png', bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(9.6, 7.0), dpi=240, sharex=True)
    action_map = {'none': 0, 'accept': 1, 'project': 2, 'reuse': 3}
    axes[0].step(injection_log['step'], [action_map[a] for a in injection_log['validator_action']], where='mid', color='#4C78A8', linewidth=1.8)
    axes[0].set_yticks([0, 1, 2, 3])
    axes[0].set_yticklabels(['none', 'accept', 'project', 'reuse'])
    axes[0].set_ylabel('Validator')
    axes[1].plot(injection_log['step'], injection_log['care_trigger_count_cum'], color='#E45756', linewidth=2.0)
    axes[1].plot(injection_log['step'], injection_log['stale_event_count_cum'], color='#B279A2', linewidth=1.7, linestyle='--')
    axes[1].set_ylabel('CARE / stale')
    axes[2].plot(injection_log['step'], injection_log['mission_completion'], color='#0B84F3', linewidth=2.1, label='completion')
    axes[2].plot(injection_log['step'], injection_log['disagreement_proxy'], color='#2A9D8F', linewidth=1.8, label='disagreement')
    axes[2].set_ylabel('Mission / disagreement')
    axes[2].set_xlabel('Closed-loop step')
    axes[2].legend(fontsize=8, frameon=True)
    for ax in axes:
        for step in (45, 95):
            ax.axvline(step, linestyle='--', color='#777777', linewidth=0.9)
    fig.tight_layout()
    fig.savefig(RESULTS / 'fig_real_injection_log.png', bbox_inches='tight')
    plt.close(fig)

    manifest = {
        'summary_rows': int(len(summary_df)),
        'injection_rows': int(len(injection_log)),
        'representative_logs': sorted([str(p.name) for p in ART.glob('real_closed_loop_*.csv')]),
    }
    (ART / 'real_closed_loop_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    build_outputs()
