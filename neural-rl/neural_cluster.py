#!/usr/bin/env python3
"""Neural (GPU-trained) variant of cluster.py. Lives in <package>/neural/ as an overlay.

Rollouts are unchanged: full normal games on the stock simulator through HTTP /predict, the
hash-checked eat/rest baseline, the same protected modes, options and rewards. What changes:
the Q table becomes an MLP ensemble (policy_neural.py), games only record transitions, and the
reducer is a GPU fitted-Q trainer (train_gpu.py) replaying the transitions of EVERY round so far.
Planning, workers, completion checks and summaries are the package's own functions.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import ast
import json
import os
import signal
import subprocess
import sys
import threading
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]
import cluster  # noqa: E402
import common  # noqa: E402
import episode  # noqa: E402
import train_gpu  # noqa: E402
from common import digest, lock, read_json, value_hash, write_json, write_new  # noqa: E402

NEURAL_FILES = ('neural_cluster.py', 'policy_neural.py', 'train_gpu.py')
_package_fingerprint = common.code_fingerprint


def code_fingerprint():
    return value_hash(dict(package=_package_fingerprint(), neural={n: digest(HERE / n) for n in NEURAL_FILES}))


def validate_model(model):
    if set(model) != {'q', 'counts', 'net'} or model['q'] or model['counts']:
        raise ValueError('Neural model requires empty q/counts and a net')
    train_gpu.validate_net(model['net'])
    return model


def build_source(model, *, training=False, learner_seed=123001, epsilon=0.0, start_time=None):
    validate_model(model)
    baseline = (ROOT / 'policy/baseline.py').read_text()
    if digest(ROOT / 'policy/baseline.py') != common.BASELINE_SHA:
        raise ValueError('Preserved baseline has changed')
    settings = read_json(ROOT / 'policy/settings.json')
    settings.update(training=training, epsilon=epsilon if training else 0., random_seed=learner_seed,
                    start_time=(600. if training else 900.) if start_time is None else start_time,
                    neural_uncertainty=.5)
    insertion = 'RL_SETTINGS = ' + repr(settings) + '\nRL_MODEL = ' + repr(model) + '\n\n'
    for part in (ROOT / 'policy/hybrid.py', ROOT / 'policy/foraging.py', HERE / 'policy_neural.py'):
        insertion += part.read_text() + '\n\n'
    insertion += "def make_policy():\n    return NeuralRLPolicy('appetite', **CONFIG)"
    marker = "def make_policy():\n    return AppetitePolicy('appetite', **CONFIG)"
    if baseline.count(marker) != 1:
        raise ValueError('Baseline insertion point is ambiguous')
    source = baseline.replace(marker, insertion)
    ast.parse(source)
    return source, settings


# The package's planner/worker/summary code is reused as is; only these three names differ.
cluster.code_fingerprint = code_fingerprint
cluster.validate_model = validate_model
episode.build_source = build_source


def initialize(run, training, validation, test, *, epsilon, hidden, members, steps, batch, lr, target_sync):
    sets = [set(x) for x in (training, validation, test)]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError('Training, validation and test seeds overlap')
    if (sets[1] | sets[2]) & set(range(20)):
        raise ValueError('Seeds 0-19 are historical development seeds, not fresh holdouts')
    run = Path(run).resolve()
    with lock(run / '.initialize.lock'):
        config = dict(schema=1, train_seeds=training, validation_seeds=validation, test_seeds=test,
                      epsilon=epsilon, start_from_scratch=True,
                      policy='preserved eat/rest + bounded foraging search-speed RL, neural Q ensemble',
                      train_start_time=600., evaluation_start_time=900.,
                      historical_development_seeds=list(range(20)),
                      trainer=dict(hidden=hidden, members=members, steps=steps, batch=batch, lr=lr,
                                   target_sync=target_sync),
                      code_fingerprint=code_fingerprint(), simulator_fingerprint=common.check_simulator())
        write_json(run / 'config.json', config)
        net = dict(trained=False, features=train_gpu.FEATURES, hidden=hidden, members=[])
        write_json(run / 'checkpoints/initial.json',
                   dict(schema=1, round=-1, model=dict(q={}, counts={}, net=net), lineage=[], training_seeds=[],
                        added_transitions=0, total_new_transitions=0,
                        provenance='untrained network: round 0 explores uniformly, evaluation is pure baseline'))
    return run


def round_transitions(path, plan):
    rows, records = [], []
    for job in plan['jobs']:
        done = cluster.completed(path, plan, job)
        if done is None:
            raise ValueError(f'Incomplete round: {job["job_id"]}; rerun missing workers')
        attempt, result = done
        transitions = read_json(attempt / 'transitions.json')
        if len(transitions) != result['transitions'] or len(transitions) != result['rl_stats'].get('q_updates', 0):
            raise ValueError('Transition count mismatch')
        rows.extend(transitions)
        records.append(dict(job_id=job['job_id'], seed=job['seed'], transitions=len(transitions),
                            sha256=digest(attempt / 'transitions.json'), survival=result['survival_seconds']))
    return rows, records


def merge(path, device='auto'):
    path, plan, run, model = cluster.load_plan(path)
    if plan['kind'] != 'train':
        raise ValueError('Evaluation experience must never be merged into training')
    config = read_json(run / 'config.json')
    target = cluster.checkpoint_path(run, plan['round'])
    with lock(run / '.merge.lock'):
        parent = read_json(path.parent / plan['checkpoint'])
        plan_hash = digest(path)
        if plan_hash in parent['lineage']:
            raise ValueError('This round was already consumed by its parent')
        rows, records = round_transitions(path, plan)
        if target.exists():
            # GPU training is not bit-reproducible, so a finished merge is reused, never recomputed.
            saved = read_json(target)
            if saved['plan_sha256'] != plan_hash or saved['episodes'] != records:
                raise ValueError('Existing checkpoint conflicts with requested merge')
            print(json.dumps(dict(event='checkpoint_reused', checkpoint=str(target))), flush=True)
            return target
        replay = list(rows)
        for number in range(plan['round']):  # off-policy learner: replay every earlier round too
            earlier = run / 'rounds' / f'{number:04d}' / 'plan.json'
            if digest(earlier) != parent['lineage'][number]:
                raise ValueError(f'Round {number} plan is not the one this lineage consumed')
            replay.extend(round_transitions(earlier, read_json(earlier))[0])
        train_gpu.validate_transitions(replay)
        net, stats = train_gpu.train(replay, model['net'], seed=17483 + plan['round'], device=device, **config['trainer'])
        checkpoint = dict(schema=1, round=plan['round'], model=validate_model(dict(q={}, counts={}, net=net)),
                          plan_sha256=plan_hash, parent_sha256=plan['checkpoint_sha256'], replay=stats,
                          episodes=records, lineage=parent['lineage'] + [plan_hash], added_transitions=len(rows),
                          total_new_transitions=parent['total_new_transitions'] + len(rows),
                          training_seeds=sorted(set(parent['training_seeds']) | {j['seed'] for j in plan['jobs']}),
                          provenance='GPU fitted-Q ensemble over all complete normal training games so far')
        write_json(target, checkpoint)
        print(json.dumps(dict(event='merged', checkpoint=str(target), new_transitions=len(rows),
                              replayed=len(replay), **stats)), flush=True)
        return target


def run_local(path, workers):
    """cluster.run_local, but the workers are launched through this script."""
    path, plan, _, _ = cluster.load_plan(path)
    if workers < 1:
        raise ValueError('workers must be positive')
    active, mutex = set(), threading.Lock()
    logs = path.parent / 'worker_logs'
    logs.mkdir(exist_ok=True)

    def launch(index):
        logpath = logs / f'{index:05d}_{uuid.uuid4().hex}.log'
        with logpath.open('x') as output:
            process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), 'worker',
                                        '--plan', str(path), '--index', str(index)],
                                       stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            with mutex:
                active.add(process)
            try:
                code = process.wait()
                if code:
                    raise RuntimeError(f'Worker {index} exited {code}; see {logpath}')
                print(json.dumps(dict(event='worker_finished', index=index, log=str(logpath))), flush=True)
            finally:
                with mutex:
                    active.discard(process)
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [pool.submit(launch, j['index']) for j in plan['jobs']
                   if cluster.completed(path, plan, j) is None]
        for future in as_completed(futures):
            future.result()
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        with mutex:
            for process in active:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def export(checkpoint, output):
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    source, settings = build_source(read_json(checkpoint)['model'])
    write_new(output / 'survival_agent.py', source)
    write_new(output / 'requirements-agent.txt', (ROOT / 'requirements-agent.txt').read_bytes())
    write_json(output / 'export.json', dict(checkpoint_sha256=digest(checkpoint), settings=settings,
               controller_sha256=digest(output / 'survival_agent.py'), learning_enabled=False,
               note='Experimental frozen neural hybrid; export does not imply it beats baseline.'))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('init')
    p.add_argument('--run', required=True)
    p.add_argument('--train-seeds', default='1000:1256')
    p.add_argument('--validation-seeds', default='2000:2016')
    p.add_argument('--test-seeds', default='3000:3032')
    p.add_argument('--epsilon', type=float, default=.3)
    p.add_argument('--hidden', type=int, default=32)
    p.add_argument('--members', type=int, default=5)
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--batch', type=int, default=4096)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--target-sync', type=int, default=500)
    p = sub.add_parser('plan-eval')
    p.add_argument('--run', required=True)
    p.add_argument('--split', choices=['validation', 'test'], required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--checkpoint')
    group.add_argument('--baseline', action='store_true')
    p = sub.add_parser('worker')
    p.add_argument('--plan', required=True)
    p.add_argument('--index', type=int, required=True)
    p = sub.add_parser('run')
    p.add_argument('--plan', required=True)
    p.add_argument('--workers', type=int, default=4)
    p = sub.add_parser('train')
    p.add_argument('--run', required=True)
    p.add_argument('--rounds', type=int, default=1, help='Total rounds, including those already completed')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    p = sub.add_parser('summary')
    p.add_argument('--plan', required=True)
    p.add_argument('--baseline-plan')
    p = sub.add_parser('export')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'init':
        print(initialize(args.run, cluster.seeds(args.train_seeds), cluster.seeds(args.validation_seeds),
                         cluster.seeds(args.test_seeds), epsilon=args.epsilon, hidden=args.hidden,
                         members=args.members, steps=args.steps, batch=args.batch, lr=args.lr,
                         target_sync=args.target_sync))
    elif args.command == 'plan-eval':
        print(cluster.make_plan(args.run, checkpoint=args.checkpoint, split=args.split, baseline=args.baseline))
    elif args.command == 'worker':
        cluster.worker(args.plan, args.index)
    elif args.command == 'run':
        run_local(args.plan, args.workers)
    elif args.command == 'train':
        if args.rounds < 1:
            parser.error('--rounds must be positive')
        for number in range(args.rounds):
            path = cluster.make_plan(args.run, number=number)
            run_local(path, args.workers)
            merge(path, args.device)
    elif args.command == 'summary':
        print(json.dumps(cluster.summarize(args.plan, args.baseline_plan), indent=2))
    elif args.command == 'export':
        print(export(args.checkpoint, args.output))


if __name__ == '__main__':
    main()
