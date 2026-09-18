"""Evaluation tooling: the scoring wrapper must equal the reference
``local_evaluator.Statistics`` arithmetic, denominators must be complete, and
the split must be deterministic and group-safe."""

import random

import pytest

import local_evaluator
from eval_tools import error_analysis, evidence_oracle, scoring, split
from tests.conftest import make_context
from utils import gold_evidence, load_sample_questions


def _row(qid, tid, qtype, label, start='', end=''):
    return {'question_id': qid, 'transcript_id': tid, 'question_type': qtype, 'label': str(label),
            'answer': 'yes' if label else 'no', 'question': 'q?', 'evidence_start': start, 'evidence_end': end}


def _synthetic():
    rng = random.Random(7)
    rows, preds, spans = [], [], []
    for i in range(120):
        label = i % 2
        qtype = 'positive' if label else ('hard_negative' if i % 4 else 'off_topic')
        start = rng.uniform(0, 100)
        rows.append(_row(f'q{i}', f'sample_{i // 10}', qtype, label, f'{start:.2f}' if label else '', f'{start + 3:.2f}' if label else ''))
        preds.append(rng.choice([0, 1, 1, -1]))
        spans.append(rng.choice([None, (start + rng.uniform(-2, 2), start + 3 + rng.uniform(-2, 2)), (start + 50, start + 55)]))
    return rows, preds, spans


def test_score_records_matches_reference_statistics():
    rows, preds, spans = _synthetic()
    stats = local_evaluator.Statistics()
    records = []
    for row, pred, span in zip(rows, preds, spans):
        span = scoring.as_span(span)
        stats.record(row['question_type'], int(row['label']), pred, gold_evidence(row), span)
        records.append(scoring.make_record(row, pred, span))
    result = scoring.score_records(records)
    assert result['accuracy'] == pytest.approx(stats.accuracy, abs=1e-12)
    assert result['mean_tiou'] == pytest.approx(stats.mean_tiou, abs=1e-12)
    assert result['score'] == pytest.approx(stats.final_score, abs=1e-12)
    assert result['n'] == stats.total and result['p'] == len(stats.tious)
    assert result['missing_spans'] == stats.missing_spans
    assert result['unanswered'] == stats.errors
    for qtype, (correct, total) in result['by_type'].items():
        assert stats.by_type[qtype] == [correct, total]


def test_oracle_records_score_one():
    records = [scoring.make_record(row, int(row['label']), gold_evidence(row)) for row in load_sample_questions()]
    result = scoring.score_records(records)
    assert result['accuracy'] == 1.0 and result['mean_tiou'] == 1.0 and result['score'] == 1.0
    assert result['n'] == 390 and result['p'] == 195


def test_full_denominator_counts_unsent_questions_wrong():
    rows, preds, spans = _synthetic()
    sent = [scoring.make_record(row, 1, None) for row in rows[:60]]
    completed = scoring.full_denominator(sent, rows)
    assert len(completed) == len(rows)
    assert all(r['prediction'] == local_evaluator.UNANSWERED for r in completed[60:])
    assert all(r.get('unsent') for r in completed[60:])
    result = scoring.score_records(completed)
    assert result['n'] == 120 and result['unanswered'] == 60
    partial = scoring.score_records(sent)
    assert result['accuracy'] <= partial['accuracy']


def _fake_rows(n=20):
    rows_by_tid = {}
    for i in range(n):
        tid = f'sample_{i}'
        rows_by_tid[tid] = [_row(f'{tid}_q{j}', tid, 'positive', 1, '1', '2') | {'question': f'question {i} {j}'} for j in range(10)]
    return rows_by_tid


def test_make_split_is_deterministic_and_respects_force_dev():
    rows = _fake_rows()
    hashes = {tid: f'hash{i}' for i, tid in enumerate(rows)}
    a = split.make_split(rows, hashes, seed=2026, n_holdout=5, force_dev=('sample_3',), created='t')
    b = split.make_split(rows, hashes, seed=2026, n_holdout=5, force_dev=('sample_3',), created='t')
    assert a['dev'] == b['dev'] and a['holdout'] == b['holdout']
    assert len(a['holdout']) == 5 and len(a['dev']) == 15
    assert 'sample_3' in a['dev'] and not set(a['dev']) & set(a['holdout'])
    c = split.make_split(rows, hashes, seed=1, n_holdout=5, force_dev=('sample_3',), created='t')
    assert c['holdout'] != a['holdout']


def test_groups_keep_duplicates_together():
    rows = _fake_rows(12)
    rows['sample_1'] = [dict(r, question=q['question']) for r, q in zip(rows['sample_1'], rows['sample_0'])]  # same questions
    hashes = {tid: f'hash{i}' for i, tid in enumerate(rows)}
    hashes['sample_5'] = hashes['sample_4']  # identical audio
    groups = split.build_groups(rows, hashes)
    as_sets = [set(g) for g in groups]
    assert {'sample_0', 'sample_1'} in as_sets
    assert {'sample_4', 'sample_5'} in as_sets
    result = split.make_split(rows, hashes, seed=3, n_holdout=4, force_dev=(), created='t')
    for group in ({'sample_0', 'sample_1'}, {'sample_4', 'sample_5'}):
        sides = {('holdout' if t in result['holdout'] else 'dev') for t in group}
        assert len(sides) == 1


def test_evidence_oracle_helpers_on_synthetic_context():
    context = make_context(['one two three', 'four five six', 'seven eight nine', 'ten eleven twelve'])
    unit = context.units[1]
    gold = (unit.start_s, unit.end_s)
    assert evidence_oracle.best_unit_tiou(context, gold).tiou == pytest.approx(1.0)
    gold2 = (context.units[1].start_s, context.units[2].end_s)
    assert evidence_oracle.best_unit_tiou(context, gold2).tiou < 1.0
    assert evidence_oracle.best_run_tiou(context, gold2, 2).tiou == pytest.approx(1.0)
    words = context.words()
    gold3 = (words[4].start_s, words[7].end_s)
    assert evidence_oracle.best_word_window_tiou(context, gold3, [1, 2]).tiou == pytest.approx(1.0)
    empty = make_context([])
    assert evidence_oracle.best_unit_tiou(empty, gold).span is None


def test_error_analysis_buckets():
    labels = error_analysis.bucket_labels()
    assert labels[0].startswith('[0.0') and labels[-1].endswith('1.0]')
    assert error_analysis.bucket_index(1.0) == len(labels) - 1
    assert error_analysis.bucket_index(0.0) == 0
    assert error_analysis.bucket_index(float('nan')) is None and error_analysis.bucket_index(None) is None
    rows, preds, spans = _synthetic()
    records = [scoring.make_record(row, pred, scoring.as_span(span)) for row, pred, span in zip(rows, preds, spans)]
    hist = error_analysis.tiou_histogram(records)
    assert sum(count for name, count in hist if name != 'invalid') == sum(1 for r in rows if r['label'] == '1')
