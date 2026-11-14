"""Aggregate agent traces with human labels; never infer grounding from citation IDs."""
import argparse
import json
import sys
from pathlib import Path


def aggregate(cases, traces, reviews):
    """Cases: id/answerable and optional primary_group; traces include query_id/status.

    Reviews: query_id, answer_grade (0..2 for answerable), citation_support
    (one boolean per citation), and abstention_correct (boolean for unanswerable).
    Provider errors are reported separately and remain failures in benchmark_pass.
    """
    if len({c['id'] for c in cases}) != len(cases):
        raise ValueError('case ID trùng')
    if len({r['query_id'] for r in traces}) != len(traces) or len({r['query_id'] for r in reviews}) != len(reviews):
        raise ValueError('trace/review ID trùng')
    trace_by_id = {r['query_id']: r for r in traces}
    review_by_id = {r['query_id']: r for r in reviews}
    out = {'cases': len(cases), 'provider_errors': 0, 'tool_use': {'n': 0, 'N': 0},
           'grounded_answer': {'0': 0, '1': 0, '2': 0, 'N': 0, 'mean': None},
           'citation_validity': {'n': 0, 'N': 0}, 'answer_citation_coverage': {'n': 0, 'N': 0},
           'abstention': {'n': 0, 'N': 0}, 'false_abstention': {'n': 0, 'N': 0},
           'benchmark_pass': {'n': 0, 'N': len(cases), 'rate': None},
           'by_group': {}, 'elapsed_seconds': {'sum': 0.0, 'N': 0},
           'api_tokens': {'prompt': 0, 'completion': 0, 'total': 0},
           'estimated_cost_usd': None}
    for case in cases:
        cid = case['id']
        if not isinstance(case.get('answerable'), bool): raise ValueError(f'{cid}: thiếu answerable boolean')
        group = case.get('primary_group', 'unassigned')
        group_counts = out['by_group'].setdefault(group, {'n': 0, 'N': 0, 'rate': None})
        group_counts['N'] += 1
        t = trace_by_id.get(cid)
        if t is None:
            out['provider_errors'] += 1
            continue
        elapsed = t.get('elapsed_seconds')
        if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool):
            out['elapsed_seconds']['sum'] += elapsed
            out['elapsed_seconds']['N'] += 1
        api_usage = t.get('api_usage') or {}
        prompt_tokens = api_usage.get('prompt_tokens', 0)
        completion_tokens = api_usage.get('completion_tokens', 0)
        if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
            out['api_tokens']['prompt'] += prompt_tokens
            out['api_tokens']['completion'] += completion_tokens
        if t.get('status') == 'error':
            out['provider_errors'] += 1
            continue
        if t.get('status') not in ('answered', 'abstain'): raise ValueError(f'{cid}: status không hợp lệ')
        if cid not in review_by_id: raise ValueError(f'{cid}: thiếu nhãn audit')
        r = review_by_id[cid]
        events = t.get('events')
        citations = t.get('citations')
        support = r.get('citation_support')
        if not isinstance(events, list) or not isinstance(citations, list): raise ValueError(f'{cid}: events/citations không hợp lệ')
        if not isinstance(support, list) or len(support) != len(citations) or any(type(v) is not bool for v in support):
            raise ValueError(f'{cid}: cần một nhãn boolean cho mỗi citation')
        out['tool_use']['N'] += 1
        out['tool_use']['n'] += any(e.get('event') == 'search' for e in events)
        out['citation_validity']['n'] += sum(support)
        out['citation_validity']['N'] += len(citations)
        if t['status'] == 'answered':
            out['answer_citation_coverage']['N'] += 1
            out['answer_citation_coverage']['n'] += bool(citations)
        if case['answerable']:
            grade = r.get('answer_grade')
            if type(grade) is not int or grade not in (0, 1, 2): raise ValueError(f'{cid}: answer_grade phải là 0, 1 hoặc 2')
            if t['status'] == 'abstain' and grade != 0: raise ValueError(f'{cid}: abstain trên câu có đáp án phải được chấm 0')
            out['grounded_answer'][str(grade)] += 1
            out['grounded_answer']['N'] += 1
            out['false_abstention']['N'] += 1
            out['false_abstention']['n'] += t['status'] == 'abstain'
        else:
            correct = r.get('abstention_correct')
            if type(correct) is not bool:
                raise ValueError(f'{cid}: cần nhãn abstention_correct do người chấm xác nhận')
            if correct and t['status'] != 'abstain':
                raise ValueError(f'{cid}: nhãn từ chối đúng không khớp status')
            out['abstention']['N'] += 1
            out['abstention']['n'] += correct
            case_pass = t['status'] == 'abstain' and correct
        if case['answerable']:
            case_pass = (
                t['status'] == 'answered'
                and grade == 2
                and bool(citations)
                and all(support)
            )
        out['benchmark_pass']['n'] += case_pass
        group_counts['n'] += case_pass
    g = out['grounded_answer']
    if g['N']: g['mean'] = (g['1'] + 2*g['2']) / g['N']
    if out['benchmark_pass']['N']:
        out['benchmark_pass']['rate'] = (
            out['benchmark_pass']['n'] / out['benchmark_pass']['N']
        )
    for group_counts in out['by_group'].values():
        if group_counts['N']:
            group_counts['rate'] = group_counts['n'] / group_counts['N']
    out['api_tokens']['total'] = (
        out['api_tokens']['prompt'] + out['api_tokens']['completion']
    )
    return out


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='Tổng hợp trace agent và nhãn audit thủ công')
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--traces', type=Path, required=True)
    parser.add_argument('--reviews', type=Path, required=True)
    parser.add_argument('--split', choices=('dev', 'test'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    def load(path): return json.loads(path.read_text(encoding='utf-8'))
    cases = [c for c in load(args.cases) if c.get('split') == args.split]
    if not cases: raise ValueError('không có case trong split')
    trace_data = load(args.traces)
    traces = trace_data.get('traces') if isinstance(trace_data, dict) else trace_data
    if not isinstance(traces, list):
        raise ValueError('traces phải là JSON array hoặc report có trường traces')
    report = {'split': args.split, **aggregate(cases, traces, load(args.reviews))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
