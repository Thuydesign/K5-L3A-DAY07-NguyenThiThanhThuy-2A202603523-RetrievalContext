import json
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'starter'))
from agent_metrics import aggregate

class AgentMetricTests(unittest.TestCase):
    def test_counts_and_errors_are_separate(self):
        cases = [{'id':'a','answerable':True}, {'id':'b','answerable':False}, {'id':'c','answerable':True}]
        traces = [{'query_id':'a','status':'answered','events':[{'event':'search'}],'citations':['c1','c2']},
                  {'query_id':'b','status':'abstain','events':[{'event':'search'}],'citations':[]},
                  {'query_id':'c','status':'error','events':[],'citations':[]}]
        reviews = [{'query_id':'a','answer_grade':1,'citation_support':[True,False]},
                   {'query_id':'b','citation_support':[],'abstention_correct':True}]
        result = aggregate(cases, traces, reviews)
        self.assertEqual(result['provider_errors'], 1)
        self.assertEqual(result['tool_use'], {'n':2,'N':2})
        self.assertEqual(result['citation_validity'], {'n':1,'N':2})
        self.assertEqual(result['abstention'], {'n':1,'N':1})
        self.assertEqual(result['grounded_answer']['mean'], 1)

    def test_unanswerable_requires_human_confirmation(self):
        cases = [{'id':'a','answerable':False}]
        traces = [{'query_id':'a','status':'abstain','events':[],'citations':[]}]
        with self.assertRaises(ValueError): aggregate(cases, traces, [{'query_id':'a','citation_support':[]}])
        result = aggregate(cases, traces, [{'query_id':'a','citation_support':[], 'abstention_correct':False}])
        self.assertEqual(result['abstention'], {'n':0, 'N':1})

    def test_rejects_missing_review_and_false_abstention_grade(self):
        cases = [{'id':'a','answerable':True}]
        traces = [{'query_id':'a','status':'abstain','events':[],'citations':[]}]
        with self.assertRaises(ValueError): aggregate(cases,traces,[])
        with self.assertRaises(ValueError): aggregate(cases,traces,[{'query_id':'a','answer_grade':2,'citation_support':[]}])

    def test_cli_accepts_agent_report_with_nested_traces(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            cases = [{'id': 'q1', 'split': 'test', 'answerable': True}]
            report = {'split': 'test', 'traces': [{
                'query_id': 'q1', 'status': 'answered',
                'events': [{'event': 'search'}], 'citations': ['c1'],
            }]}
            reviews = [{
                'query_id': 'q1', 'answer_grade': 2, 'citation_support': [True],
            }]
            paths = {}
            for name, value in (('cases', cases), ('report', report), ('reviews', reviews)):
                paths[name] = temp / f'{name}.json'
                paths[name].write_text(json.dumps(value), encoding='utf-8')
            output = temp / 'summary.json'
            subprocess.run([
                sys.executable, str(root / 'starter' / 'agent_metrics.py'),
                '--cases', str(paths['cases']), '--traces', str(paths['report']),
                '--reviews', str(paths['reviews']), '--split', 'test',
                '--output', str(output),
            ], check=True, capture_output=True, text=True)
            summary = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(summary['grounded_answer']['mean'], 2)

    def test_strict_benchmark_pass_rate_includes_groups_and_system_errors(self):
        cases = [
            {'id': 'table', 'split': 'test', 'answerable': True, 'primary_group': 'table'},
            {'id': 'missing', 'split': 'test', 'answerable': False, 'primary_group': 'unanswerable'},
            {'id': 'chart', 'split': 'test', 'answerable': True, 'primary_group': 'chart'},
        ]
        traces = [
            {
                'query_id': 'table', 'status': 'answered',
                'events': [{'event': 'search'}], 'citations': ['p1-c1'],
                'elapsed_seconds': 2.5,
                'api_usage': {'prompt_tokens': 10, 'completion_tokens': 4},
            },
            {
                'query_id': 'missing', 'status': 'abstain',
                'events': [{'event': 'search'}], 'citations': [],
                'elapsed_seconds': 1.0,
                'api_usage': {'prompt_tokens': 5, 'completion_tokens': 2},
            },
            {
                'query_id': 'chart', 'status': 'error',
                'events': [], 'citations': [],
                'elapsed_seconds': 0.5,
                'api_usage': {'prompt_tokens': 3, 'completion_tokens': 0},
            },
        ]
        reviews = [
            {'query_id': 'table', 'answer_grade': 2, 'citation_support': [True]},
            {'query_id': 'missing', 'abstention_correct': True, 'citation_support': []},
        ]
        result = aggregate(cases, traces, reviews)
        self.assertEqual(result['benchmark_pass'], {'n': 2, 'N': 3, 'rate': 2 / 3})
        self.assertEqual(result['by_group']['table'], {'n': 1, 'N': 1, 'rate': 1.0})
        self.assertEqual(result['by_group']['chart'], {'n': 0, 'N': 1, 'rate': 0.0})
        self.assertEqual(result['by_group']['unanswerable']['n'], 1)
        self.assertEqual(result['provider_errors'], 1)
        self.assertEqual(result['api_tokens']['total'], 24)

    def test_missing_trace_counts_as_a_failed_benchmark_case(self):
        result = aggregate(
            [{'id': 'q', 'answerable': True, 'primary_group': 'text'}],
            [],
            [],
        )
        self.assertEqual(result['provider_errors'], 1)
        self.assertEqual(result['benchmark_pass'], {'n': 0, 'N': 1, 'rate': 0.0})
        self.assertEqual(result['by_group']['text']['N'], 1)

if __name__ == '__main__': unittest.main()
