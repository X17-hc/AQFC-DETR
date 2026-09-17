"""Check the actual evaluation call's interval and its progress logger."""
import ast
import re
from pathlib import Path

from util.misc import MetricLogger


def test_eval_progress_interval_and_final_summary(capsys):
    tree = ast.parse((Path(__import__('util').__file__).parent.parent / 'engine.py').read_text(encoding='utf-8'))
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = node.value.value
    evaluate = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'evaluate')
    call = next(node for node in ast.walk(evaluate) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and node.func.attr == 'log_every')
    interval = eval(compile(ast.Expression(call.args[1]), '<interval>', 'eval'), {'min': min}, constants)
    assert interval == 2000
    assert constants['print_freq'] == 1000  # Training interval must not change.
    assert list(MetricLogger().log_every(range(4002), interval, 'Test:')) == list(range(4002))
    output = capsys.readouterr().out
    assert [int(value) for value in re.findall(r'\[\s*(\d+)/4002\]', output)] == [0, 2000, 4000, 4001]
    assert 'Test: Total time:' in output
