"""Prevent changing score extraction or measured work alongside prompt reuse."""
import ast
from pathlib import Path
import unittest


class PromptCacheBoundaryTests(unittest.TestCase):
    def test_scoring_extraction_and_timed_loop_match_precision_control(self):
        root=Path(__file__).resolve().parents[1]/'rapidtinyrec'
        control=ast.parse((root/'benchmark_precision.py').read_text())
        candidate=ast.parse((root/'benchmark_prompt_cache.py').read_text())
        def unpack(tree):
            return next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='unpack')
        def timed_loop(tree):
            return next(n for n in ast.walk(tree) if isinstance(n,ast.For)
                and isinstance(n.target,ast.Tuple)
                and [getattr(x,'id',None) for x in n.target.elts]==['repeat','user']
                and isinstance(n.iter,ast.Name) and n.iter.id=='trials')
        self.assertEqual(ast.dump(unpack(candidate)),ast.dump(unpack(control)))
        self.assertEqual(ast.dump(timed_loop(candidate)),ast.dump(timed_loop(control)))


if __name__=='__main__':unittest.main()
