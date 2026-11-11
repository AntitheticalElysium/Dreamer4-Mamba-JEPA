"""P1-HUD-Change trainer contract: AST test against saccade.train.p1."""
import ast
import importlib
import inspect
import pytest
from saccade import train as TR

T = importlib.import_module('artifacts.experiments.20261109_p1_hud_change.trainer')


def test_world_loop_ast_matches_canonical():
    a = ast.parse(inspect.getsource(TR.p1)).body[0]
    b = ast.parse(inspect.getsource(T.p1)).body[0]
    b.args.args.pop()
    b.args.defaults.pop()

    class Strip(ast.NodeTransformer):
        def visit_Attribute(self, n):
            if isinstance(n.value, ast.Name) and n.value.id == 'TR':
                return ast.copy_location(ast.Name(n.attr, ast.Load()), n)
            return self.generic_visit(n)

        def visit_Assign(self, n):
            if any(isinstance(t, ast.Name) and t.id == 'controller' for t in n.targets):
                return ast.copy_location(ast.Assign(targets=[ast.Name(id='controller', ctx=ast.Store())], value=ast.Constant(value=None)), n)
            return self.generic_visit(n)

    assert ast.dump(Strip().visit(a), include_attributes=False) == ast.dump(Strip().visit(b), include_attributes=False)
