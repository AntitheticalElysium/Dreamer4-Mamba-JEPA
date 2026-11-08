"""P1-F4 trainer contract: F2's AST test (../20261103_p1f2/test_contract.py) against this folder's trainer."""
import ast
import copy
import importlib
import inspect
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from saccade import model as M, train as TR
C=importlib.import_module('artifacts.experiments.20261103_p1f2.chooser')
T=importlib.import_module('artifacts.experiments.20261106_p1f4.trainer')


def test_world_loop_ast_matches_canonical_except_declared_hooks():
    a=ast.parse(inspect.getsource(TR.p1)).body[0]
    b=ast.parse(inspect.getsource(T.p1)).body[0];b.args.args.pop();b.args.defaults.pop()
    class Strip(ast.NodeTransformer):
        def visit_Attribute(self,n):
            if isinstance(n.value,ast.Name) and n.value.id=='TR':return ast.copy_location(ast.Name(n.attr,ast.Load()),n)
            return self.generic_visit(n)
        def visit_Assign(self,n):
            if any(isinstance(t,ast.Name) and t.id=='controller' for t in n.targets):return None
            if any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='ck' and isinstance(t.slice,ast.Constant) and t.slice.value=='fixation_controller' for t in n.targets):return None
            return self.generic_visit(n)
        def visit_Expr(self,n):
            if isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='controller':return None
            return self.generic_visit(n)
    assert ast.dump(a,include_attributes=False)==ast.dump(Strip().visit(b),include_attributes=False)
