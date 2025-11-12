#!/usr/bin/env python3
"""
Static analyzer for ML training code that catches common errors without execution.
Focuses on PyTorch/HuggingFace training patterns.
"""

import ast
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class Severity(Enum):
    CRITICAL = "CRITICAL"  # Will cause immediate failure
    WARNING = "WARNING"    # May cause issues or unexpected behavior
    SUGGESTION = "SUGGESTION"  # Best practice violations


@dataclass
class Issue:
    severity: Severity
    line: int
    category: str
    message: str
    suggestion: str = ""


class TrainingCodeAnalyzer(ast.NodeVisitor):
    """AST-based analyzer for training code patterns."""
    
    def __init__(self, code: str, filename: str = "<notebook>"):
        self.code = code
        self.filename = filename
        self.lines = code.split('\n')
        self.issues: List[Issue] = []
        
        # Track imports
        self.imports = set()
        self.from_imports = {}
        
        # Track variable assignments
        self.variables = {}
        self.model_vars = set()
        self.optimizer_vars = set()
        
        # Track function definitions
        self.functions = set()
        
        # Track loop contexts
        self.in_training_loop = False
        self.loop_depth = 0
        
    def analyze(self) -> List[Issue]:
        """Run full analysis."""
        try:
            tree = ast.parse(self.code, self.filename)
            self.visit(tree)
        except SyntaxError as e:
            self.issues.append(Issue(
                severity=Severity.CRITICAL,
                line=e.lineno or 0,
                category="Syntax Error",
                message=f"Syntax error: {e.msg}",
                suggestion="Fix the syntax error before proceeding"
            ))
            return self.issues
        
        # Post-processing checks
        self._check_missing_imports()
        self._check_training_patterns()
        self._check_logging_setup()
        self._check_model_saving()
        
        return sorted(self.issues, key=lambda x: (x.severity.value, x.line))
    
    def visit_Import(self, node):
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        if node.module:
            for alias in node.names:
                self.from_imports[alias.name] = node.module
        self.generic_visit(node)
    
    def visit_Assign(self, node):
        # Track variable assignments
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.variables[target.id] = node.value
                
                # Track model and optimizer variables
                if self._is_model_creation(node.value):
                    self.model_vars.add(target.id)
                if self._is_optimizer_creation(node.value):
                    self.optimizer_vars.add(target.id)
        
        self.generic_visit(node)
    
    def visit_For(self, node):
        """Check training loops."""
        self.loop_depth += 1
        
        # Check if this looks like a training loop
        if self._is_training_loop(node):
            self.in_training_loop = True
            self._check_training_loop_body(node)
        
        self.generic_visit(node)
        self.loop_depth -= 1
        self.in_training_loop = False
    
    def visit_Call(self, node):
        """Check function calls for common issues."""
        func_name = self._get_func_name(node)
        
        # Check for common tensor operation mistakes
        if func_name == 'backward':
            self._check_backward_call(node)
        elif func_name == 'zero_grad':
            pass  # Will be checked in loop context
        elif func_name == 'step':
            pass  # Will be checked in loop context
        elif func_name in ['save', 'save_pretrained']:
            self._check_save_call(node)
        
        self.generic_visit(node)
    
    def _is_model_creation(self, node) -> bool:
        """Check if node represents model instantiation."""
        if isinstance(node, ast.Call):
            func_name = self._get_func_name(node)
            return any(pattern in func_name.lower() for pattern in 
                      ['model', 'network', 'automodel', 'gpt', 'bert', 'llama'])
        return False
    
    def _is_optimizer_creation(self, node) -> bool:
        """Check if node represents optimizer instantiation."""
        if isinstance(node, ast.Call):
            func_name = self._get_func_name(node)
            return any(pattern in func_name for pattern in 
                      ['Adam', 'SGD', 'AdamW', 'Optimizer'])
        return False
    
    def _is_training_loop(self, node) -> bool:
        """Heuristic to detect training loops."""
        # Check variable names
        if isinstance(node.target, ast.Name):
            target_name = node.target.id.lower()
            if any(kw in target_name for kw in ['epoch', 'step', 'batch', 'iteration']):
                return True
        
        # Check if iterating over dataloader
        if isinstance(node.iter, ast.Name):
            iter_name = node.iter.id.lower()
            if 'loader' in iter_name or 'dataset' in iter_name:
                return True
        
        return False
    
    def _check_training_loop_body(self, node):
        """Analyze training loop for common mistakes."""
        body_code = ast.unparse(node)
        
        has_backward = 'backward()' in body_code
        has_zero_grad = 'zero_grad()' in body_code
        has_step = '.step()' in body_code
        
        if has_backward and not has_zero_grad:
            self.issues.append(Issue(
                severity=Severity.CRITICAL,
                line=node.lineno,
                category="Training Loop",
                message="backward() called without zero_grad() in training loop",
                suggestion="Add optimizer.zero_grad() before loss.backward()"
            ))
        
        if has_backward and not has_step:
            self.issues.append(Issue(
                severity=Severity.CRITICAL,
                line=node.lineno,
                category="Training Loop",
                message="backward() called without optimizer.step()",
                suggestion="Add optimizer.step() after loss.backward()"
            ))
        
        # Check order: should be zero_grad -> forward -> backward -> step
        if has_zero_grad and has_backward and has_step:
            # Simplified order check
            zg_pos = body_code.find('zero_grad()')
            bw_pos = body_code.find('backward()')
            st_pos = body_code.find('.step()')
            
            if not (zg_pos < bw_pos < st_pos):
                self.issues.append(Issue(
                    severity=Severity.WARNING,
                    line=node.lineno,
                    category="Training Loop",
                    message="Unusual order of training operations",
                    suggestion="Standard order: zero_grad() -> forward pass -> backward() -> step()"
                ))
    
    def _check_backward_call(self, node):
        """Check backward() usage."""
        if node.args:
            self.issues.append(Issue(
                severity=Severity.WARNING,
                line=node.lineno,
                category="Gradient Computation",
                message="backward() called with arguments - ensure this is intentional",
                suggestion="Usually backward() is called without arguments"
            ))
    
    def _check_save_call(self, node):
        """Check model saving patterns."""
        func_name = self._get_func_name(node)
        # Add checks for proper saving patterns if needed
        pass
    
    def _check_missing_imports(self):
        """Check for commonly missing imports."""
        required_imports = {
            'torch': ['model', 'tensor', 'optim', 'nn', 'cuda'],
            'transformers': ['automodel', 'autotokenizer', 'trainer'],
            'wandb': ['wandb.init', 'wandb.log'],
            'tqdm': ['progress', 'tqdm'],
        }
        
        code_lower = self.code.lower()
        
        for package, keywords in required_imports.items():
            if any(kw in code_lower for kw in keywords):
                if package not in self.imports and package not in self.from_imports.values():
                    self.issues.append(Issue(
                        severity=Severity.CRITICAL,
                        line=1,
                        category="Missing Import",
                        message=f"Using {package} functionality without importing {package}",
                        suggestion=f"Add: import {package}"
                    ))
    
    def _check_training_patterns(self):
        """Check for proper training patterns."""
        code_lower = self.code.lower()
        
        # Check for model.train() / model.eval()
        if 'forward' in code_lower or 'model(' in code_lower:
            if 'model.train()' not in code_lower and '.train()' not in code_lower:
                self.issues.append(Issue(
                    severity=Severity.WARNING,
                    line=0,
                    category="Training Mode",
                    message="No model.train() call found",
                    suggestion="Add model.train() before training loop and model.eval() before evaluation"
                ))
    
    def _check_logging_setup(self):
        """Check for logging/monitoring setup."""
        has_wandb = 'wandb' in self.imports or 'wandb' in self.from_imports.values()
        has_tensorboard = 'tensorboard' in self.code.lower()
        
        if not has_wandb and not has_tensorboard:
            # Check if there's actual training happening
            if any(var in self.optimizer_vars for var in self.variables):
                self.issues.append(Issue(
                    severity=Severity.SUGGESTION,
                    line=0,
                    category="Logging",
                    message="No logging framework detected (wandb, tensorboard)",
                    suggestion="Consider adding wandb.init() for experiment tracking"
                ))
        
        # Check wandb initialization
        if has_wandb and 'wandb.init' not in self.code:
            self.issues.append(Issue(
                severity=Severity.WARNING,
                line=0,
                category="Logging",
                message="wandb imported but wandb.init() not called",
                suggestion="Add wandb.init(project='...', name='...') before training"
            ))
    
    def _check_model_saving(self):
        """Check for model checkpoint saving."""
        has_save = any(pattern in self.code for pattern in 
                      ['torch.save', 'save_pretrained', 'save_checkpoint'])
        
        if self.model_vars and not has_save:
            self.issues.append(Issue(
                severity=Severity.SUGGESTION,
                line=0,
                category="Model Saving",
                message="No model saving code detected",
                suggestion="Add model.save_pretrained() or torch.save() to save checkpoints"
            ))
    
    def _get_func_name(self, node) -> str:
        """Extract function name from call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""


def analyze_notebook_cell(code: str, cell_index: int = 0) -> List[Issue]:
    """Analyze a single notebook cell."""
    analyzer = TrainingCodeAnalyzer(code, f"<cell {cell_index}>")
    return analyzer.analyze()


def format_report(issues: List[Issue]) -> str:
    """Format issues into readable report."""
    if not issues:
        return "✅ No issues found!"
    
    report = []
    report.append(f"\n{'='*70}")
    report.append(f"Found {len(issues)} issue(s):")
    report.append(f"{'='*70}\n")
    
    for issue in issues:
        icon = {"CRITICAL": "🔴", "WARNING": "🟡", "SUGGESTION": "💡"}[issue.severity.name]
        report.append(f"{icon} [{issue.severity.name}] {issue.category}")
        if issue.line > 0:
            report.append(f"   Line {issue.line}")
        report.append(f"   {issue.message}")
        if issue.suggestion:
            report.append(f"   💡 {issue.suggestion}")
        report.append("")
    
    return "\n".join(report)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_training_code.py <file.py>")
        sys.exit(1)
    
    with open(sys.argv[1], 'r') as f:
        code = f.read()
    
    analyzer = TrainingCodeAnalyzer(code, sys.argv[1])
    issues = analyzer.analyze()
    print(format_report(issues))
