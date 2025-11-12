#!/usr/bin/env python3
"""
Check for outdated libraries and dependency conflicts in PyTorch/HF/vLLM/SGLang code.
Focused on catching wrong framework choices and version issues.
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DependencyIssue:
    package: str
    current_pattern: str
    issue_type: str  # 'wrong_framework', 'outdated', 'deprecated', 'missing_install'
    message: str
    recommendation: str


class DependencyChecker:
    """Check for dependency issues in training/inference code."""
    
    # Focus on PyTorch, HuggingFace, vLLM, SGLang only
    RECOMMENDED_VERSIONS = {
        'torch': '2.5.0',
        'transformers': '4.47.0', 
        'vllm': '0.6.0',
        'sglang': '0.3.0',
        'peft': '0.14.0',
        'accelerate': '1.2.0',
    }
    
    # Framework choice checks
    FRAMEWORK_CHECKS = {
        'inference_should_use_vllm': {
            'pattern': r'pipeline\(["\']text-generation["\']',
            'condition': 'No vLLM import found',
            'message': 'Using transformers pipeline for inference',
            'recommendation': 'Consider vLLM for much faster inference (5-10x speedup)'
        },
        'structured_output_should_use_sglang': {
            'pattern': r'(json\.loads|regex|parse)',
            'condition': 'No SGLang import found',
            'message': 'Using manual parsing for structured output',
            'recommendation': 'Consider SGLang for reliable structured generation'
        }
    }
    
    # Deprecated patterns specific to our frameworks
    DEPRECATED_PATTERNS = {
        'AutoTokenizer.from_pretrained(..., use_fast=False)': {
            'replacement': 'Remove use_fast=False (fast tokenizers are default)',
            'reason': 'Slow tokenizers deprecated'
        },
    }
    
    # Critical packages to check installation
    INSTALL_CHECKS = {
        'torch': 'pip install torch',
        'transformers': 'pip install transformers',
        'vllm': 'pip install vllm',
        'sglang': 'pip install sglang',
        'peft': 'pip install peft',
        'accelerate': 'pip install accelerate',
    }
    
    def __init__(self, code: str):
        self.code = code
        self.issues: List[DependencyIssue] = []
        
    def check(self) -> List[DependencyIssue]:
        """Run all dependency checks."""
        self._check_imports()
        self._check_framework_choices()
        self._check_deprecated_patterns()
        self._check_missing_requirements()
        return self.issues
    
    def _check_framework_choices(self):
        """Check if using appropriate framework for the task."""
        # Check for inference with transformers instead of vLLM
        if 'pipeline' in self.code and 'text-generation' in self.code:
            if 'vllm' not in self.code.lower():
                self.issues.append(DependencyIssue(
                    package='framework_choice',
                    current_pattern='transformers pipeline',
                    issue_type='wrong_framework',
                    message='Using transformers pipeline for inference (slower)',
                    recommendation='Consider vLLM for 5-10x faster inference: from vllm import LLM; llm = LLM(model=...)'
                ))
        
        # Check for manual parsing when SGLang would help
        if any(pattern in self.code for pattern in ['json.loads', 're.findall', 're.search']) and 'generate' in self.code:
            if 'sglang' not in self.code.lower():
                self.issues.append(DependencyIssue(
                    package='framework_choice',
                    current_pattern='manual parsing',
                    issue_type='wrong_framework',
                    message='Using manual parsing for structured output',
                    recommendation='Consider SGLang for reliable structured generation with constrained decoding'
                ))
    
    def _check_imports(self):
        """Check imported packages."""
        import_pattern = r'^\s*(?:import|from)\s+(\w+)'
        
        for match in re.finditer(import_pattern, self.code, re.MULTILINE):
            package = match.group(1)
            if package in self.INSTALL_CHECKS:
                # Check if pip install is mentioned
                if f'pip install {package}' not in self.code and f'!pip install {package}' not in self.code:
                    self.issues.append(DependencyIssue(
                        package=package,
                        current_pattern=match.group(0),
                        issue_type='missing_install',
                        message=f'{package} imported but no installation cell found',
                        recommendation=f'Add a cell with: !{self.INSTALL_CHECKS[package]}'
                    ))
    
    def _check_deprecated_patterns(self):
        """Check for deprecated code patterns."""
        for pattern, info in self.DEPRECATED_PATTERNS.items():
            if pattern in self.code:
                self.issues.append(DependencyIssue(
                    package='code_pattern',
                    current_pattern=pattern,
                    issue_type='deprecated',
                    message=f'Deprecated pattern found: {pattern}',
                    recommendation=f"{info['replacement']} (Reason: {info['reason']})"
                ))
    
    def _check_missing_requirements(self):
        """Check for missing common requirements."""
        # Check for HuggingFace hub usage
        if 'save_pretrained' in self.code or 'push_to_hub' in self.code:
            if 'huggingface-cli login' not in self.code and 'HF_TOKEN' not in self.code:
                self.issues.append(DependencyIssue(
                    package='huggingface_hub',
                    current_pattern='save_pretrained/push_to_hub',
                    issue_type='missing_auth',
                    message='Using HuggingFace hub features without authentication check',
                    recommendation='Add: from huggingface_hub import login; login() or set HF_TOKEN env var'
                ))
    
    def _extract_version(self, package: str) -> Optional[str]:
        """Extract version requirement from code."""
        pattern = rf'pip install {package}==([0-9.]+)'
        match = re.search(pattern, self.code)
        return match.group(1) if match else None


def format_dependency_report(issues: List[DependencyIssue]) -> str:
    """Format dependency issues into readable report."""
    if not issues:
        return "✅ No dependency issues found!"
    
    report = []
    report.append(f"\n{'='*70}")
    report.append(f"Found {len(issues)} dependency issue(s):")
    report.append(f"{'='*70}\n")
    
    by_type = {}
    for issue in issues:
        by_type.setdefault(issue.issue_type, []).append(issue)
    
    # Show framework choice issues first (most critical)
    priority_order = ['wrong_framework', 'missing_install', 'deprecated', 'missing_auth']
    
    for issue_type in priority_order:
        if issue_type not in by_type:
            continue
            
        items = by_type[issue_type]
        icon = {
            'wrong_framework': '🔴',
            'missing_install': '📦',
            'deprecated': '🔶',
            'missing_auth': '🔑',
        }.get(issue_type, '❓')
        
        report.append(f"{icon} {issue_type.upper().replace('_', ' ')}")
        report.append("-" * 70)
        
        for issue in items:
            report.append(f"  Package: {issue.package}")
            report.append(f"  {issue.message}")
            report.append(f"  💡 {issue.recommendation}")
            report.append("")
    
    return "\n".join(report)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python check_dependencies.py <file.py>")
        sys.exit(1)
    
    with open(sys.argv[1], 'r') as f:
        code = f.read()
    
    checker = DependencyChecker(code)
    issues = checker.check()
    print(format_dependency_report(issues))
