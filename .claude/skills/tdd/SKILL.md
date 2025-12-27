---
name: tdd
description: Test-driven development workflow. Use when user wants to implement a feature using TDD.
allowed-tools: Task, Bash
---

# TDD Workflow

User provides:
- Feature description
- Specific testing instructions for this task

## Process

1. **Red**: Spawn test-writer agent with feature description AND testing instructions
2. Run `pytest` on the new test to confirm it fails
3. **Green**: Spawn implementer agent with the failing test location
4. Run `pytest` to confirm the test passes

Report the outcome of each step.
