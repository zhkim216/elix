---
name: diagnose
description: Use only when the user explicitly invokes diagnose for a failure or regression.
---

# Diagnose

Start from the failing command, log, or artifact. Inspect the active call path,
separate live causal alternatives, and reproduce or instrument only when needed
to distinguish them.

When an earlier check passed, diagnose both the system defect and the evidence
defect. Compare the validation and production paths and locate their earliest
material divergence across code, data, control flow, environment, or state.
State the exact claim each check supports; do not combine proxy checks into
end-to-end evidence for a path they did not execute.

Explain the cause before proposing a fix. Implement a fix only when requested,
then run one targeted regression through a production-equivalent non-mutating
path and the consequential transformations that failed. A diagnose, fix, or
test request does not authorize submitting jobs, deploying, sending messages,
deleting data, or otherwise mutating external state. If the outer production
entrypoint has such effects, use its dry-run or inner worker path; invoke the
real entrypoint only when the user separately authorized that exact action. If
no safe equivalent exists, identify the unexecuted difference and report the
residual risk.
