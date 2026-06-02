---
name: catchup
description: Orient yourself at session start — check issues, PRs, branch state, and recent progress. Use when starting a new session, resuming after a break, or when you need to understand what was accomplished last. Prevents redoing completed work.
---

# Catchup: Session Start

Verify project state before starting work. Each agent session is ephemeral — previous work may sit on an unmerged branch, an open PR, or a stale issue. Without checking, you risk redoing completed work.

## Protocol

### 1. Check Git State

```bash
# Current branch and tracking
git status -sb

# Feature branches with potential work
git branch -a

# Local commits not on remote
git log origin/HEAD..HEAD --oneline 2>/dev/null
```

### 2. Check Open PRs

```bash
gh pr list --repo stfc/goldilocks-core --state open --limit 10
```

For each open PR, note: which issue it closes, whether CI passes, whether it's been reviewed.

### 3. Read the Project Board

Use the `github-projects` skill for details. Projects are the board/state layer.

```bash
gh project list --owner stfc --format json
gh project view <project-number> --owner stfc --format json
```

If Project access fails with missing scopes, report that explicitly and continue with issues/PRs. Do not pretend the board was checked.

What's in **In Progress**? What's in **In Review**? Cross-reference with PRs — if something is In Progress but has no branch, find out why.

### 4. Read Recent Issue Activity

```bash
gh issue list --repo stfc/goldilocks-core --state open --limit 10
gh issue list --repo stfc/goldilocks-core --state all --limit 5 --search "sort:updated-desc"
```

Read the most recently updated issues. Check their comments for progress reports from previous sessions.

Focus on:
- **Open questions** — unresolved decisions that block work
- **Next steps** — what was the intended next action?
- **Blockers** — is anything waiting on review, external input, or another PR?

### 5. Cross-Reference

Compare what you found:
- Branch exists locally, not pushed → risk of lost work
- Branch pushed, no PR → risk another session won't find it
- PR open, not merged → starting new work may cause conflicts
- Issue says "in progress" but no branch → stale status
- PR merged but issue still open → close it
- Issue/PR status differs from GitHub Project status → update the Project or report missing access

### 6. Present Summary

```markdown
## Project State

**Branch**: [current]
**Open PRs**: [list or none]
**Board**: [what's in flight]

### Pending Work
- Issue #N: [status, what's left]
- Issue #M: [status, what's left]

### Discrepancies
- [Any mismatch between issues/board and actual git state]

### Recommended Next Step
[What to do next, based on the above]
```

## After Catchup

1. Fix discrepancies first — push stranded branches, close stale issues, update board status
2. Confirm the next step with the user
3. Proceed with work

If catchup reveals completed work that wasn't integrated:
- **Do NOT redo the work.** Push, PR, or merge the existing work first.