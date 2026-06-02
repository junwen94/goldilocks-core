---
name: github-projects
description: Manage GitHub Projects as the project board for this repo. Use when starting work, creating issues, opening PRs, updating status, or checking what is in flight.
---

# GitHub Projects

GitHub Issues are the work items. Pull requests are review units. **GitHub Projects is the board/state layer.** Keep it current so future contributors and agents can see what is in flight.

Repo: `stfc/goldilocks-core`
Owner: `stfc`

## Authentication

Project commands require GitHub token scopes that normal issue/PR commands may not have.

Check access:

```bash
gh project list --owner stfc --format json
```

If you see a missing-scope error, ask the user to refresh auth:

```bash
gh auth refresh -s read:project,project
```

Do not work around Projects by inventing an external board. If project access is unavailable, say so and record intended status changes in issue comments.

## Board columns

Use these standard statuses:

- **Backlog** — planned but not started
- **In Progress** — active branch/work exists
- **In Review** — PR is open
- **Done** — PR merged or issue completed

If the project uses different names, follow the existing project rather than renaming everything.

## Catch up on board state

```bash
gh project list --owner stfc --format json
gh project view <project-number> --owner stfc --format json
```

Cross-check board state against git and PR state:

- In Progress but no branch/updates → stale or blocked
- In Review but no PR → stale
- Done but issue still open → close/update issue
- PR merged but project item not Done → update project status

## Add an issue to the project

After creating a planning issue:

```bash
gh project item-add <project-number> --owner stfc --url <issue-url>
```

Then set its status to **Backlog** if the project does not do that automatically.

Project v2 status updates require item, field, and option IDs. Fetch them first:

```bash
gh project view <project-number> --owner stfc --format json
```

If IDs are not obvious from `gh project`, use GraphQL through `gh api graphql` rather than guessing.

## Move work through statuses

Common transitions:

- Issue created/planned → **Backlog**
- Branch started or work begun → **In Progress**
- PR opened → **In Review**
- PR merged / issue closed → **Done**

Every transition should also be visible in the issue/PR timeline through normal GitHub events or an agent-written report comment.

## Status update fallback

If project access is unavailable, add a comment to the relevant issue:

```markdown
## Project status update

Intended project status: In Progress / In Review / Done

Could not update GitHub Projects because: [missing scope / no project access / other]

---
Written by an agent on behalf of <user>.
```

Replace `<user>` with the human who requested the work.

## Gotchas

- Project commands are more brittle than issue/PR commands. Prefer read-before-write.
- Never claim the board was updated unless the command succeeded.
- Do not create a new Project if one already exists. Inspect first.
- Do not rename statuses or fields without explicit maintainer approval.
- If writing a project-status comment as an agent, include `Written by an agent on behalf of <user>.`