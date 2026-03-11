# Principle 3: Docs scale by folders, not file length

When a file exceeds 3 to 4 logical sections, promote it to a folder. This allows independent evolution without merge conflicts.

## When to Promote a File to a Folder

- File has more than 3 to 4 distinct logical sections
- File requires a table of contents to navigate
- Multiple parts could evolve at different rates

## Example Transformation

`gitops.md` becomes too long. Create folder `gitops/` with focused documents:

- `overview.md` - High-level introduction
- `reconciliation.md` - Reconciliation process
- `secrets.md` - Secret management
- `index.md` - Navigation hub
