# Principle 5: Format, style, and consistency

Avoid patterns that undermine readability or signal poor authorship. These standards keep documentation clean and professional.

## Em-dashes

Hyphens and parentheses are clearer than em-dashes.

**DON'T:**

- Use em-dashes between clauses

**DO:**

- "Configuration (provided in the setup file) is required"
- "Configuration - provided in the setup file - is required"

## Directional Arrows and Symbols

Use prose and lists instead of arrows or complex symbols.

**DON'T:**

- Sequences written with arrow symbols in text

**DO:**

- Use numbered lists for sequences

## ASCII Art and Box Diagrams

Plain prose and numbered lists are clearer and maintainable.

**DON'T:**

- Box drawings or ASCII flow diagrams

**DO:**

- Prose descriptions and numbered lists

## Emojis in Code Blocks

Code blocks must be copy-paste safe and readable in terminals.

**DON'T:**

```text
(DONE) kubectl get pods
(NOT DONE) kubectl delete pods
```

**DO:**

```shell
kubectl get pods
kubectl delete pods
```

## Unicode Escape Sequences in Prose

Escape sequences appear as literal text in many systems and are not accessible.

**DON'T:**

- Use "\u274c" or "\u2705" in documentation text

**DO:**

- Use plain text labels: "(DO)", "(DON'T)", "(GOOD)", "(AVOID)"

## Code Block Language Specifiers

All code blocks require language tags for syntax highlighting and clarity.

**DON'T:**

```text
kubectl apply -f deployment.yaml
```

**DO:**

```shell
kubectl apply -f deployment.yaml
```
