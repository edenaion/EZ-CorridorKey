# ez-CorridorKey Brand Bible

> A fork of Corridor Crew's CorridorKey with a unified UI. This brand bible documents the visual identity derived from Corridor Digital's established brand language.

---

## 1. Brand Foundation

### Product Name
**ez-CorridorKey** — camelCase, hyphenated prefix. The `ez-` prefix signals our fork; `CorridorKey` preserves the original branding.

### Tagline
**"Perfect Green Screen Keys"** *(inherited from Corridor Digital)*

### Description
AI-powered green screen keying tool that reconstructs true foreground colors — hair, motion blur, transparency — using neural networks. Not a traditional keyer. A foreground reconstructor.

### Brand Personality
- **Technical** — VFX professionals are the primary audience
- **Bold** — Cinematic, high-contrast, confident visual language
- **Innovative** — AI/ML-powered, pushing boundaries of what keying can do
- **Communal** — Open source, crew mentality, shared craft

### Target Audience
- VFX artists and compositors (Nuke, DaVinci Resolve, Fusion)
- Independent filmmakers with green screen footage
- Content creators needing professional-grade keying
- Technical users comfortable with Python/CLI (transitioning to GUI)

### Positioning
ez-CorridorKey sits at the intersection of Corridor Digital's cinematic brand authority and accessible open-source tooling. The UI should feel like a professional VFX tool made by people who actually use it — not enterprise software and not a toy.

---

## 2. Visual Identity

### 2.1 Logo System

**Primary Mark:** Corridor Digital's geometric diamond/rhombus icon
- Yellow `#FFF203` on dark backgrounds
- Monochrome black on light backgrounds (rare)
- The diamond divided into angular quadrants with an inner star/cross form
- SVG source: `cdn.watchcorridor.com/assets/config/logo_short_accent.svg`

**Wordmark:** "CorridorKey" in Open Sans Bold, ALL CAPS
- `ez-` prefix in secondary text weight or muted color

**Asset locations:**
- `assets/logos/primary/` — Yellow-on-black diamond mark + wordmark
- `assets/logos/monochrome/` — Single-color variants
- `assets/logos/favicon/` — Icon-only diamond mark for browser/app

### 2.2 Color Palette

**Philosophy:** Warm near-blacks with a single electric yellow accent. The backgrounds carry a subtle yellow/olive undertone — never cold blue-black, never pure `#000000`. Yellow is used sparingly but decisively.

| Role | Hex | CSS Variable |
|------|-----|-------------|
| Brand Accent | `#FFF203` | `--color-brand-accent` |
| Body Background | `#141300` | `--color-bg-primary` |
| Nav Background | `#0E0D00` | `--color-bg-deepest` |
| Card/Panel | `#1E1D13` | `--color-bg-card` |
| Hover Surface | `#454430` | `--color-bg-elevated` |
| Primary Text | `#FFFFFF` | `--color-text-primary` |
| Secondary Text | `#9CA3AF` | `--color-text-muted` |
| Interactive Blue | `#009ADA` | `--color-link` |
| Border | `#565546` | `--color-border-primary` |
| Error/Live | `#D10000` | `--color-error` |
| Warning | `#EC942C` | `--color-warning` |

**Coolors.co Interactive Palette:**
https://coolors.co/0e0d00-141300-fff203-009ada-ffffff

**Token files:** `tokens/colors.css`, `tokens/colors.json`

### Domain Color: Green Screen Green
`#00FF00` — This is the **adversary color**, not a brand color. It appears in the UI only where green screen footage is being processed. The tool's purpose is to *eliminate* this color. Use it contextually (input preview, processing indicators) but never as decoration.

### 2.3 Typography

**Font Family:** Open Sans (Google Fonts, free, Apache 2.0 license)

| Context | Weight | Transform | Tracking | Size |
|---------|--------|-----------|----------|------|
| Navigation | 600 (Semibold) | UPPERCASE | 0.2em | base |
| Section Headers | 700 (Bold) | UPPERCASE | 0.2em | xl |
| Body Copy | 400 (Regular) | none | normal | base (16px) |
| CTA Buttons | 700 (Bold) | UPPERCASE | normal | base |
| Metadata/Labels | 400 (Regular) | none | normal | sm |
| Code/Technical | Consolas | none | normal | sm |

**Token file:** `tokens/typography.css`

### 2.4 Imagery Direction

**Aesthetic:** Cinematic, dark, atmospheric. Content-first.
- Screenshots and previews of keying results are the hero visual
- Before/after comparisons: green screen input → clean foreground output
- Dark UI context — all screenshots on dark backgrounds
- The Corridor diamond mark as subtle watermark on output previews

**What to avoid:**
- Stock photo aesthetics
- Light/airy/pastel treatments
- Rounded, bubbly UI patterns
- Green as a decorative color

---

## 3. Voice & Tone

### Voice Definition
The voice of a VFX professional who genuinely loves their craft and wants to share it. Technical authority delivered casually.

### Tone by Context

| Context | Tone |
|---------|------|
| UI labels & buttons | Direct, concise, ALL CAPS. "RUN INFERENCE", "GENERATE ALPHA" |
| Status messages | Friendly technical. "Processing frame 42/180..." |
| Error messages | Honest, helpful. "VRAM exceeded — try reducing batch size to 4" |
| Documentation | Informal authority. First person. "I built this to solve..." |
| Marketing/README | Enthusiastic but earned. "Perfect Green Screen Keys" |

### Writing Principles
1. **Be direct** — "Run inference" not "Initiate the inference pipeline"
2. **Be communal** — "The Crew", "we", shared ownership language
3. **Be technical when needed** — Don't dumb down sRGB/Linear/EXR concepts
4. **Be casual, not sloppy** — Professional quality with informal delivery
5. **ALL CAPS for emphasis** — Section headers, navigation, CTAs

### Do's and Don'ts

**Do:**
- Use ALL CAPS for navigation and section labels
- Use first person in docs and changelogs
- Reference VFX terminology directly (despill, matte, premultiplied alpha)
- Use "Crew" language when addressing the community

**Don't:**
- Use corporate marketing speak
- Add decorative emojis
- Over-explain fundamentals (the audience knows compositing)
- Use light backgrounds anywhere in the UI

---

## 4. UI Design Principles

### Layout
- **Dark mode only** — no light mode toggle
- **Square corners** — `border-radius: 0` is the default
- **Content-first** — video frames and keying results are the visual substance
- **Minimal chrome** — let the work speak, not the interface

### Component Patterns
- **Buttons:** Yellow fill `#FFF203` with black text for primary CTA; outlined with border for secondary
- **Cards:** `#1E1D13` background, `#565546` border, no radius
- **Navigation:** `#0E0D00` background, white text, yellow underline for active item
- **Progress indicators:** Yellow `#FFF203` fill bar on dark background
- **Input fields:** `#1E1D13` background, `#565546` border, white text, slight radius (2px) only for input fields

### Spacing
- 4px base unit
- 8px, 16px, 24px, 32px, 48px, 64px scale
- Generous whitespace between sections — cinematic, not cramped

---

## 5. Technical Implementation

### CSS Variables
All design tokens are available as CSS custom properties:
- Color: `tokens/colors.css`
- Typography: `tokens/typography.css`
- JSON format: `tokens/colors.json`

### Asset Naming Conventions
- Logos: `corridor-key-{variant}-{size}.{ext}` (e.g., `corridor-key-primary-512.png`)
- Icons: `icon-{name}-{size}.svg`
- Screenshots: `screenshot-{feature}-{resolution}.png`

### Dark Mode Implementation
There is no light mode. The app is dark by default. All color tokens are designed for dark backgrounds. If a component needs contrast, use `--color-bg-card` or `--color-bg-elevated` — never white or light gray backgrounds.

---

## 6. Governance

### Version History
| Date | Version | Changes |
|------|---------|---------|
| 2026-02-28 | 1.0.0 | Initial brand bible — derived from Corridor Digital research |

### Update Process
1. Propose changes in a PR with `brand:` prefix
2. Update tokens (CSS + JSON) for any color/typography changes
3. Update this document to reflect changes
4. Update Coolors.co link if palette changes
