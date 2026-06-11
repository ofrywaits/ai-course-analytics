---
name: AI Analytics Framework
colors:
  surface: '#0e1417'
  surface-dim: '#0e1417'
  surface-bright: '#333a3d'
  surface-container-lowest: '#080f12'
  surface-container-low: '#161d1f'
  surface-container: '#1a2123'
  surface-container-high: '#242b2e'
  surface-container-highest: '#2f3639'
  on-surface: '#dde3e7'
  on-surface-variant: '#bbc9cf'
  inverse-surface: '#dde3e7'
  inverse-on-surface: '#2b3134'
  outline: '#859398'
  outline-variant: '#3c494e'
  surface-tint: '#3cd7ff'
  primary: '#a8e8ff'
  on-primary: '#003642'
  primary-container: '#00d4ff'
  on-primary-container: '#00586b'
  inverse-primary: '#00677e'
  secondary: '#bac9d0'
  on-secondary: '#243238'
  secondary-container: '#3d4b52'
  on-secondary-container: '#abbbc2'
  tertiary: '#ffd9a1'
  on-tertiary: '#432c00'
  tertiary-container: '#feb528'
  on-tertiary-container: '#6c4900'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#b4ebff'
  primary-fixed-dim: '#3cd7ff'
  on-primary-fixed: '#001f27'
  on-primary-fixed-variant: '#004e5f'
  secondary-fixed: '#d5e5ed'
  secondary-fixed-dim: '#bac9d0'
  on-secondary-fixed: '#0f1d23'
  on-secondary-fixed-variant: '#3b494f'
  tertiary-fixed: '#ffdeae'
  tertiary-fixed-dim: '#ffba3d'
  on-tertiary-fixed: '#281900'
  on-tertiary-fixed-variant: '#604100'
  background: '#0e1417'
  on-background: '#dde3e7'
  surface-variant: '#2f3639'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 48px
    fontWeight: '700'
    lineHeight: 56px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
    letterSpacing: -0.01em
  headline-lg-mobile:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
  headline-md:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
  title-lg:
    fontFamily: Inter
    fontSize: 20px
    fontWeight: '500'
    lineHeight: 28px
  body-lg:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  label-md:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.05em
  mono-data:
    fontFamily: JetBrains Mono
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  base: 4px
  xs: 8px
  sm: 16px
  md: 24px
  lg: 32px
  xl: 48px
  container-max: 1440px
  gutter: 24px
---

## Brand & Style

The design system is engineered for high-density data environments where precision and clarity are paramount. It targets an audience of educators, data scientists, and administrators who require immediate, actionable insights from complex AI-driven datasets. 

The visual style is **Corporate / Modern** with a distinct technical edge, drawing inspiration from high-performance development environments and Streamlit-style dashboards. It prioritizes a "content-first" approach, utilizing high-contrast typography and a dark, low-fatigue background to make vibrant data visualizations and primary accents pop. The overall emotional response should be one of professional reliability, intellectual rigor, and cutting-edge technological capability.

## Colors

The palette is anchored by a deep obsidian background to minimize ocular strain during long analytical sessions. The **Primary Cyan (#00D4FF)** is used sparingly for interactive elements, progress indicators, and key brand touchpoints to maintain its high-energy impact.

Data visualization should utilize a distinct, vibrant secondary palette (Violet, Emerald, Amber, and Rose) that maintains high contrast against the dark background. Neutral tones are strictly cool-shifted to maintain a cohesive, high-tech atmosphere, moving from deep charcoal surfaces to crisp white for primary text.

## Typography

The typography system relies on **Inter** for its exceptional legibility in digital interfaces and systematic weight distribution. For data-heavy components such as tables or logs, a monospaced font is introduced to ensure numerical alignment.

Standardize on `Inter` for all UI controls and narrative text. Use `JetBrains Mono` for specific data values or code snippets. Headlines should utilize tighter letter spacing to maintain a structured, "engineered" appearance. Body text remains generous in line height to prevent information density from becoming overwhelming.

## Layout & Spacing

This design system utilizes a **12-column fluid grid** for dashboard views, transitioning to a single-column stack on mobile devices. The layout is built on an 8px rhythmic scale to ensure consistent alignment across all custom components.

**Breakpoints:**
- Mobile: < 600px (4 columns, 16px margins)
- Tablet: 600px - 1024px (8 columns, 24px margins)
- Desktop: > 1024px (12 columns, 32px margins)

Metric cards should span 3 or 4 columns on desktop, while primary data visualizations should span 8 to 12 columns to allow for granular detail. Vertical spacing between dashboard sections should be consistently `xl` (48px) to provide visual breathing room between distinct data clusters.

## Elevation & Depth

Hierarchy is established through **Tonal Layers** and **Low-contrast Outlines** rather than traditional shadows. This creates a flat, professional "dashboard" aesthetic that feels integrated and stable.

- **Level 0 (Background):** #0E1117 - The base canvas.
- **Level 1 (Cards/Containers):** #161B22 - Surfaces for content. These must have a 1px solid border (#30363D).
- **Level 2 (Popovers/Modals):** #1C2128 - Elevated surfaces for temporary interaction. These use a subtle 8px blur shadow with 20% opacity black to separate from the Level 1 surfaces.

Avoid using drop shadows on standard dashboard cards; use the 1px border to define boundaries.

## Shapes

The shape language is precise and disciplined. A **Soft (Level 1)** roundedness is applied to all standard components to strike a balance between modern friendliness and professional rigidity.

- **Standard Buttons/Inputs:** 0.25rem (4px)
- **Metric Cards:** 0.5rem (8px)
- **Status Banners:** 0.25rem (4px) or Sharp (0px) depending on the container context.

Interactive elements like radio buttons and checkboxes remain traditional (fully round or slightly rounded squares) to ensure instant recognition of their function.

## Components

### Dashboard Metric Cards
Cards feature a `title-lg` for the label and a `display-lg` for the primary metric. Optional sparkline charts should use the primary accent color. Background is Level 1 Surface with Level 1 Border.

### Tab Navigation
Tabs use a minimal underline style. The active tab is indicated by a 2px Cyan (#00D4FF) bottom border and a weight shift to Medium. Inactive tabs are `secondary_color`.

### Form Inputs
- **Sliders:** The track is a dark neutral (#30363D), while the filled portion and thumb are Primary Cyan.
- **Dropdowns/Selects:** Use Level 1 Surface backgrounds. On hover, the border changes to Primary Cyan.
- **Radio Buttons:** Use a standard circle; when selected, a Primary Cyan dot is centered within.

### Status Banners
Banners are full-width or card-width with a subtle tinted background (e.g., 10% opacity of the status color) and a 2px left-accent border of the solid status color. Text should be `body-md` with high contrast.

### Data Tables
Rows are separated by Level 1 borders. Header rows use `label-md` in all-caps with a slightly darker background than the body rows to establish clear structure.