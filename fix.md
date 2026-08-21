
```
You are working in the `sentinel/` project — a local AI chat UI served by an Express proxy that streams model responses to the browser via Server-Sent Events. Files:
  - public/index.html          (markup)
  - public/js/app.js           (state, localStorage, theme, wiring)
  - public/js/chat.js          (DOM rendering, streaming bubble, typing indicator)
  - public/js/api.js           (fetch wrapper + SSE reader)
  - css/input.css              (Tailwind v4 source; compiled to public/css/styles.css via `npm run build`)

TASK: Fix the UI/UX issues listed below, ONE AT A TIME. Do NOT batch them. For each task:
  1. Make the change.
  2. State what you changed and why.
  3. Give me a one-line "how to verify" note (what to look for in the browser).
  Then stop and wait for my "next" before starting the following issue.

Rules:
- Keep existing code style (vanilla JS, no frameworks; Tailwind utilities + a small `input.css`).
- Do not refactor unrelated code. Do not touch server.js or api.js unless a task says otherwise.
- After editing input.css, run `npm run build` so styles.css is regenerated.
- Confirm each fix works before moving on.

ISSUES (work in this order):

1. HIDDEN DRAWER ON LOAD (critical)
   In public/index.html, #logBackdrop and #chatLogPanel have no `hidden` attribute, so a black overlay + the "Chat Log" panel cover the screen on load. Add `hidden` to BOTH elements. app.js already toggles `.hidden = true/false`, so this is enough.

2. MISSING MESSAGE STYLES (critical)
   chat.js assigns these classes but NONE are styled in css/input.css: .msg, .user, .ai, .bubble-content, .answer, .timestamp, .copy-btn, .code-block-header, .thinking, .log-row, .log-ts. Add component styles so user/AI messages are visually distinct (e.g. user right-aligned bubble, AI left-aligned), timestamps are muted small text, copy buttons look like buttons, code blocks have a header bar, and the "Thinking Process" <details> is styled. Respect light/dark themes via [data-theme].

3. DEAD LOG/EXPORT WIRING (critical)
   app.js defines openLog/closeLog/exportLog/download but has NO addEventListener calls for #logBtn, #logClose, #exportBtn, or the export-menu buttons (#exportMenu). Wire them up: Log opens the drawer, Close + backdrop-click close it, Export toggles the format menu (un-hide #exportMenu), and each format button (data-format="md"/"json") calls exportLog(format).

4. USER VS AI VISUAL ANCHOR
   Ensure every message clearly shows who spoke (alignment + optional "You"/"AI" label). This is largely covered by #2; verify it and fill any gap.

5. ERROR BUBBLE STYLING
   Error replies are rendered as plain text prefixed with "❌". Add a distinct error style (red-tinted border/background) so a failure doesn't look like a normal reply. Add an .error class in chat.js when rendering errors and style it in input.css.

6. GENERATING STATE CLARITY
   The typing indicator is three bouncing dots with no label. Add an accessible "Generating…" status (e.g. visually-hidden text or aria-live="polite" announcement) so the state isn't ambiguous. Keep it subtle.

7. DEDUPE NEW / CLEAR
   #newChat and #clear both call resetChat(). Pick one control (or differentiate them: New = fresh chat, Clear = wipe). Remove the duplicate button from index.html and app.js.

8. A11Y: SCREEN-READER SPAM DURING STREAMING
   <main id="chat"> has role="log" + aria-live="polite", but streaming rewrites innerHTML per token, which spams screen readers. Throttle announcements: announce on completion (and on error), not per token — e.g. set the streamed bubble's container aria-hidden during the stream and only expose final text.

9. POLISH
   - Theme button label reads oddly ("🌙 Dark" while in light mode). Make it clearer (icon-only or "Theme").
   - Replace the magic number in autoExpand() with a named constant / CSS-driven max height.

For each issue, show me the diff-level summary and how to verify it before you continue.
```

