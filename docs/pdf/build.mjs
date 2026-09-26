// Build every writeup in docs/writeups/ into a PDF next to it, in docs/writeups/pdf/.
//
//   cd docs/pdf && npm install && node build.mjs            # all writeups
//   node build.mjs c6-strategy-zoo.md c1-glft-baseline.md    # just these
//
// Why this and not pandoc + LaTeX: the machine these were written on has no
// TeX, and a TeX install is a gigabyte the market data needs more. KaTeX
// renders the same LaTeX subset the writeups use, at build time, to HTML;
// Chromium (via Playwright) prints it. No CDN is touched at runtime, so the
// PDFs build offline once `npm install` has run.
//
// Every formula is also parsed once with throwOnError on, and the build
// fails if any does not parse. "The formulas render" is therefore checked,
// not assumed: an unparseable formula would otherwise be drawn in red inside
// the PDF, where nobody reads it until it matters.

import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";
import katex from "katex";
import MarkdownIt from "markdown-it";
import anchor from "markdown-it-anchor";
import texmath from "markdown-it-texmath";

const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, "..", "..");
const writeups = path.join(repo, "docs", "writeups");
const outDir = path.join(writeups, "pdf");
const REPO_URL = "https://github.com/davidmiheev/rl-ladder/blob/claude/rl-ladder-setup-docs-d43b31";

function loadPlaywright() {
  // Prefer a local install; fall back to the global one this machine ships
  // with, whose Chromium is already on disk.
  for (const where of ["playwright", "/opt/node22/lib/node_modules/playwright"]) {
    try { return require(where); } catch { /* try the next */ }
  }
  throw new Error("playwright not found: npm install playwright, or install it globally");
}

// ---- math validation --------------------------------------------------------
function findMath(markdown) {
  // Strip fenced and inline code first: a `$` inside code is not math.
  const text = markdown.replace(/```[\s\S]*?```/g, "").replace(/`[^`\n]*`/g, "");
  const out = [];
  for (const m of text.matchAll(/\$\$([\s\S]+?)\$\$/g)) out.push({ tex: m[1], display: true });
  const inline = text.replace(/\$\$[\s\S]+?\$\$/g, "");
  // Pandoc's rule, which markdown-it-texmath follows: an opening $ is not
  // followed by a space, a closing $ is not preceded by one or followed by a
  // digit. That is what keeps "$5,000 ... $100,000" from being read as math.
  for (const m of inline.matchAll(/(?<![\\$])\$(?!\s)((?:[^$\\\n]|\\.)+?)(?<!\s)\$(?!\d)/g)) {
    out.push({ tex: m[1], display: false });
  }
  return out;
}

function validate(file, markdown) {
  const errors = [];
  for (const { tex, display } of findMath(markdown)) {
    try { katex.renderToString(tex, { displayMode: display, throwOnError: true, strict: "ignore" }); }
    catch (err) { errors.push(`${file}: ${String(err.message).split("\n")[0]}\n    ${tex.trim().slice(0, 120)}`); }
  }
  return errors;
}

// ---- markdown -> html --------------------------------------------------------
function renderer() {
  const md = new MarkdownIt({ html: true, linkify: true, typographer: false });
  md.use(texmath, {
    engine: katex,
    delimiters: "dollars",
    katexOptions: { throwOnError: false, strict: "ignore", output: "html" },
  });
  // GitHub's heading slugs, so a contents list works on GitHub and in the PDF:
  // lowercase, drop everything but word characters, spaces and hyphens, then
  // spaces to hyphens ("3.1 What they share" -> "31-what-they-share").
  md.use(anchor, {
    slugify: (s) => s.trim().toLowerCase().replace(/[^\p{L}\p{N}_\- ]/gu, "").replace(/ /g, "-"),
  });

  // Links: another writeup becomes its PDF; anything else in the repo
  // becomes a GitHub link, since a relative path is dead inside a PDF.
  const defaultLink = md.renderer.rules.link_open || ((t, i, o, e, s) => s.renderToken(t, i, o));
  md.renderer.rules.link_open = (tokens, idx, options, env, self) => {
    const href = tokens[idx].attrGet("href") || "";
    if (href && !/^(https?:|mailto:|#)/.test(href)) {
      const [target, frag] = href.split("#");
      const abs = path.resolve(env.dir, target);
      if (path.dirname(abs) === writeups && abs.endsWith(".md")) {
        tokens[idx].attrSet("href", path.basename(abs, ".md") + ".pdf" + (frag ? "#" + frag : ""));
      } else if (abs.startsWith(repo)) {
        tokens[idx].attrSet("href", `${REPO_URL}/${path.relative(repo, abs)}${frag ? "#" + frag : ""}`);
      }
    }
    return defaultLink(tokens, idx, options, env, self);
  };
  return md;
}

const CSS = `
@page { size: A4; margin: 18mm 17mm 20mm 17mm; }
@page wide { size: A4 landscape; margin: 12mm; }
html { font-size: 10.5pt; }
body { font-family: "DejaVu Serif", "Liberation Serif", Georgia, serif; line-height: 1.5;
       color: #1a1a1a; max-width: none; margin: 0; }
body.wide { page: wide; font-size: 8pt; }
h1 { font-size: 1.9em; line-height: 1.2; margin: 0 0 .6em; border-bottom: 2px solid #333; padding-bottom: .25em; }
h2 { font-size: 1.35em; margin: 1.6em 0 .5em; border-bottom: 1px solid #bbb; padding-bottom: .15em; break-after: avoid; }
h3 { font-size: 1.12em; margin: 1.3em 0 .4em; break-after: avoid; }
h4 { font-size: 1em; margin: 1.1em 0 .3em; break-after: avoid; }
p, li { orphans: 3; widows: 3; }
a { color: #1a4f9c; text-decoration: none; }
code { font-family: "DejaVu Sans Mono", "Liberation Mono", monospace; font-size: .86em;
       background: #f3f3f3; padding: .05em .25em; border-radius: 3px; }
pre { background: #f6f6f6; border: 1px solid #e2e2e2; border-radius: 4px; padding: .6em .8em;
      white-space: pre-wrap; word-break: break-word; break-inside: avoid; font-size: .92em; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; margin: .8em 0; font-size: .86em; break-inside: auto; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
th, td { border: 1px solid #ccc; padding: .25em .5em; vertical-align: top; }
th { background: #eef1f5; }
tr:nth-child(even) td { background: #fafafa; }
img { max-width: 100%; max-height: 95mm; width: auto; display: block; margin: .8em auto; break-inside: avoid; }
blockquote { border-left: 3px solid #999; margin: .8em 0; padding: .1em 1em; color: #444; }
hr { border: 0; border-top: 1px solid #ccc; margin: 1.5em 0; }
/* markdown-it-texmath wraps inline math in <eq> and display math in <eqn>. */
eq { display: inline; }
eqn { display: block; }
.katex-display { margin: .7em 0; overflow-x: auto; overflow-y: hidden; }
.katex { font-size: 1.05em; }
`;

function html(title, body, wide) {
  const katexCss = pathToFileURL(require.resolve("katex/dist/katex.min.css")).href;
  return `<!doctype html><html><head><meta charset="utf-8"><title>${title}</title>
<link rel="stylesheet" href="${katexCss}"><style>${CSS}</style></head>
<body class="${wide ? "wide" : ""}">${body}</body></html>`;
}

// ---- main --------------------------------------------------------------------
const wanted = process.argv.slice(2);
const files = fs.readdirSync(writeups)
  .filter((f) => f.endsWith(".md"))
  .filter((f) => wanted.length === 0 || wanted.includes(f))
  .sort();
// A name that matches nothing used to build nothing and say nothing.
const unknown = wanted.filter((w) => !files.includes(w));
if (unknown.length) {
  console.error(`no such writeup: ${unknown.join(", ")} (pass bare names, e.g. c7-signals.md)`);
  process.exit(1);
}

const problems = files.flatMap((f) => validate(f, fs.readFileSync(path.join(writeups, f), "utf8")));
if (problems.length) {
  console.error(`${problems.length} formula(s) do not parse:\n` + problems.join("\n"));
  process.exit(1);
}

fs.mkdirSync(outDir, { recursive: true });
const md = renderer();
const { chromium } = loadPlaywright();
const browser = await chromium.launch();
const page = await browser.newPage();
for (const f of files) {
  const source = fs.readFileSync(path.join(writeups, f), "utf8");
  const title = (source.match(/^#\s+(.+)$/m) || [null, f])[1];
  // The two table dumps are 15-21 columns wide and only readable landscape.
  const wide = /-(table|compare)\.md$/.test(f);
  const body = md.render(source, { dir: writeups });
  // Written beside the markdown, so relative image paths resolve exactly as
  // they do on GitHub.
  const tmp = path.join(writeups, `.${f}.render.html`);
  fs.writeFileSync(tmp, html(title, body, wide));
  await page.goto(pathToFileURL(tmp).href, { waitUntil: "networkidle" });
  const out = path.join(outDir, f.replace(/\.md$/, ".pdf"));
  await page.pdf({
    path: out, format: "A4", landscape: wide, printBackground: true,
    preferCSSPageSize: true, displayHeaderFooter: true,
    headerTemplate: `<div style="font-size:7pt;color:#888;width:100%;text-align:right;padding-right:12mm">${title.replace(/</g, "&lt;")}</div>`,
    footerTemplate: `<div style="font-size:7pt;color:#888;width:100%;text-align:center"><span class="pageNumber"></span> / <span class="totalPages"></span></div>`,
  });
  fs.unlinkSync(tmp);
  console.log(`wrote ${path.relative(repo, out)}`);
}
await browser.close();
