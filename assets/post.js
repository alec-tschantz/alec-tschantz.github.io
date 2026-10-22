// Maths (KaTeX) and code highlighting (highlight.js) for post pages. Load with `defer`.
// Write maths as $inline$ or $$display$$, and code as <pre><code class="language-python">.

const CDN = "https://cdnjs.cloudflare.com/ajax/libs";

const load = (src) => new Promise((resolve, reject) =>
  document.head.append(Object.assign(document.createElement("script"), { src, onload: resolve, onerror: reject })));

document.head.insertAdjacentHTML("beforeend", `<link rel="stylesheet" href="${CDN}/KaTeX/0.16.11/katex.min.css">`);

load(`${CDN}/highlight.js/11.9.0/highlight.min.js`).then(() => hljs.highlightAll());

load(`${CDN}/KaTeX/0.16.11/katex.min.js`)
  .then(() => load(`${CDN}/KaTeX/0.16.11/contrib/auto-render.min.js`))
  .then(() => renderMathInElement(document.body, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false },
    ],
  }));
