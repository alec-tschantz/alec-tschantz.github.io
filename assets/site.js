// Shared header. Load this in <head> on every page, then drop in <site-header></site-header>.

const SOCIAL = [
  {
    label: "Google Scholar",
    href: "https://scholar.google.com/citations?user=5NbVgO0AAAAJ",
    icon: '<path d="M5.242 13.769 0 9.5 12 0l12 9.5-5.242 4.269C17.548 11.249 14.978 9.5 12 9.5c-2.977 0-5.548 1.748-6.758 4.269zM12 10a7 7 0 1 0 0 14 7 7 0 0 0 0-14z"/>',
  },
  {
    label: "LinkedIn",
    href: "https://www.linkedin.com/in/alexander-tschantz/",
    icon: '<path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 1 1 0-4.125 2.062 2.062 0 0 1 0 4.125zM7.119 20.452H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>',
  },
  {
    label: "Email",
    href: "mailto:tschantz.alec@gmail.com",
    icon: '<path fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round" d="M3 5h18a1 1 0 0 1 1 1v12a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1ZM2 6l10 7 10-7"/>',
  },
  {
    label: "X / Twitter",
    href: "https://x.com/a_tschantz",
    icon: '<path d="M18.901 1.153h3.68l-8.04 9.19L24 22.846h-7.406l-5.8-7.584-6.638 7.584H.474l8.6-9.83L0 1.154h7.594l5.243 6.932ZM17.61 20.644h2.039L6.486 3.24H4.298Z"/>',
  },
];

class SiteHeader extends HTMLElement {
  connectedCallback() {
    const social = SOCIAL.map(({ label, href, icon }) => {
      const external = href.startsWith("http") ? ' target="_blank" rel="noopener"' : "";
      return `<li><a href="${href}" aria-label="${label}" title="${label}"${external}><svg viewBox="0 0 24 24" aria-hidden="true">${icon}</svg></a></li>`;
    }).join("");

    this.innerHTML = `
      <a class="wordmark" href="/">Alexander Tschantz</a>
      <ul class="social">${social}</ul>`;
  }
}

customElements.define("site-header", SiteHeader);
