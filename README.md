# alec-tschantz.github.io

Plain HTML, CSS and JS, served as-is by GitHub Pages. No build step.

```sh
python3 -m http.server 8000   # then open http://localhost:8000
```

- `index.html`: the home page and post list
- `assets/site.js`: the shared header (name and social links)
- `assets/hero.js`: the animated contour backdrop
- `assets/post.js`: maths (KaTeX) and code highlighting for posts
- `assets/style.css`: all styles

## Adding a post

A post is any HTML page, so it can hold figures, canvas or full interactive visualisations.
Create `posts/my-post.html` (or `posts/my-post/index.html` with its files alongside):

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>My post · Alexander Tschantz</title>
  <link rel="icon" href="/assets/favicon.svg">
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&family=IBM+Plex+Sans:wght@400;500&display=swap">
  <link rel="stylesheet" href="/assets/style.css">
  <script src="/assets/site.js"></script>
  <script src="/assets/post.js" defer></script>
</head>
<body class="page-body">
  <site-header></site-header>
  <main class="page">
    <a class="back" href="/">← All posts</a>
    <h1 class="page-title">My post</h1>
    <time class="post-date" datetime="2026-10-22">22 Oct 2026</time>
    <article class="prose">
      <p>Inline maths like $q(s)$, and display maths on its own line:</p>
      <p>$$F = \mathbb{E}_{q(s)}[\ln q(s) - \ln p(o, s)]$$</p>
      <pre><code class="language-python">print("code is highlighted")</code></pre>
    </article>
  </main>
</body>
</html>
```

Then add it to the top of the list in `index.html`:

```html
<li>
  <a href="/posts/my-post.html">
    <time datetime="2026-10-22">22 Oct 2026</time>
    <span class="title">My post</span>
  </a>
</li>
```
