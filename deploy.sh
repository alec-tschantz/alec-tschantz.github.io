#!/bin/bash
set -e

echo "🔨 Building Jekyll site..."
bundle exec jekyll build

echo "🚀 Deploying to gh-pages branch..."

# Save current branch name (assumes main, but good for generality)
CURRENT_BRANCH=$(git symbolic-ref --short HEAD)

# Create worktree for gh-pages branch if it doesn't exist
if ! git show-ref --quiet refs/heads/gh-pages; then
  echo "🔧 Creating gh-pages branch..."
  git worktree add -B gh-pages ../gh-pages-tmp
else
  git worktree add gh-pages-tmp gh-pages
fi

# Copy built site to gh-pages
rm -rf gh-pages-tmp/*
cp -r _site/* gh-pages-tmp/
touch gh-pages-tmp/.nojekyll

cd gh-pages-tmp
git add .
git commit -m "Deploy at $(date)" || echo "No changes to commit"
git push origin gh-pages
cd ..

# Clean up
git worktree remove gh-pages-tmp

echo "✅ Deployment done! Site is live at: https://<your-username>.github.io"
