#!/bin/bash
set -e

echo "🔨 Building Jekyll site in Docker..."
docker-compose run --rm jekyll bundle exec jekyll build

echo "🚀 Deploying to gh-pages branch..."

TMP_DIR=../gh-pages-tmp

echo "🧼 Pruning broken Git worktree metadata..."
git worktree prune

# Clean up existing or broken worktree
if git worktree list | grep -q 'gh-pages-tmp'; then
  echo "🧹 Removing registered gh-pages-tmp worktree..."
  git worktree remove $TMP_DIR --force || true
fi

if [ -d "$TMP_DIR" ]; then
  echo "🗑️  Removing leftover $TMP_DIR directory..."
  rm -rf "$TMP_DIR"
fi

# Create worktree for gh-pages branch
if ! git show-ref --quiet refs/heads/gh-pages; then
  echo "🔧 Creating new gh-pages branch..."
  git worktree add -B gh-pages $TMP_DIR
else
  git worktree add $TMP_DIR gh-pages
fi

# Copy built site into worktree
rm -rf $TMP_DIR/*
cp -r _site/* $TMP_DIR/
touch $TMP_DIR/.nojekyll

cd $TMP_DIR
git pull origin gh-pages --rebase
git add .
git commit -m "Deploy at $(date)" || echo "⚠️  No changes to commit"
git push origin gh-pages
cd ..

# Final clean up
git worktree remove $TMP_DIR

echo "✅ Deployment done! Site is live at: https://alec-tschantz.github.io"
