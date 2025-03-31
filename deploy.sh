#!/bin/bash
set -e

bundle exec jekyll build

git checkout gh-pages
rm -rf *
cp -r _site/* .
touch .nojekyll

git add .
git commit -m "Deploy at $(date)"
git push origin gh-pages
git checkout master