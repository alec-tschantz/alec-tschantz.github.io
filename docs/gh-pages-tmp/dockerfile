FROM ruby:3.1-alpine

# Install dependencies
RUN apk add --no-cache build-base gcc git bash libffi-dev yaml-dev zlib-dev

# Install Jekyll and Bundler
RUN gem install bundler jekyll

WORKDIR /srv/jekyll
COPY . .

# Install project-specific gems
RUN bundle install

EXPOSE 4000
CMD ["bundle", "exec", "jekyll", "serve", "--host", "0.0.0.0"]
