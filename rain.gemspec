Gem::Specification.new do |spec|
  spec.name          = "rain"
  spec.version       = "0.1.0"
  spec.authors       = ["Inela Avdic Hukic"]
  spec.email         = ["inelaah@gmail.com"]

  spec.summary       = "Rain is a simple and clean Jekyll theme with focus on content."
  spec.homepage      = ""
  spec.license       = "MIT"

  spec.files         = Dir.glob("{assets,_layouts,_includes,_sass,LICENSE,README*}", File::FNM_CASEFOLD)

  # ✅ Use more modern versions
  spec.add_runtime_dependency "jekyll", "~> 4.3"
  spec.add_runtime_dependency "jekyll-paginate", "~> 1.1"
  spec.add_runtime_dependency "jekyll-feed", "~> 0.17"

  # ✅ Remove hard pinning to old bundler
  spec.add_development_dependency "bundler", ">= 2.0"
  spec.add_development_dependency "rake", "~> 13.0"
end