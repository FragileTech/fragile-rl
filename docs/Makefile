.PHONY: build docs serve clean pdf pdf-deps svg-to-pdf

# SVG images directory
SVG_DIR := svg_images
SVG_FILES := $(wildcard $(SVG_DIR)/*.svg)
PDF_FILES := $(SVG_FILES:.svg=.pdf)

# Build the Jupyter Book
build:
	$(MAKE) -C .. prompt
	uv run --with "jupyter-book>=1.0.0,<2.0.0" --with sphinx-tippy --with sphinx-proof --with sphinxcontrib-bibtex --with sphinxcontrib-mermaid jupyter-book build .

# Alias for build
docs: build

# Serve the built site
serve: build
	python3 -m http.server -d _build/html 8000

# Convert SVG files to PDF for LaTeX compatibility
svg-to-pdf: $(PDF_FILES)

$(SVG_DIR)/%.pdf: $(SVG_DIR)/%.svg
	@echo "Converting $< to $@"
	@rsvg-convert -f pdf -o $@ $<

# Build PDF output (requires LaTeX and librsvg2-bin)
# Install deps on Ubuntu/Debian: sudo apt-get install texlive-xetex texlive-fonts-recommended texlive-latex-extra latexmk librsvg2-bin
pdf: svg-to-pdf
	uv run --with "jupyter-book>=1.0.0,<2.0.0" --with sphinx-tippy --with sphinx-proof --with sphinxcontrib-bibtex --with sphinxcontrib-mermaid jupyter-book build . --builder pdflatex

# Install PDF build dependencies (Ubuntu/Debian)
pdf-deps:
	@echo "Installing PDF build dependencies..."
	sudo apt-get update
	sudo apt-get install -y texlive-xetex texlive-fonts-recommended texlive-fonts-extra texlive-latex-extra latexmk librsvg2-bin

# Clean build artifacts
clean:
	rm -rf _build .jupyter_cache
	rm -f $(PDF_FILES)
