.PHONY: style check test tldr tldr-html tldr-debug tldr-fallback check-tldr-deps prompt claude mlflow physics-code

style:
	uv run ruff check --fix-only --unsafe-fixes .
	uv run ruff format .

check:
	uv run ruff check .
	uv run ruff format --diff .

test:
	PYGLET_HEADLESS=1 uv run pytest tests/

tldr:
	@echo "Generating PDF with rendered Mermaid diagrams..."
	@python3 generate_pdf_with_mermaid.py
	@echo "✓ PDF generated: single_agent_tldr.pdf"

tldr-debug:
	@echo "Generating PDF (debug mode - keeping temp files)..."
	@python3 generate_pdf_with_mermaid.py --debug
	@echo "✓ PDF generated: single_agent_tldr.pdf"
	@echo "  Temp files retained in /tmp/tldr_build_*/"

tldr-fallback:
	@echo "Generating PDF without Mermaid rendering (fallback)..."
	@pandoc single_agent_tldr.md -o single_agent_tldr.pdf \
		--pdf-engine=xelatex \
		-V geometry:margin=1in \
		-V fontsize=11pt
	@echo "✓ PDF generated: single_agent_tldr.pdf"

check-tldr-deps:
	@echo "Checking dependencies for PDF generation..."
	@which mmdc > /dev/null 2>&1 || (echo "✗ mmdc not found. Install: npm install -g @mermaid-js/mermaid-cli" && exit 1)
	@which pandoc > /dev/null 2>&1 || (echo "✗ pandoc not found. Install: apt-get install pandoc" && exit 1)
	@echo "✓ All dependencies available"

tldr-html:
	@chmod +x generate_html.sh
	@bash generate_html.sh

prompt:
	@echo "Collecting prf directives into prompts/..."
	@python3 docs/collect_prf_directives.py --include-proofs --include-file-headings --out-dir prompts
	@echo "Preparing prompt downloads for docs..."
	@python3 docs/build_prompt_downloads.py
	@echo "✓ Prompts generated in prompts/"

mlflow:
	uv run mlflow server --host 127.0.0.1 --port 5000

code:
	@echo "# RL Code" > rl_code.md
	@echo "" >> rl_code.md
	@find src/fragile -name '*.py' | sort | while read f; do \
		echo "## $$f" >> rl_code.md; \
		echo "" >> rl_code.md; \
		echo '```python' >> rl_code.md; \
		cat "$$f" >> rl_code.md; \
		echo "" >> rl_code.md; \
		echo '```' >> rl_code.md; \
		echo "" >> rl_code.md; \
	done
	@echo "✓ rl_code.md generated"

claude:
	CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1 claude --dangerously-skip-permissions
