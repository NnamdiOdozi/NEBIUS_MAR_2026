## Coding Style Guide

**This guide should be read at the start of every coding project and reflected on at the end of each session to see if there is anything that should be added based on feedback during the session.**

### Configuration and Constants
- Use `config.toml` for all tuneable parameters: word counts, file size limits, timeouts, exclusion patterns, provider settings, LLM and other API configuration (model names, base URLs, auth env var names, frequency penalties, timeouts). Users should never need to edit Python code to change operational defaults.
- API keys and secrets stay in `.env` / `.env.claude` (gitignored). Configuration that is not secret goes in `config.toml` (committed to repo).
- No hardcoding of values that may differ between local development and server deployment. If a value could change per environment, it belongs in config.
- **Config file consistency and comments:** When a config file has multiple parallel sections (e.g. `[llm.doubleword]`, `[llm.openai]`, `[llm.nebius]`), keep the structure as consistent as possible across sections. If a parameter appears in one section but not others, add an inline comment on that parameter explaining why it is specific to that section (e.g. "reasoning model only — would cause API errors on other providers"). If a parameter is present in all sections but has different semantics or constraints in one of them, note that too. If a key exists purely for documentation and is not read by code, mark it explicitly as `# informational only — not read by code`. The goal: a fresh session reading the config file should understand the full picture without needing to read the source code.

### Server Deployment
- Every deployable project should include a `systemd` service file and a `setup.sh` bash script for reproducible server setup.
- Be explicit about `PATH` in systemd service files — virtual environment binaries, system binaries (`/usr/bin`, `/usr/local/bin`) must all be included. This is a common source of "works locally, fails on server" bugs.
- Keep local dev and server environments consistent. If a tool (e.g., `gitingest`) is installed in a venv locally, ensure the code can find it in a venv on the server too (`sys.executable` path lookup pattern).

### Code Style
- Keep code short and direct. One or two error handling cases is sufficient — do not code for every hypothetical edge case. Maintainability matters more than defensive completeness.
- Use functions when logic is repeated more than twice. Three similar blocks of code should become a function.
- Avoid bloated code that handles too many eventualities. Simple code with clear failure modes is easier to debug than over-engineered code with graceful fallbacks nobody will ever trigger.
- Prefer flat code over deeply nested conditionals. Early returns are clearer than nested if/else chains.

### Linting and Import Checks
- Run `ruff check .` from the project root before committing Python code. Fix all errors; do not suppress warnings without a comment explaining why.
- After any file rename or directory restructure, verify imports still resolve: `python -c "import app; import api"` (adjust to the project's top-level packages). A misnamed file (e.g. `__init__,py` instead of `__init__.py`) will not raise an error at rename time but will silently break imports at runtime or packaging time.

### Logging
- Use Python's `logging` module with timestamps and log levels — not bare `print()` statements. Format should include timestamp, level, and message at minimum (e.g., `2025-06-15 14:32:01 INFO  Processing repo owner/repo`).
- Log to a file in a dedicated `logs/` directory at the project root (e.g., `logs/api.log`). This directory should be gitignored.
- Use `RotatingFileHandler` to prevent unbounded log growth. Configure `maxBytes` and `backupCount` in `config.toml` so they can be tuned without code changes. A good default is `maxBytes=150_000` (~1000 lines at ~150 chars/line) with `backupCount=5` (keeping `api.log` + 5 rotated files).
- Log enough context to trace what happened: request parameters, which external calls were made, timing of slow operations (LLM calls, subprocess runs), and full tracebacks on errors.
- Replace ad-hoc `print(..., file=sys.stderr)` debug statements with proper logger calls before shipping. Debug prints are fine during development but should not persist in committed code.
- Generated output files (digests, summaries) should also go in dedicated directories (e.g., `git_summaries/`) to keep the project root clean. These directories should be gitignored.
- Cap the number of generated output files. Use a `max_summaries` config parameter and a `cleanup_summaries()` function that deletes the oldest pairs (e.g., `.txt` + `_llm.json`) when the count exceeds the cap. 20 is a sensible default. Run cleanup at the start of each job before writing new output.

### Testing
- Keep test outputs in a dedicated `tests/` directory at the project root, gitignored.
- Each test should capture both **inputs** (the parameters or request body sent) and **outputs** (the response or result received) so that you can reconcile what went in vs what came out. Store these together — either as fields in a single JSON file, columns in a TSV, or equivalent structured format.
- Name test output files descriptively so the scenario is obvious from the filename (e.g., `test_basic_no_llm.json`, `test_with_focus.json`, `test_openai_llm.json`).
- For API testing, save the full request payload alongside the response. This makes it easy to reproduce issues and compare runs over time.

### README Standards
- Every README should include an **effective date** indicating when the documentation was last verified against the codebase.
- Document **known dependencies** on external tools and their output formats. Flag what would break if those tools change and what would need updating.
- Document **breaking changes** and how to handle them (e.g., "if gitingest changes its output format, update the tree parser in `app/main.py`").
- Keep the README in sync with the code. When adding a new endpoint, config option, or feature, update the README in the same change.

### Project Structure
- Separate concerns: `api/` for routes and HTTP handling, `app/` for business logic, `config.toml` for settings, `.env.claude` for secrets.
- Use FastAPI routers to group related endpoints by domain rather than putting everything in one file.
- Gitignore generated outputs, test artifacts, environment files, and build artifacts.

## Developer Education Goals

The user is actively developing their Python and Git/GitHub skills. When opportunities arise naturally in the code, point them out and briefly explain the concept. Do not force them into every piece of code, but when a pattern fits well, suggest it and explain why.

### Python concepts to introduce when relevant
- **Type annotations** — function signatures with `->` return types and parameter types; `typing` module (`Optional`, `Union`, `list[str]`); Pydantic models as an alternative to plain dicts for structured data
- **Protocols and ABCs** — when to define an interface vs a base class; `typing.Protocol` for structural subtyping; `abc.ABC` and `@abstractmethod`
- **Generators and iterators** — `yield`, `__iter__`/`__next__`; when a generator is more memory-efficient than building a list; `itertools`
- **Decorators** — `@functools.wraps`, timing/logging decorators, `@property`; when a decorator reduces repetition cleanly
- **Wrappers and context managers** — `__enter__`/`__exit__`, `contextlib.contextmanager`

### Git/GitHub concepts to introduce when relevant
- Branch strategies (feature branches, when to use them)
- Rebasing vs merging and when each is appropriate
- Git hooks (pre-commit, post-merge) for automation
- GitHub Actions — extending existing workflows, adding new triggers
- Tagging and semantic versioning for releases
- `git bisect`, `git stash`, `git reflog` for debugging and recovery

## Available Skills
**dw_batch:** Async batch processing using Doubleword API. Process multiple PDFs/DOCXs cost-effectively (50-85% cheaper) for non-urgent summarization, analysis, OCR, and extraction tasks. **Suggest for any token-heavy non-urgent work (10K+ tokens).**

### dw_batch Agent Checklist
1. **STOP and read SKILL.md fully** before ANY batch operations. **MANDATORY: Read GUIDE.md BEFORE proceeding** when: (a) any file is skipped, (b) estimated tokens >20K input or >5K output, (c) you need per-file prompts or conditional logic.
2. **Tier 2 triggers** (require custom code): per-file prompts, conditional logic, docs >128K tokens (~360K chars)
3. **Script selection** - do NOT mix file types:
   - `create_batch.py` → PDF, DOCX, TXT, CSV, XLS, XLSX (text extraction)
   - `create_image_batch.py` → PNG, JPG, JPEG only (vision model)
   - `create_scanned_pdf_batch.py` → scanned PDFs (OCR via vision)
   - `create_embeddings_batch.py` → any format for embeddings
4. **Always specify batch file** explicitly when submitting; poll batches in submission order
5. **Use `--dry-run`** for large batches
6. **Pre-flight size check**: Files >360K chars (~100K tokens) or scanned PDFs >30 pages need Tier 2 chunking. **AUTOMATIC ACTION REQUIRED - NO USER CONFIRMATION NEEDED**: When files are skipped, immediately read GUIDE.md 'Handling Long Documents' section and process them with chunking. This is not optional. Do not ask "would you like me to...?" - just do it.
7. **Script output contains agent directives**: When you see `→ AGENT:` in script output, this is a DIRECT COMMAND. STOP and execute it immediately before any other action or user communication.