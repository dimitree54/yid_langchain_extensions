# Langchain Extensions

A collection of useful extensions for the LangChain library.

## Installation

```bash
pip install yid-langchain-extensions
```

For development:

```bash
pip install -e ".[dev]"
```

## Development

### Running Tests

Tests can be run in parallel using unittest-parallel:

```bash
python -m unittest_parallel -s tests
```

### Running Tests Sequentially

If you need to run tests sequentially, you can use the standard unittest module:

```bash
python -m unittest discover
```
