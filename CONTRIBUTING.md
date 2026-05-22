# Contributing to RTMDK

Thank you for your interest in contributing! 🚀

## Development Setup

```bash
git clone https://github.com/butovnikita/RTMDK.git
cd RTMDK
pip install -e ".[dev,sot]"
pre-commit install
```

## Code Style

- **Black** for formatting (`black rtmdk tests`)
- **isort** for import sorting (`isort rtmdk tests`)
- **flake8** for linting (`flake8 rtmdk tests`)
- **mypy** for type checking (`mypy rtmdk`)

## Testing

```bash
# Fast tests
pytest tests/ -x -q

# With coverage
pytest tests/ --cov=rtmdk --cov-report=term-missing

# Specific module
pytest tests/test_tiered_storage.py -v
```

## Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(scope): add new feature
fix(scope): fix bug
docs(scope): update documentation
refactor(scope): code refactoring
test(scope): add tests
chore(scope): maintenance tasks
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Make your changes
4. Run tests (`pytest tests/ -x`)
5. Run linting (`flake8 rtmdk tests`)
6. Commit with conventional format
7. Push and create a Pull Request

## PR Checklist

- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] CI passes (green)
- [ ] Code review addressed

## Questions?

Join our [Discord](https://discord.gg/rtmdk) or open a GitHub Discussion.
