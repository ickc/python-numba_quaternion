# Contributing

Contributions are welcome. Please report bugs and request features at <https://github.com/ickc/python-numba_quaternion/issues>.

## Development

Development environments are managed by [pixi](https://pixi.sh):

```sh
pixi run test            # run tests with the latest Python
pixi run -e min test     # run tests with the oldest supported dependencies
pixi run -e py310 test   # run tests with a specific Python version
pixi run lint            # lint with ruff
pixi run docs            # build the docs to dist/docs
pixi run build           # build sdist and wheel to dist/
```

If you prefer not to use pixi, `uv run pytest` also works.

## Releasing

1. Bump the version with `uv version --bump minor` (or `major`/`patch`) and update `CHANGELOG.md`.
2. Commit, tag and push, e.g. `git tag v0.3.0 && git push --follow-tags`.
3. The `Release` workflow tests, builds, publishes to PyPI via trusted publishing, and creates a GitHub release.
