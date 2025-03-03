# Install

This project is with `poetry`. I recommend to use a virtual environment created by poetry.

```
poetry install
```

To enter in poetry shell, `poetry shell`.

To install all packages including ones for experiments,

```
poetry install --extras "dask_visual experiment_package"
```

# Examples code

See `./examples`


# Test

```bash
pytest tests/
pytest --nbmake ./example.ipynb
```

Algorithm validation tests

```
pytest -c pytest-algorithm-single.ini algorithm-tests/single
```

# Dev environment with Docker

```
docker compose up -d dev_container
```

```
docker exec -it [container-name] /bin/bash
```

## Deploying worker env. with Docker

```
docker-compose up
```
