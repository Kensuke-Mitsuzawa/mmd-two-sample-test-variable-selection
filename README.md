# What's this?

This is the codebase of Two-Sample-Test Variable Selection by Maximum Mean Discrepancy.

### References

Please refer to either of these works.

```
@misc{mitsuzawa2023variableselectionmaximummean,
      title={Variable Selection in Maximum Mean Discrepancy for Interpretable Distribution Comparison}, 
      author={Kensuke Mitsuzawa and Motonobu Kanagawa and Stefano Bortoli and Margherita Grossi and Paolo Papotti},
      year={2023},
      eprint={2311.01537},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2311.01537}, 
}
```


```
@phdthesis{EURECOM+7711,
  author = {Mitsuzawa, Kensuke},
  title = {Comparing high-dimension data: Interpretable two-sample testing by variable eslection},
  year = {2024},
  note = {© EURECOM. Personal use of this material is permitted. The definitive version of this paper was published in Thesis and is available at :},
}
```


# Install

This project is with `uv`. I recommend to use a virtual environment created by uv.

```
uv sync
```

To enter in uv shell, `uv shell`.


# Examples code

See `./examples` or `.demos/`


# Test

```bash
pytest tests/
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
