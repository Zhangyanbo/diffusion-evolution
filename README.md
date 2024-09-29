# Diffusion Evolution

Diffusion models are evolutionary algorithms.

![](./experiments/2d_models/two_peaks/images/framwork.jpg)

![](./experiments/2d_models/figures/process.png)


## Install

```bash
clone https://github.com/Zhangyanbo/diffusion-evolution
cd diffevo/
pip install .
```

## Quick Start

For simple experiments, parameter changes may not be necessary. In such cases, we can use the `DiffEvo` class to simplify the code.

```python
from diffevo import DiffEvo
import torch
from diffevo.examples import two_peak_density, two_peak_density_step

optimizer = DiffEvo(noise=0.1)

xT = torch.randn(512, 2)
x, population_trace, fitness_count = optimizer.optimize(two_peak_density, xT, trace=True)
```

Output:

```text
100%|██████████| 99/99 [00:00<00:00, 175.58it/s]
```

## Typical Usage

In most cases, tuning hyperparameters or adding custom operations is necessary to achieve higher performance. We recommend using the following form for the best balance between conciseness and versatility.

```python
from diffevo import DDIMScheduler, BayesianGenerator
from diffevo.examples import two_peak_density

scheduler = DDIMScheduler(num_step=100)

x = torch.randn(512, 2)

for t, alpha in scheduler:
    fitness = two_peak_density(x, std=0.25)
    generator = BayesianGenerator(x, fitness, alpha)
    x = generator(noise=0)
```

The following are two evolution trajectories of different fitness functions.

## Advanced Usage

We also offer multiple choices for each component to accommodate more advanced use cases:

* In addition to the `DDIMScheduler`, we provide the `DDIMSchedulerCosine`, which features a different $\alpha$ scheduler.
* We offer multiple fitness mapping functions that map the original fitness to a different value. These can be found in `diffevo.fitnessmapping`.
* Currently, we have only one version of the generator.

Below is an example of how to change the diffusion process and conduct advanced experiments:

```python
import torch
from diffevo import DDIMScheduler, BayesianGenerator, DDIMSchedulerCosine
from diffevo.examples import two_peak_density
from diffevo.fitnessmapping import Power, Energy, Identity

scheduler = DDIMSchedulerCosine(num_step=100) # use a different scheduler

x = torch.randn(512, 2)

trace = [] # store the trace of the population

mapping_fn = Power(3) # setup the power mapping function

for t, alpha in scheduler:
    fitness = two_peak_density(x, std=0.25)
    # apply the power mapping function
    generator = BayesianGenerator(x, mapping_fn(fitness), alpha)
    x = generator(noise=0.1)
    trace.append(x)

trace = torch.stack(trace)
```