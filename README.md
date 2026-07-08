# FMA-model

Official implementation of the Future Mutual Anticipation (FMA) model.

## Repository structure

```
simulation/    Core simulation code
analysis/      Criticality and Lévy analyses
figures/       Figure generation scripts
movies/        Example simulation movies
```
## Requirements

```bash
pip install -r requirements.txt
```

## Main simulation

```bash
python simulation/fma_simulation.py
```

## License

MIT License# FMA-model

Official implementation of the **Future Mode Anticipation (FMA)** model.

## Repository structure

```text
simulation/    Core simulation code
analysis/      Criticality and Lévy analyses
figures/       Figure generation scripts
movies/        Example simulation movies
```

## Requirements

```bash
pip install -r requirements.txt
```

## Running the simulation

Run the simulation:

```bash
python simulation/fma_simulation.py
```

The simulation output is saved as:

```text
outputs/rec.npz
```

## Creating an animation

Generate an animation from the simulation output:

```bash
python simulation/animation.py --input outputs/rec.npz --output outputs/animation.mp4
```

The resulting animation is saved as:

```text
outputs/animation.mp4
```

## Figures

Scripts for reproducing representative figures are provided in the `figures/` directory.

## Analysis

Scripts for criticality and Lévy analyses are provided in the `analysis/` directory.

## License

MIT License.
