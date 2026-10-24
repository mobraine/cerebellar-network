# CerebellarNet — a neural network modeled on cerebellar cortex circuitry

A NumPy implementation of a neural network whose architecture follows the
neuronal structure of the cerebellar cortex, plus a set of classic
cerebellar learning experiments run as stimulus–feedback simulations.

## Architecture (biology → code)

| Cerebellar structure | Module | Role |
|---|---|---|
| Pontine / mossy-fiber preprocessing | `MossyLayer` | Gaussian receptive-field encoding of the input |
| Granule cells | `GranuleLayer` | Massive expansion recoding; each cell samples ~4 mossy fibers (sparse random connectivity); Golgi-cell feedback inhibition implemented as k-WTA sparsification (~4% active) |
| Parallel fibers | — | Granule-cell axons = the sparse high-dimensional code |
| Basket / stellate cells | `BasketLayer` | Feedforward lateral inhibition (population-mean subtraction) |
| Purkinje cells | `PurkinjeLayer` | The cortex's only output (inhibitory); plastic parallel-fiber synapses (LTD/LTP) |
| Climbing fibers | error signal | Olivary teaching signal gating synaptic updates, Marr–Albus–Ito style |
| Deep cerebellar nuclei | readout | Disinhibition readout: `y = tonic_baseline − purkinje_output` |

Learning options:
- `fit_local` — climbing-fiber-gated local LTD/LTP rule, **no backpropagation**
- `fit_ridge` — closed-form ridge readout on the fixed granule expansion (control)

## Experiments (`cerebellar_experiments.py`)

1. **Delay eyeblink conditioning** — CS (tone) + US (airpuff) training.
   The learned conditioned response peaks one step *before* US onset
   (timed response), via granule-cell spectral timing + climbing-fiber
   updates restricted to synapses active at US time.
2. **VOR gain adaptation** — retinal-slip error drives vestibulo-ocular
   reflex gain up (×1.5) or down (×0.5); a climbing-fiber "lesion"
   control abolishes adaptation, matching animal lesion studies.
3. **Structural ablation** — removing the mossy encoding collapses
   performance (RMSE 0.044 → 0.43); granule-cell count scales
   performance; sparsification helps pattern separation but not
   smooth function approximation under a global readout.

Also included:
- `embodied_arm.py` — two-link arm reaching with a cerebellar inverse
  model (Kawato feedback-error learning): payload adaptation and a
  washout aftereffect; frozen-cerebellum lesion control never adapts.
- `humanoid_balance.py` + `humanoid2d.xml` — a planar (sagittal) MuJoCo
  humanoid learning push recovery. Predictive feedback-error learning
  (state paired with feedback torque 120 ms ahead, importance-weighted)
  cuts COM excursion by ~37% at the trained force and raises the fall
  threshold: after training it survives 190 N pushes that topple the
  no-cerebellum control 3/3 times. (Full 3-D Gym humanoid collapses
  under naive PD — measured — hence the planar model, the standard
  simplification in biped-balance research.)

- `granule_scaling.py` — granule-cell scaling law (sparse CSR layer,
  1k → 1M cells). Function approximation improves 0.070 → 0.015 then
  saturates hard beyond ~100k cells: the ceiling is set by the fixed
  mossy front-end and readout, not by granule count. Classification on
  mean-separable Gaussian classes stays *below* a raw-input linear
  readout even at 1M cells — expansion pays off for highly overlapping /
  nonlinear class geometry, not everywhere. Cost grows ~linearly while
  gains flatten: the cerebellar scaling bargain in one plot.

## Quick start

```bash
python3 cerebellar_net.py          # architecture demos (pattern separation, function approximation)
python3 cerebellar_experiments.py  # conditioning / VOR / ablation experiments, writes PNG figures
```

Requires only NumPy (and Matplotlib for the experiment figures).

## Notes / honest simplifications

- Granule-layer weights mix excitatory/inhibitory signs (real granule
  cells are purely excitatory; the mixture summarizes net upstream effects).
- Climbing-fiber complex-spike timing windows are abstracted into an
  error-gated batch rule.
- Local LTD learning converges more slowly than gradient methods —
  consistent with biological trial counts.
