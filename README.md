# Optimización de geometría — SenecaBot

Continuación del proyecto **SenecaBot**, un robot cuadrúpedo bioinspirado en la cabra que aprende a caminar
imitando la marcha de un ovino capturada con MOCAP (aprendizaje por imitación estilo DeepMimic con PPO sobre
[LocoMuJoCo](https://github.com/robfiras/loco-mujoco)).

Sobre el proyecto original este repositorio agrega:

- **Corrección de la marcha de referencia y de la recompensa**: la referencia hacía avanzar al robot a la
  velocidad de la oveja (1.40 m/s), unas 7 veces más de lo que sus patas pueden empujar sin deslizar, y el agente
  aprendía a dar "microsaltos". Se añadió una velocidad de referencia sin deslizamiento (0.218 m/s), una
  recompensa por patrón de contacto de las patas y penalizaciones de suavidad.
- **Corrección de un error de reinicio en MJWarp** que dejaba entornos atascados y distorsionaba las curvas de
  entrenamiento.
- **Dos agentes entrenados de 100 M pasos** (referencia a 0.218 m/s y a 1.40 m/s) sin microsaltos.
- **`geometry_opt/`**: pipeline iterativo de dinámica inversa → selección de actuadores MyActuator →
  dimensionamiento estructural de los eslabones como tubos huecos de PLA/PETG, con reporte y notebook de
  visualización.

## Estructura

| Carpeta | Contenido |
|---|---|
| [`seneca_loco/`](seneca_loco/) | Proyecto SenecaBot: modelo MuJoCo, pipeline MOCAP → trayectoria, entrenamiento, evaluación, agentes entrenados (`artifacts/`). Basado en [Robiolab/seneca_loco](https://github.com/Robiolab/seneca_loco). |
| [`loco-mujoco-seneca/`](loco-mujoco-seneca/) | Fork de LocoMuJoCo con el entorno `SenecaBot` y los cambios locales (`SENECA_LOCAL_CHANGES.md`). Basado en [andrademarique-cpu/loco-mujoco-seneca](https://github.com/andrademarique-cpu/loco-mujoco-seneca). |
| [`geometry_opt/`](geometry_opt/) | Pipeline de optimización de geometría y selección de motores ([README](geometry_opt/README.md), [reporte](geometry_opt/results_summary.md), [notebook](geometry_opt/visualize_results.ipynb)). |
| [`analisis_microsaltos/`](analisis_microsaltos/) | Scripts y figuras del diagnóstico de los microsaltos (antes/después). |
| [`my actuator/`](<my actuator/>) | Catálogo de actuadores MyActuator (series L, H y X) usado para la selección de motores. |
| `requirements.txt` | Versiones exactas del entorno con el que se obtuvieron los resultados. |

`seneca_loco/` y `loco-mujoco-seneca/` están incluidos como carpetas normales (sin su historial de git); el
historial y los autores originales están en los repositorios enlazados arriba. Los cambios hechos sobre ellos
están documentados en `seneca_loco/CLAUDE.md` (sección *Change Log*) y en
`loco-mujoco-seneca/SENECA_LOCAL_CHANGES.md`.

## Instalación

Probado en Linux con Python 3.12 y GPU NVIDIA (JAX + CUDA 12).

```bash
git clone <url-de-este-repositorio> "Optimizacion Geometria"
cd "Optimizacion Geometria"
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e loco-mujoco-seneca -e seneca_loco
export SENECA_HOME="$PWD/seneca_loco"      # raíz de seneca_loco (Hydra guarda aquí los entrenamientos)
```

Si tienes ROS instalado, ejecuta `unset PYTHONPATH` después de activar el entorno para que sus paquetes no se
mezclen con los del `.venv`.

## Uso

```bash
# Menú del proyecto SenecaBot (generar referencia, entrenar, evaluar, ...)
cd seneca_loco && python main.py

# Entrenar directamente (configuración en seneca_loco/simulation/training/conf.yaml)
python seneca_loco/simulation/training/train.py experiment.total_timesteps=100e6

# Pipeline de optimización de geometría (≈1.5 min, no necesita GPU con las marchas ya guardadas)
python -m geometry_opt.run_pipeline                # motor elegido por articulación
python -m geometry_opt.run_pipeline --symmetric    # mismo motor en la pata izquierda y derecha
```

Los resultados del pipeline quedan en `geometry_opt/results_summary.md` y se visualizan en
`geometry_opt/visualize_results.ipynb`.

## Resultados principales

| Política (100 M pasos) | Velocidad | Microsaltos | Coincidencia de contacto | Masa final (PLA) |
|---|---|---|---|---|
| Referencia original, recompensa original | 1.38 m/s | sí (hasta 3 toques por ciclo) | 61 % | — |
| Referencia sin deslizamiento (0.218 m/s) | 0.21 m/s | no (1 toque por ciclo) | 94 % | 8.77 kg |
| Referencia 1.40 m/s + nuevas recompensas | 1.39 m/s | no | 93 % | 9.72 kg |

Selección de motores y tubos (ver el reporte completo): las rodillas delanteras son las más exigentes (**X6-60**),
el resto usa **X4-36 / X8-32 / X4-10**; la velocidad articular, más que el torque, decide la mayoría de los
motores. Los eslabones quedan como tubos de **≈Ø15–19 mm con pared de 4 mm** (la flexión gobierna; el criterio
solo axial no dimensiona nada).

## Créditos

Este trabajo se construye sobre los siguientes proyectos, a cuyos autores corresponde el crédito de su parte:

- **SenecaBot / `seneca_loco`** — Nicolás Andrade Manrique, tesis de pregrado *"Model & Locomotion Dynamics of a
  Goat Bio-Inspired Quadruped Robot Through Reinforcement Learning"*, Departamento de Ingeniería Mecánica,
  Universidad de los Andes (junio 2026), asesor Jonathan Camargo Leyva, PhD.
  Repositorio: [Robiolab/seneca_loco](https://github.com/Robiolab/seneca_loco) — licencia MIT
  (© 2026 andrademarique-cpu, ver `seneca_loco/LICENSE`). Los parámetros iniciales del robot usados en
  `geometry_opt/thesis_params.py` provienen de esta tesis (Tabla V, Sec. 3.2.1, Tabla II).
- **`loco-mujoco-seneca`** — fork de LocoMuJoCo adaptado a SenecaBot por Nicolás Andrade Manrique:
  [andrademarique-cpu/loco-mujoco-seneca](https://github.com/andrademarique-cpu/loco-mujoco-seneca).
- **LocoMuJoCo** — Firas Al-Hafez, Guoping Zhao, Jan Peters y Davide Tateo:
  [robfiras/loco-mujoco](https://github.com/robfiras/loco-mujoco) y
  [robfiras/loco-mujoco-models](https://github.com/robfiras/loco-mujoco-models) — licencia MIT
  (© 2024 Al-Hafez, ver `loco-mujoco-seneca/LICENSE`). Si usas este trabajo, cita:

  ```bibtex
  @inproceedings{alhafez2023b,
    title={LocoMuJoCo: A Comprehensive Imitation Learning Benchmark for Locomotion},
    author={Firas Al-Hafez and Guoping Zhao and Jan Peters and Davide Tateo},
    booktitle={6th Robot Learning Workshop, NeurIPS},
    year={2023}
  }
  ```
- **Simulación y aprendizaje**: [MuJoCo, MJX y MuJoCo Warp](https://github.com/google-deepmind/mujoco)
  (Google DeepMind / NVIDIA), [NVIDIA Warp](https://github.com/NVIDIA/warp), [JAX](https://github.com/jax-ml/jax),
  [Flax](https://github.com/google/flax), [Optax](https://github.com/google-deepmind/optax),
  [Hydra](https://github.com/facebookresearch/hydra), [Weights & Biases](https://wandb.ai),
  [Optuna](https://github.com/optuna/optuna) y [ezc3d](https://github.com/pyomeca/ezc3d).
- **Datos técnicos**: catálogo de actuadores de [MyActuator](https://www.myactuator.com) (carpeta
  `my actuator/`) y hojas técnicas de Prusament PLA y PETG de Prusa Polymers (propiedades de los materiales en
  `geometry_opt/config.py`).

## Licencia

`seneca_loco/` y `loco-mujoco-seneca/` conservan sus licencias MIT originales (archivos `LICENSE` en cada
carpeta). El código nuevo de este repositorio (`geometry_opt/`, `analisis_microsaltos/`) no tiene todavía una
licencia definida.
