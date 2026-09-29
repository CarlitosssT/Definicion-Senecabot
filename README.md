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

### Instalación en Windows (WSL2)

En Windows nativo el entrenamiento no corre: JAX con CUDA y MJWarp solo existen para Linux, y varios paquetes
`nvidia-*-cu12` de `requirements.txt` no tienen versión para Windows. La solución es WSL2 (Ubuntu dentro de
Windows), con la que el repo corre sin cambios y con GPU. Requiere Windows 10 22H2 / Windows 11 y GPU NVIDIA.

**1. Instalar Ubuntu 24.04** — en PowerShell (Windows) como administrador:

```powershell
wsl --install -d Ubuntu-24.04
wsl --set-default Ubuntu-24.04
```

Usa **24.04** explícitamente: `wsl --install` sin argumentos instala la última versión de Ubuntu, cuyo Python
no coincide con el 3.12 que fija `requirements.txt`. Al terminar se crea un usuario y contraseña de Linux. Para
entrar después: `wsl -d Ubuntu-24.04` o "Ubuntu 24.04" en el menú de Windows Terminal.

> Los comandos `wsl ...` se escriben en PowerShell (`PS C:\...>`), no dentro de Ubuntu (`usuario@PC:~$`): dentro
> de Ubuntu `wsl` es otro programa sin relación.

**2. Paquetes base y comprobación de la GPU** — dentro de Ubuntu. Trabaja siempre en tu carpeta de Linux (`~`),
no en `/mnt/c/...` (el disco de Windows es mucho más lento desde WSL):

```bash
cd ~
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl build-essential python3-venv python3-dev gh
lsb_release -a        # debe decir 24.04
python3 --version     # debe decir 3.12.x
nvidia-smi            # debe mostrar la GPU
```

El driver NVIDIA se instala **solo en Windows** (no dentro de Ubuntu). Si `nvidia-smi` falla, actualiza el
driver en Windows y ejecuta `wsl --shutdown` en PowerShell.

**3. Clonar el repositorio:**

```bash
gh auth login         # GitHub.com -> HTTPS -> Login with a web browser
gh repo clone CarlitosssT/Definicion-Senecabot
cd ~/Definicion-Senecabot
```

**4. Entorno Python** (igual que en Linux):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt            # varios GB (paquetes CUDA), tarda unos minutos
pip install -e loco-mujoco-seneca -e seneca_loco
echo 'export SENECA_HOME="$HOME/Definicion-Senecabot/seneca_loco"' >> ~/.bashrc
echo 'export MUJOCO_GL=egl' >> ~/.bashrc   # render sin pantalla (videos de entrenamiento)
source ~/.bashrc
```

**5. Verificar:**

```bash
cd ~/Definicion-Senecabot && source .venv/bin/activate
python -c "import jax; print(jax.devices())"        # [CudaDevice(id=0)]
python -c "import mujoco; print(mujoco.__version__)" # 3.8.0
```

Si JAX muestra `CpuDevice`, la GPU no está disponible dentro de WSL (revisa el paso 2).

**6. Abrir el proyecto en VS Code (modo WSL):**

1. Instala VS Code **en Windows** y, dentro de él, la extensión **WSL** (`ms-vscode-remote.remote-wsl`).
2. Desde Ubuntu:
   ```bash
   cd ~/Definicion-Senecabot
   code .
   ```
3. Abajo a la izquierda debe aparecer **`WSL: Ubuntu-24.04`**. Si en cambio aparece el aviso *"The host
   'wsl.localhost' was not found in the list of allowed hosts"*, VS Code está abriendo la carpeta como Windows
   (sin la extensión WSL): cancela, instala la extensión y repite `code .`.
4. Instala la extensión **Python** (VS Code la ofrece "en WSL") y elige el intérprete:
   `Ctrl+Shift+P` → *Python: Select Interpreter* → `./.venv/bin/python`.

Así la terminal integrada, los notebooks y el debugger usan Ubuntu y el `.venv` con GPU.

**Opcional — Claude Code dentro de Ubuntu** (el de Windows no es visible desde WSL):

```bash
curl -fsSL https://claude.ai/install.sh | bash
source ~/.bashrc
cd ~/Definicion-Senecabot && claude
```

**GPU con poca VRAM (p. ej. 8 GB).** `conf.yaml` usa 2048 entornos en paralelo; si el entrenamiento falla con
`RESOURCE_EXHAUSTED` / `out of memory`, reduce los entornos desde la línea de comandos (sin editar archivos):

```bash
python seneca_loco/simulation/training/train.py experiment.num_envs=1024 experiment.env_params.nconmax=13000
```

Para una prueba rápida sin cuenta de wandb:

```bash
WANDB_MODE=disabled python seneca_loco/simulation/training/train.py \
  experiment.total_timesteps=4e6 experiment.num_envs=1024 \
  experiment.env_params.nconmax=13000 experiment.validation.num=2
```

(`validation.num` debe ser ≤ número de actualizaciones = `total_timesteps / num_steps / num_envs`.) Entrena
con el portátil conectado a la corriente. Para usar wandb: crea una cuenta en wandb.ai y ejecuta `wandb login`
una vez.

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
