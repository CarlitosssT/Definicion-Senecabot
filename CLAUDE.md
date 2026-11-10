# CLAUDE.md — Definicion-Senecabot (raíz del repositorio)

Contexto para Claude Code (y para personas) que trabajan en este repo. Resume el proyecto completo y el trabajo
hecho desde sep. 2026. El detalle de cada cambio vive en los dos change logs que se citan abajo; este archivo es el
mapa. Última actualización: 2026-11-09.

---

## 1. Qué es el proyecto

**SenecaBot** es un cuadrúpedo bioinspirado en la cabra que aprende a caminar imitando la marcha de una oveja
capturada con MOCAP (imitación estilo DeepMimic, PPO en JAX sobre LocoMuJoCo con MJX/MJWarp en GPU).

- La base (`seneca_loco/` y el fork `loco-mujoco-seneca/`) es la tesis de pregrado de Nicolás Andrade Manrique
  (Uniandes, jun. 2026), terminada en junio. Su mejor agente: `seneca_loco/artifacts/trained_agents/curated/4_best_try/17-19-49`.
- **Este repo es la continuación de Carlos** (usuario actual, `ca.tovar2@uniandes.edu.co`), desde 2026-09-26:
  1. Corregir la referencia y la recompensa (microsaltos, velocidad imposible).
  2. Arreglar un bug de reinicio de MJWarp.
  3. Entrenar agentes de 100 M pasos sin microsaltos (0.218 m/s y 1.40 m/s).
  4. Pruebas y fine-tune en pendientes.
  5. `geometry_opt/`: dinámica inversa → selección de motores MyActuator → dimensionamiento de tubos PLA/PETG.
  6. **Rediseño de la pata de RRR a RR** (sin cadera; rodilla y tobillo movidos desde el torso por dos mecanismos
     de barras), modelo MuJoCo generado, entorno nuevo y agentes RR entrenados. **Este es el frente activo.**

El usuario escribe en español; los commits y la documentación del usuario están en español, los change logs técnicos
(`seneca_loco/CLAUDE.md`, `SENECA_LOCAL_CHANGES.md`, `geometry_opt/README.md`) en inglés.

---

## 2. Estructura

| Carpeta | Contenido |
|---|---|
| `seneca_loco/` | Proyecto SenecaBot: XML MuJoCo, pipeline MOCAP → referencia, entrenamiento (Hydra), evaluación, agentes (`artifacts/`). **Tiene su propio `CLAUDE.md` con arquitectura y Change Log: léelo antes de tocar entrenamiento.** |
| `loco-mujoco-seneca/` | Fork de LocoMuJoCo con los entornos `SenecaBot` y `SenecaBotRR`. Todo cambio local va marcado `[SENECA LOCAL CHANGE]` en el código y registrado en `SENECA_LOCAL_CHANGES.md` (más nuevo arriba). Tiene su propio `CLAUDE.md` (genérico de LocoMuJoCo). |
| `geometry_opt/` | Pipeline de motores + estructura + mecanismo RR (`linkage.py`) + generador del XML RR (`make_rr_xml.py`). Ver `geometry_opt/README.md`, `results_summary.md`. |
| `analisis_rr/` | Análisis de los agentes RR: `patas_rr.ipynb` (gráficas, GIFs, figura de geometría 2D), rollouts `.npz`, copia del XML de la geometría anterior (`senecabot_loco_rr_10-10-34.xml`). |
| `analisis_microsaltos/` | Diagnóstico antes/después de los microsaltos de los agentes RRR (alturas de patas, deslizamiento). |
| `diagnostico_microsaltos/` | `hop_diagnosis.py`: microsaltos traseros de v140 (vienen de la referencia, ver §6). |
| `pruebas_pendiente/` | Pruebas en pendientes (`slope_test.py`, `gait_by_slope.py`, `render_slopes.py`, `visualize_slopes.ipynb`, JSON de resultados, videos). |
| `my actuator/` | Catálogo MyActuator series L/H/X (CSV + xlsx) para la selección de motores. |
| `requirements.txt` | Versiones exactas del entorno (Python 3.12, CUDA 12, mujoco 3.8.0). |

`seneca_loco/` y `loco-mujoco-seneca/` son carpetas normales (sin su historial git original). `seneca_loco/artifacts/`
**sí está versionado** en este repo (agentes `.pkl`, figuras; ~300 MB). `/tesis nicolas.pdf` y `/reporte_latex/`
están en `.gitignore`.

---

## 3. Entorno

- WSL2 Ubuntu 24.04, GPU NVIDIA, **`.venv` en la raíz** (no conda). Paquetes editables: `pip install -e loco-mujoco-seneca -e seneca_loco`.
- `export SENECA_HOME="$HOME/Definicion-Senecabot/seneca_loco"` y `export MUJOCO_GL=egl` (en `~/.bashrc`).
- Los `hydra.run.dir` de los yaml apuntan a `${oc.env:SENECA_HOME,/home/andre/Definicion-Senecabot/seneca_loco}`.
- `seneca_loco/CLAUDE.md` todavía menciona el entorno de Nicolás (conda `workspace`, `/home/ndam/...`, RTX 5070):
  **eso está desactualizado**, aquí se usa el `.venv`.
- Con poca VRAM: `experiment.num_envs=1024 experiment.env_params.nconmax=13000`. Sin wandb: `WANDB_MODE=disabled`.
- Si el usuario tiene ROS: `unset PYTHONPATH` tras activar el venv.

---

## 4. Acoplamiento entre repos (crítico)

Muchas funciones están repartidas entre `seneca_loco` y el fork: términos de `MimicReward` (`rootvel_*`, `contact_*`,
`action_rate_w`, `action_jerk_w`, `qacc_w`, `joint_limit_w`, `contact_ref_frac` por pata), logging en vivo a wandb,
checkpoint del mejor agente, reinicio de MJWarp, aleatorizador de pendiente, `SlopeLocomotionReward`,
`UprightTerminalStateHandler`, entorno `SenecaBotRR`. Antes de cambiar comportamiento de entrenamiento revisa
`seneca_loco/simulation/training/train.py` **y** el fork. **Convención de documentación** (respétala):

- Cambio en el fork → comentario `[SENECA LOCAL CHANGE]` + entrada fechada en `loco-mujoco-seneca/SENECA_LOCAL_CHANGES.md`.
- Cambio en `seneca_loco` → entrada fechada en el *Change Log* de `seneca_loco/CLAUDE.md` (más nuevo arriba).
- Cambio en `geometry_opt` → actualizar `geometry_opt/README.md`.
- Términos nuevos de recompensa con default que deja el comportamiento anterior intacto (p. ej. `= 0`).

---

## 5. Cronología del trabajo (sep.–nov. 2026)

Detalle completo en `seneca_loco/CLAUDE.md` (Change Log) y `SENECA_LOCAL_CHANGES.md`.

- **09-26** Caminata scripted arreglada (columnas `*_knee_angle`).
- **09-27** Diagnóstico de microsaltos: la referencia avanzaba a 1.40 m/s (oveja) pero las patas del robot solo
  empujan ~0.2 m/s sin deslizar. Se añadieron `contact_w_sum`, `action_rate_w`, `action_jerk_w`; velocidad de
  referencia sin deslizamiento (`cyclic.no_slip_root_speed`, `+root_speed=no_slip|mocap|<m/s>`); bug de reinicio
  MJWarp arreglado (`mujoco_mjx.py` restaura todo el `Data` por entorno; antes un entorno con NaN terminaba en cada
  paso y hundía la curva de longitud de episodio). Dos agentes de 100 M:
  - **v0218** `artifacts/trained_agents/2026-09-27/07-12-14` (ref. `trajectory_adapted.noslip_0218.npz`), 0.21 m/s, 94 % coincidencia de contacto.
  - **v140** `artifacts/trained_agents/2026-09-27/18-22-14` (ref. `trajectory_adapted.preRootSpeed.bak.npz`, `rootvel_w_exp=4.0`), 1.39 m/s, 93 %.
  - `trajectory_adapted.npz` activo = referencia de 1.40 m/s.
- **09-27** `geometry_opt` completo: masa final 8.77 kg (v0218) / 9.72 kg (v140) en PLA; rodillas delanteras X6-60,
  resto X4-36 / X8-32 / X4-10; tubos ≈Ø15–19 mm, pared 4 mm (gobierna la flexión). Usa dinámica inversa sobre la
  marcha grabada, no re-simulación (las políticas se degradan en robots más pesados).
- **09-29** Pendientes: v140 cae desde +25° de subida y en −35°. Fine-tune por imitación (`conf_finetune_slope.yaml`)
  y luego con recompensa de tarea (`conf_finetune_slope_task.yaml`) → `artifacts/trained_agents/2026-09-29/18-49-47`:
  sin caídas de −35° a +30°, velocidad 1.33–1.46 m/s en todas las pendientes. Usar `PPOJax_final.pkl` (el `saved` es
  casi aleatorio con validación de 1 s). `finetune.py` mapea pesos por nombre de observación.
- **10-06** Microsaltos traseros de v140: vienen de la referencia (apoyo trasero con doble bajada), no del agente.
- **11-03 → 11-04** **Modelo RR** (ver §7). Mejor/elegido: **`artifacts/trained_agents_rr/2026-11-04/17-37-28/PPOJax_final.pkl`**.

---

## 6. Agentes y referencias de referencia rápida

| Nombre | Modelo | Ruta (`seneca_loco/artifacts/...`) | Referencia | Notas |
|---|---|---|---|---|
| Tesis (Nicolás) | RRR | `trained_agents/curated/4_best_try/17-19-49/PPOJax_saved.pkl` | original | speed_ratio 0.985, con microsaltos |
| v0218 | RRR | `trained_agents/2026-09-27/07-12-14/PPOJax_saved.pkl` | `trajectory_adapted.noslip_0218.npz` | 0.21 m/s |
| v140 | RRR | `trained_agents/2026-09-27/18-22-14/PPOJax_saved.pkl` | `trajectory_adapted.preRootSpeed.bak.npz` | 1.39 m/s; base de geometry_opt y pendientes |
| pendiente (tarea) | RRR | `trained_agents/2026-09-29/18-49-47/PPOJax_final.pkl` | v140 | ±30° sin caídas |
| **RR elegido** | RR | `trained_agents_rr/2026-11-04/17-37-28/PPOJax_final.pkl` | `trajectory_adapted_rr_1p40.npz` | 1.156 m/s (82 %), 0 caídas, RMS rodilla/tobillo 9.6° |

Referencias en `seneca_loco/data/processed_data/`; los `*.bak.npz` son versiones anteriores conservadas a propósito
(no borrar: agentes viejos se comparan contra ellas). `trajectory_adapted_rr*.v2_geometry_10-10-34.bak.npz` =
geometría RR anterior.

---

## 7. Frente activo: pata RR

**Idea.** Se elimina el muslo (articulación y motor de cadera). Tibia y metatarso conservan posición y longitud.
Cada pata tiene dos manivelas en el torso: RR 1 (`l1`, `l2`) mueve la tibia, RR 2 (`l3`, `l4`) el metatarso.
Rodilla y tobillo son pasivos, cerrados por igualdades `connect`. 8 actuadores (`<pata>_rr1`, `<pata>_rr2`).

**Pipeline RR:**

```bash
source .venv/bin/activate                                   # desde la raíz
python -m geometry_opt.run_pipeline --linkage               # dimensiona el mecanismo (FR y BR; FL/BL espejo) + motores
python -m geometry_opt.make_rr_xml [--policy v140] [--material PLA]   # genera senecabot_loco_rr.xml (¡no editar a mano!)
cd seneca_loco
python -m simulation.mocap.trajectory.trajectory_generation_rr        # ref. sin deslizamiento -> trajectory_adapted_rr.npz
python -m simulation.mocap.trajectory.trajectory_generation_rr --root-speed keep \
       --out data/processed_data/trajectory_adapted_rr_1p40.npz       # ref. 1.40 m/s (la de interés)
python simulation/training/train.py --config-name conf_rr \
       reference=data/processed_data/trajectory_adapted_rr_1p40.npz \
       experiment.env_params.reward_params.rootvel_w_exp=4.0          # 30M desde cero
python simulation/training/train.py --config-name conf_rr_finetune finetune.init_agent=<pkl>   # fine-tune
python simulation/training/eval_rr.py --path <pkl> [--n_envs 32] [--n_steps 1000] [--video] [--reference <npz>]
```

**Piezas:**
- `geometry_opt/linkage.py`: evolución diferencial por cada `(f1, f2)` de `LINKAGE_F_VALUES`; minimiza `l4`, longitudes,
  offsets `dx, dy` y amplificación de torque; exige 100 % del rango de movimiento de la marcha y holgura
  `LINKAGE_CLEARANCE_M` entre barras. `a3` se optimiza (±45° fijo no tiene solución). Todo en `LINKAGE_*` de `config.py`.
- `geometry_opt/make_rr_xml.py`: monta todas las patas dentro del torso (`PIVOT_INSET` 2 mm); pivotes traseros más
  altos (`HIND_EXTRA_RAISE`) para que el tronco vaya nivelado; patas traseras 2.7 cm adelante. Masa total 6.06 kg.
- `trajectory_generation_rr.py`: rodilla RR = cabeceo absoluto de la tibia respecto al torso en la referencia RRR;
  barras por IK; pies en apoyo a 0.027 m; `flatten_double_dip` quita el rebote de la doble bajada trasera
  (`--keep-hind-double-dip` lo conserva).
- Fork: `environments/quadrupeds/senecabot_rr.py` (`SenecaBotRR`: obs = raíz + rodilla/tobillo; acciones = manivelas).
- `conf_rr.yaml`: `joints_for_mimic` = raíz + rodillas + tobillos; `qpos_w_sum 1.1`, `qvel_w_sum 0.65`;
  `action_rate_w/jerk_w 0.1` (desde cero); `joint_torque_coeff 0.0033`; `joint_limit_w 0.5`;
  **terminación temprana `root_height_margin: 0.06`** (imprescindible).
- El 17-37-28 añadió seguimiento de pies (`sites_for_mimic` = base + 4 pies, `rpos_w_sum 0.5 / rpos_w_exp 400`,
  `rvel_w_sum 0.15`), `rootvel_w_exp 20`, y suavidad `action_rate/jerk 0.5`, `qacc_w 0.1` (vía su cadena de fine-tunes).

**Estado del agente RR elegido (17-37-28):** 0 caídas en 32×1000 pasos, 1.156 m/s, ~1.67 toques/s en cada pata,
RMS 9.6°. Problema pendiente: **las patas traseras rebotan ~2 cm tras aterrizar** (pico 34–37 mm, justo bajo el umbral
de contacto de 37 mm); el tronco va 3.5 cm bajo la referencia. Probado sin éxito: subir el tronco
(`root_height_margin 0.03`, agente 18-32-41, peor). Candidato restante: el contacto de pie muy blando
(`solimp 0.015 1 0.022`), **pero el usuario decidió no cambiarlo** (se mantiene como en el modelo original).

---

## 8. Lecciones aprendidas (no repetir los errores)

- **Recompensa recortada a 0**: si los términos positivos máximos quedan por debajo de las penalizaciones de suavidad
  con el std inicial (0.3), la recompensa se recorta a 0 y la entropía explota. Al cambiar pesos, compara recompensa
  máxima vs. penalizaciones al std inicial. Por eso RR desde cero usa `action_rate/jerk 0.1` y sube a 0.5 en el fine-tune.
- **Agacharse no es límite de torque**: los agentes RR caminaban con el tronco a 0.25 m (ref. 0.37) por falta de
  terminación (el margen por defecto 0.3 m nunca termina episodios). Lo arregló `root_height_margin: 0.06`, no pesos
  más altos de `qpos`. Torques estáticos de la referencia: 2–32 N·m vs. límites 32–120 N·m.
- **El balance se aprende**: reproducir la referencia en lazo abierto con PD cae de lado en ~0.7 s tanto en RR como en RRR.
- **Microsaltos que vienen de la referencia**: el apoyo trasero tenía dos mínimos de altura → la etiqueta de contacto
  (`contact_ref_frac 0.3`) premiaba dos apoyos por ciclo. Arreglar la referencia (o `contact_ref_frac` por pata)
  antes de añadir penalizaciones anti-salto, que pelearían contra la imitación.
- **Los pies delanteros de la referencia RRR flotan ~6 cm** (nunca tocan): la referencia es cinemática, no resuelta en contacto.
- **Velocidad**: la referencia a 1.40 m/s es ~7× la velocidad sin deslizamiento; `rootvel_w_exp` debe ajustarse a la
  velocidad de la referencia (100 para 0.218 m/s, 4 para 1.40 m/s, 20 en el fine-tune RR).
- `action_rate_coeff` y `joint_acc_coeff` heredados se aplican al cuadrado / sobre todos los dof → casi no-op; usar
  los términos nuevos (`action_rate_w`, `action_jerk_w`, `qacc_w`).
- `Mean Episode Length` en entrenamiento no mide supervivencia; evaluar con `eval_rr.py` / `debug_agent.py` /
  `agent_diagnostics.py`. MJWarp no es determinista entre corridas.
- Las políticas RRR no se transfieren al modelo RR (otro espacio de acción). Cada cambio de geometría en
  `make_rr_xml.py` invalida los agentes RR anteriores y obliga a regenerar las referencias RR.
- Notebooks abiertos en Jupyter sobrescriben ediciones hechas en disco: cerrarlos/recargarlos antes.

---

## 9. Pendientes conocidos / próximos pasos posibles

- Rebote trasero de ~2 cm en el agente RR (ver §7); velocidad al 82 % de la referencia; tobillos delanteros con RMS alto.
- `geometry_opt` con el mecanismo **no cierra el lazo dinámico**: los motores no se mueven al torso ni se suma la masa
  de las barras; los torques usan la marcha RRR (`LINKAGE_XML = XML_PATH`). Ya existe un agente RR, así que se podría
  grabar su marcha y pasar `LINKAGE_XML` a `XML_RR_PATH`.
- FL/BL se copian de FR/BR (espejo), no se dimensionan.
- Todo el frente RR (código, referencias, agentes, `analisis_rr/`, grabaciones) está commiteado en `main` desde el
  2026-11-09. El usuario commitea directamente en `main` e incluye agentes y grabaciones; haz commit/push solo si lo pide.
- `__pycache__/` está versionado por accidente en algunas carpetas; no añadir más.

---

## 10. Comandos frecuentes (RRR)

```bash
cd seneca_loco && python main.py                             # menú interactivo
python simulation/training/train.py [experiment.total_timesteps=100e6]
python simulation/training/eval.py --path <pkl> [--use_mujoco] [--camera follow]
python simulation/analysis/debug_agent.py --path <pkl> [--video]
python simulation/analysis/agent_diagnostics.py --path <pkl>
cd .. && python -m geometry_opt.run_pipeline [--models v140] [--materials PLA] [--symmetric] [--no-cache]
```

Tests solo en el fork: `cd loco-mujoco-seneca && pytest --ignore=tests/test_task_factories.py`.
