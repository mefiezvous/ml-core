# ml-core — CLAUDE.md

## Rôle

Layer Apache-2.0 partagé entre PUBLIC (`lerobot-playground-portfolio`) et PRIVATE (`my-robot-stack`).
Contient le code ML générique : wrappers de politique, boucle d'entraînement, évaluation.

## Règles d'import

- `ml-core` peut importer depuis `lerobot`, `mlflow`, `hydra`, `numpy`, `torch`, `loguru`.
- `ml-core` **ne doit pas** importer depuis `playground`, `my_robot_stack`, ni `robotics_platform_template`.
- PUBLIC et PRIVATE importent depuis `mlcore.*`.

## Convention de nommage des chemins (non-négociable)

Tous les chemins de sortie sont construits automatiquement depuis `robot_name` et `policy_type` :

| Artefact | Chemin |
|---|---|
| Checkpoints | `checkpoints/{robot_name}/{policy_type}/checkpoint_XXXXXXXX.ckpt` |
| MLflow runs | `mlruns/{robot_name}_{policy_type}/` |
| Rapports d'éval | `eval_reports/{robot_name}/{policy_type}/eval_report.json` |

`robot_name` et `policy_type` sont passés comme paramètres directs à `Trainer` et `Evaluator`
(pas dans la config Hydra — pour rester utilisable sans Hydra depuis PRIVATE).

## Conventions de code

- SPDX header dans tout `.py` : `# SPDX-FileCopyrightText: 2026 Arthur Mouraud` + `# SPDX-License-Identifier: Apache-2.0`
- Pas de `print()` — `from loguru import logger` uniquement
- `mypy --strict` — type hints partout
- TDD : tests écrits avant le code
- `@pytest.mark.unit` sur tous les tests
