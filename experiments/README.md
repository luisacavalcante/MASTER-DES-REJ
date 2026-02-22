# Estrutura inicial do mestrado (30 bases + reject option)

Este diretório propõe uma base para começar os experimentos comparativos com:

1. **Ensemble simples** (10 ou 20 modelos)
2. **Combinação majoritária**
3. **Plugin com seleção dinâmica + rejeição**

## Fluxo recomendado

1. Definir as 30 bases em `study_config.json`.
2. Rodar `python run_study.py --config study_config.json`.
3. Salvar resultados no formato longo (`dataset`, `metodo`, `taxa_rejeicao`, métricas).
4. Gerar tabela final com média/ranking por base.

## Como está organizado

- `study_config.json`: configuração da execução (datasets, seeds, métodos, rejeição).
- `run_study.py`: pipeline inicial para treinar, aplicar reject option e salvar resultados.

## Métricas sugeridas

Para cada método e base:

- `coverage`: fração de exemplos **não rejeitados**.
- `accuracy_accept`: acurácia apenas nos exemplos aceitos.
- `f1_accept`: F1 apenas nos exemplos aceitos.
- `reject_rate_observed`: taxa de rejeição observada.

> Dica: no artigo/dissertação, sempre reportar `coverage x performance` para mostrar o trade-off da rejeição.
