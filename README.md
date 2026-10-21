# PSP-8---Telco-Customer-Churn
# Análise Tradicional × Inteligência Artificial na Retenção de Clientes

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SEU-USUARIO/SEU-REPOSITORIO/blob/main/churn_tradicional_vs_IA.ipynb)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/status-MVP-green)

Mecanismos de inteligência (aprendizado de máquina) melhoram as análises e as respostas de uma análise tradicional (estatística descritiva + regras de negócio)?

Este projeto responde a essa pergunta com um experimento controlado. A mesma pergunta de negócio é respondida **das duas formas**, com os mesmos dados de teste e as mesmas métricas, formando **pares de comparação**. O experimento cobre os quatro níveis de análise (descritiva, diagnóstica, preditiva e prescritiva) e termina com um **plano de ação exportável** para o time comercial.

---

## Sumário
- [Problema de negócio](#problema-de-negócio)
- [Base de dados](#base-de-dados)
- [Metodologia: os 5 pares](#metodologia-os-5-pares)
- [Níveis de análise cobertos](#níveis-de-análise-cobertos)
- [Como executar](#como-executar)
- [Estrutura do notebook](#estrutura-do-notebook)
- [Resultados](#resultados)
- [Premissas ajustáveis](#premissas-ajustáveis)
- [Limitações e próximos passos](#limitações-e-próximos-passos)
- [Tecnologias](#tecnologias)
- [Autoria](#autoria)

---

## Problema de negócio

Uma operadora de telecomunicações perde clientes todo mês (*churn*). O gestor precisa de quatro respostas:

1. **Quem** vai cancelar?
2. **Por que** os clientes cancelam?
3. **Quem priorizar** com um orçamento limitado de retenção?
4. **Qual ação** oferecer a cada cliente, e quanto isso deve render?

## Base de dados

**Telco Customer Churn**, publicada no Kaggle ([blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)).

| Item | Descrição |
|---|---|
| Registros | 7.043 clientes |
| Variáveis | 20 explicativas + 1 alvo (`Churn`) |
| Conteúdo | Perfil demográfico, serviços contratados, tipo de contrato, forma de pagamento, mensalidade e gasto total |
| Alvo | `Churn` = cancelou (Yes) ou permaneceu (No) |

## Metodologia: os 5 pares

| Par | Tipo | Pergunta | Abordagem tradicional | Mecanismo de inteligência | Critério de comparação |
|---|---|---|---|---|---|
| 1 | Preditiva | Quem vai cancelar? | Regra de pontuação de risco montada a partir de tabelas cruzadas | Regressão Logística, Random Forest e Gradient Boosting | Acurácia, precisão, recall, F1, AUC, **teste de McNemar** e IC 95% por bootstrap |
| 2 | Preditiva → decisão | Quem priorizar com orçamento limitado? | Ordenação pela pontuação de risco | Ordenação pela probabilidade prevista | Lift, % de churners no top 20% e resultado financeiro |
| 3 | Diagnóstica | Quais fatores explicam o churn? | Amplitude da taxa de churn por variável | Importância por permutação | Comparação dos rankings |
| 4 | Descritiva | Como segmentar a base? | Faixas manuais de tempo de casa | Clusterização K-Means | Separação de risco entre segmentos |
| 5 | **Prescritiva** | Qual ação oferecer a cada cliente? | Oferta única (contrato anual) para todos os clientes de risco | Simulação "e se?" com o modelo + otimização da ação por cliente sob orçamento | Ganho líquido esperado (R$), clientes retidos e ROI |

**Garantia de comparação justa:** os dados são divididos em treino (75%) e teste (25%) **antes de qualquer análise**. As regras tradicionais, os pontos de corte, os modelos e os limiares são construídos apenas com o treino. Ambas as abordagens são avaliadas no teste, que nenhuma delas viu.

## Níveis de análise cobertos

| Nível | Pergunta | Onde está |
|---|---|---|
| Descritiva | O que aconteceu? | Etapa 4 e Par 4 |
| Diagnóstica | Por que aconteceu? | Par 3 |
| Preditiva | O que vai acontecer? | Pares 1 e 2 |
| Prescritiva | O que devo fazer? | Par 5 + plano de ação exportável |

## Como executar

### Opção 1: Google Colab (recomendado)
1. Clique no botão **Open in Colab** no topo desta página.
2. Vá em **Ambiente de execução → Executar tudo**.
3. A base é baixada automaticamente pelo `kagglehub`. Se o download pedir login:
   - baixe o arquivo em [kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn);
   - descompacte o ZIP;
   - rode a célula de upload e envie o arquivo `WA_Fn-UseC_-Telco-Customer-Churn.csv`.

### Opção 2: localmente
```bash
git clone https://github.com/SEU-USUARIO/SEU-REPOSITORIO.git
cd SEU-REPOSITORIO
pip install pandas numpy scikit-learn matplotlib seaborn scipy kagglehub jupyter
jupyter notebook churn_tradicional_vs_IA.ipynb
```

O tempo de execução completo é de cerca de 3 a 5 minutos. Todas as linhas de código estão comentadas em português.

## Estrutura do notebook

```
1. Preparação do ambiente
2. Download da base (Kaggle)
3. Limpeza dos dados + divisão treino/teste
4. Análise tradicional            [DESCRITIVA]
5. Treinamento dos modelos de ML
   PAR 1 – Quem vai cancelar?     [PREDITIVA]
   PAR 2 – Quem priorizar?        [PREDITIVA → DECISÃO]
   PAR 3 – Por que cancelam?      [DIAGNÓSTICA]
   PAR 4 – Segmentação            [DESCRITIVA]
   PAR 5 – Qual ação oferecer?    [PRESCRITIVA]
6. Placar final + resumo executivo automático
7. Exportação do plano de ação (CSV)
```

**Arquivos do repositório**

| Arquivo | Descrição |
|---|---|
| `churn_tradicional_vs_IA.ipynb` | Notebook completo |
| `plano_de_acao_retencao.csv` | Gerado ao rodar o notebook: clientes priorizados, risco, ação recomendada e ganho esperado |
| `README.md` | Este documento |

## Resultados

Resultados obtidos na base real (7.043 clientes; teste com 1.761 clientes), com as premissas padrão do notebook. O melhor mecanismo de inteligência no Par 1 foi o **Gradient Boosting**, que é o modelo usado no placar e nos Pares 3 e 5.

### Placar final

| Indicador | Tradicional | Inteligência | Ganho relativo | Vencedor |
|---|---|---|---|---|
| F1 (quem vai cancelar) | 0,615 | 0,630 | +2,5% | Inteligência |
| AUC (capacidade de ordenar risco) | 0,820 | 0,843 | +2,8% | Inteligência |
| % de churners no top 20% | 46,7% | 48,2% | +3,2% | Inteligência |
| Separação de risco entre segmentos | 0,167 | 0,221 | +32,4% | Inteligência |
| Ganho líquido esperado do plano (R$) | R$ 25.727 | R$ 33.341 | +29,6% | Inteligência |

**Constatação:** a abordagem com inteligência superou a tradicional em **5 de 5 indicadores**. Os ganhos são **modestos na previsão** (+2% a +3%) e **expressivos na segmentação e na prescrição** (+30%).

### Achados por nível de análise

**Descritiva — o que aconteceu?**
- A taxa geral de churn é de **26,5%**.
- As categorias de maior risco são pagamento por cheque eletrônico (45,4% de churn), contrato mensal (42,7%) e internet por fibra (42,0%).

**Diagnóstica — por que aconteceu?**
- Os dois métodos concordam nos fatores principais: **tipo de contrato** e **tempo de casa** ocupam as duas primeiras posições nos dois rankings.
- O modelo **reposiciona** variáveis que a análise uma-variável-por-vez distorce:
  - **Gasto total** e **mensalidade** sobem da 9ª e 12ª posições para a 3ª e 5ª.
  - **Suporte técnico, backup e proteção de dispositivo** caem da 5ª–7ª para a 10ª–16ª. Na tabela cruzada elas parecem fortes, mas boa parte do efeito já está explicada pelo tipo de internet e de contrato.

**Preditiva — o que vai acontecer?**

| Abordagem | Acurácia | Precisão | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Tradicional (regra) | 0,738 | 0,503 | **0,790** | 0,615 | 0,820 |
| Regressão Logística | 0,766 | 0,543 | 0,737 | 0,625 | **0,846** |
| Random Forest | 0,761 | 0,534 | 0,769 | **0,630** | 0,842 |
| Gradient Boosting | **0,776** | **0,560** | 0,722 | **0,630** | 0,843 |

- O **teste de McNemar** mostra que os três modelos acertam significativamente mais clientes que a regra (p = 0,0004; 0,0057; < 0,0001).
- O **ganho de F1** do Gradient Boosting foi de +0,015, com IC 95% de −0,009 a +0,038. Como o intervalo inclui zero, esse ganho específico **não é estatisticamente confirmado**.
- A regra tradicional tem **recall maior** (captura 79% dos churners contra 72%), mas gera **mais alarmes falsos** (precisão de 50% contra 56%). Os modelos erram menos em quem é sinalizado.
- Na priorização (Par 2), a **Regressão Logística** foi a melhor: colocou 237 churners no top 20% contra 218 da regra (lift de 2,54x contra 2,34x), com resultado simulado de R$ 59.072 contra R$ 51.876 (+13,9%).

**Descritiva — segmentação (Par 4)**
- O K-Means separou o risco melhor que as faixas de tempo de casa (0,221 contra 0,167; silhueta = 0,204).
- Ele encontrou um segmento crítico que nenhuma faixa de uma variável só isola: **56,7% de churn**, 95% com contrato mensal, 99% com fibra, 15 meses de casa em média e mensalidade de R$ 84.
- Também encontrou dois segmentos de baixo risco com perfis opostos (7,8% e 12,2% de churn): um de clientes de baixo gasto sem internet e outro de clientes antigos e de alto valor.

**Prescritiva — o que devo fazer? (Par 5)**

| Plano (orçamento de R$ 15.000) | Clientes abordados | Clientes retidos (esperado) | Ganho líquido esperado | ROI |
|---|---|---|---|---|
| Tradicional (oferta única: contrato anual) | 250 | 43,1 | R$ 25.727 | 171,5% |
| Inteligência (ação otimizada por cliente) | 304 | 49,7 | **R$ 33.341** | **222,3%** |

- Com o mesmo orçamento, o plano otimizado aborda mais clientes, retém mais e rende **R$ 7.614 a mais**.
- A ação mais recomendada no topo da lista é o **débito automático**, que custa pouco e reduz bastante o risco de quem paga por cheque eletrônico.
- O plano inteligente vence em **todos os orçamentos testados**:

| Orçamento | Tradicional | Inteligência |
|---|---|---|
| R$ 5.000 | R$ 10.845 | R$ 13.687 |
| R$ 10.000 | R$ 20.173 | R$ 24.624 |
| R$ 15.000 | R$ 25.727 | R$ 33.341 |
| R$ 25.000 | R$ 35.338 | R$ 42.821 |
| R$ 40.000 | R$ 39.218 | R$ 45.654 |

O entregável final é o arquivo `plano_de_acao_retencao.csv`, com **304 clientes** priorizados, a ação recomendada para cada um e o ganho esperado.

### Conclusão

Os mecanismos de inteligência **melhoram as análises e as respostas** da análise tradicional, mas o tamanho do ganho depende da pergunta:
- **Na previsão de quem vai cancelar**, o ganho existe e é estatisticamente significativo em acertos, mas é pequeno. Uma regra bem construída a partir da análise descritiva já chega perto dos modelos.
- **Na segmentação, no diagnóstico e na prescrição**, a vantagem é grande. A IA combina várias variáveis ao mesmo tempo, encontra perfis de risco que tabelas cruzadas não mostram e escolhe a ação certa para cada cliente, em vez de uma oferta única.

A análise tradicional continua essencial: foi ela que gerou as hipóteses, os fatores da regra e o cardápio de ações. A IA não substitui a análise tradicional; ela a **complementa e potencializa**.

## Premissas ajustáveis

Os valores financeiros são **ilustrativos** e devem ser ajustados para cada caso real. Todos estão no início das células dos Pares 2 e 5.

| Parâmetro | Valor padrão | Significado |
|---|---|---|
| `PCT_CAMPANHA` | 20% | Fração da base que a campanha consegue abordar (Par 2) |
| `CUSTO_CONTATO` | R$ 30 | Custo de abordar um cliente (Par 2) |
| `TAXA_SUCESSO` | 30% | % dos churners abordados que são retidos (Par 2) |
| `MESES_RETIDOS` | 12 | Horizonte de receita preservada |
| `ORCAMENTO` | R$ 15.000 | Verba da campanha prescritiva (Par 5) |
| `TAXA_ADESAO` | 50% | Chance de o cliente aceitar a oferta (Par 5) |
| Custo das ações | R$ 60 / R$ 15 / R$ 35 | Contrato anual / débito automático / suporte grátis |

## Limitações e próximos passos

**Limitações**
- O experimento usa uma única base, de um único setor.
- As premissas financeiras são hipotéticas; os valores em R$ servem para comparar as abordagens, não como previsão de receita.
- Os modelos foram treinados sem ajuste fino de hiperparâmetros.
- A simulação prescritiva estima **associações**, não causas. O modelo indica o efeito provável de cada ação, mas não o comprova.

**Próximos passos**
- Validar o efeito real das ações com um **teste A/B** (grupo tratado × grupo de controle).
- Usar modelos de *uplift* para estimar o efeito causal de cada ação por cliente.
- Ajustar os hiperparâmetros (`GridSearchCV` / `Optuna`) e testar XGBoost ou LightGBM.
- Explicar as previsões cliente a cliente com SHAP.
- Publicar o plano de ação num painel interativo (Streamlit ou Power BI).

## Tecnologias

- **Python 3**
- **pandas** e **NumPy**: manipulação de dados
- **Matplotlib** e **Seaborn**: visualização
- **scikit-learn**: modelos, pipelines, validação cruzada, métricas, importância por permutação e K-Means
- **SciPy**: teste de McNemar
- **kagglehub**: download da base
- **Google Colab**: ambiente de execução

## Autoria

| | |
|---|---|
| **Aluna** | Isadora Messenberg Guimarães Macêdo Rebello |
| **Matrícula** | 231013387 |
| **Disciplina** | Projeto de Sistema de Produção 8 (PSP 8) |
| **Professor** | André Serrano |
| **Instituição** | Universidade de Brasília (UnB) |

---

*Base de dados: Telco Customer Churn (IBM Sample Data Sets), disponibilizada no Kaggle por blastchar.*
