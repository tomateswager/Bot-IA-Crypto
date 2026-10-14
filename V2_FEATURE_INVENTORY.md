# Inventario V2 para migracao para bot unico V13

Objetivo: deixar de operar dois bots e transformar o V13 no bot unico, modular e testavel, migrando do V2 apenas funcionalidades que possam ser validadas sem look-ahead e sem aumentar risco operacional.

## Estado da migracao

| Area | Existe no V2 | Estado no V13 |
|---|---|---|
| Indicadores tecnicos base | RSI, MACD, ADX, DI, EMA, ATR, BB, OBV, MFI, CMF | Migrado |
| Indicadores adicionais | VWAP, CHOP, CCI, Williams R, MOM, ROC | Migrado |
| Smart money/candle structure | FVG, order block proxy, squeeze, volume imbalance | Migrado |
| CVD/flow proxy | CVD norm, CVD z-score, slope, volume imbalance | Migrado como proxy causal |
| Padroes de velas | TA-Lib patterns amplos | Parcialmente migrado: core patterns principais |
| Multi-timeframe causal | 1h/4h/1d | Migrado com shift para candle fechado |
| ML | Stacking/XGB/LGB/RF, isotonic, PSI, purged CV | V13 tem XGB calibrado; endurecido com AUC/AP/Brier/ECE |
| DeepSeek/IA | tecnico, macro, neutral, council, adversarial, loss review | Adapter V13 criado como filtro conservador de candidatos |
| RL/self-learning | min_score por moeda, block rules, reward, DeepSeek em perdas | RL Guardian V13 criado como versao leve/conservadora |
| Macro | Fear & Greed, funding, BTC.D, DXY/SPY, regime | Context provider V13 criado |
| Noticias/sentimento | RSS, CryptoPanic, VADER/TextBlob | Context provider V13 criado |
| Social | Reddit, GitHub activity, social volume | Reddit defensivo migrado; GitHub/social volume ficam para challenger |
| Order book/execucao | order book imbalance, sweep impact, taker ratio | Migrado como filtro defensivo de execucao |
| Gestao de posicao | TP1, breakeven, trailing ATR/chandelier, panic exit | Implementado e testavel; desligado por default porque piorou backtest |
| Portfolio risk | correlation, VaR/CVaR, sector exposure | Ainda pendente |
| Telegram/dashboard | comandos, digest, forensic logs | Telegram/eventos/report periodico + dashboard sync/status V13 migrados |
| WebSocket | price/orderbook monitor | Ainda pendente |

## Principio de migracao

O V2 nao deve ser copiado inteiro para o V13. O V2 tem valor, mas tambem mistura muitas responsabilidades num ficheiro unico. A migracao correta e por adapters:

- `features.py`: features tecnicas causais.
- `context_providers.py`: macro, noticias e sentimento.
- `ai_advisor.py`: DeepSeek como filtro/adjudicador.
- `rl_guardian.py`: feedback conservador baseado em trades fechadas.
- `production_runner.py`: orquestracao, estado e risk loop.

## Ja feito neste ciclo

- V13 mantido como bot unico ativo na AWS; V2 parado.
- Resultados contaminados por look-ahead movidos para `backtest_v2/results/invalid_lookahead/`.
- Criado `backtest_v2/validate_v13.py`.
- Criado lock de dependencias AWS.
- Criado `.gitignore` e repo Git local.
- V13 ganhou `risk_loop` leve com `--risk-interval`.
- ML agora mede AUC, Average Precision, Brier e ECE.
- `decision_engine` e `ml_driven` so usam modelos considerados uteis e calibrados.
- Features V2 tecnicas principais foram migradas para o V13.
- Adapters de contexto, DeepSeek e RL Guardian foram criados.
- Filtros microstructure/flow adicionados ao `trend_pullback`.
- Filtro de execucao por order book/taker ratio criado.
- TP1/breakeven/trailing criado no backtester e runner, mas mantido off por nao superar baseline.
- Dashboard legado agora pode refletir apenas o V13 via `V13_SYNC_DASHBOARD=1`.

## Pendencias de maior impacto

1. Backtestar as novas features no pipeline ML e comparar contra `trend_pullback`.
2. Implementar paper challenger: `trend_pullback` champion vs ML/context/AI challenger.
3. Validar portfolio risk: correlation, VaR/CVaR e exposicao por setor.
4. Criar dashboard V13 proprio, separado do dashboard legado.
5. Testar variantes TP1/trailing em walk-forward antes de ativar.
6. So depois promover partes experimentais para producao real.
