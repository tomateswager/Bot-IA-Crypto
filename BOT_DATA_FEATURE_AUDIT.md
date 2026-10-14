# Auditoria de Dados e Features do Bot V2

Data: 2026-05-25

## Objetivo

Deixar o bot mais eficaz sem o sobrecarregar. A regra principal é: dados que ajudam a decisão entram com peso controlado; dados barulhentos entram apenas como contexto/risco; dados duplicados ou pouco causais não devem dominar o ML.

## Implementado nesta fase

- `feature_snapshots` em SQLite: guarda por moeda/candle as features, contexto, votos, probabilidades ML e scores finais.
- `orderflow_snapshots` em SQLite: guarda snapshots leves do order book agregado, com retenção curta.
- `liquidation_events` em SQLite: recebe eventos reais de liquidações via Binance Futures WebSocket `!forceOrder@arr`.
- Integração de liquidações WS no contexto e votos, sem substituir os guards de risco.
- Calendário local opcional de `token_unlocks.json`.
- Calendário macro local opcional de `macro_calendar.json`.
- TPO/Volume Profile, footprint, funding, OI, order book e liquidações agora ficam registados de forma causal para auditoria e melhoria futura.

## Dados que devem ficar como essenciais

1. Price action multi-timeframe: OHLCV 1h, 4h e 1d.
2. Tendência: ADX, DI+/DI-, EMA spreads, Supertrend.
3. Momentum: RSI, Stoch K, MACD histogram e slopes.
4. Volatilidade: ATR%, BB position/width, realized volatility, CHOP.
5. Volume: volume ratio, VWAP distance, OBV momentum, MFI, CMF.
6. Estrutura: support/resistance, range position, breakout/breakdown, TPO, VPOC.
7. Order flow: footprint, CVD real, order book imbalance e snapshots.
8. Derivativos: funding, funding z-score, OI, taker buy ratio, long/short ratio.
9. Risco sistémico: Fear & Greed, BTC trend, BTC dominance, macro/news risk.
10. Gestão de risco: VaR, Kelly, circuit breaker, cooldowns, exposure caps.

## Dados úteis mas com peso limitado

- Notícias RSS e feed externo X: úteis para choque e narrativa, mas não devem mandar sozinhos.
- Reddit/social volume: só como confirmação fraca ou alerta de hype.
- GitHub activity: útil para L1/infra, fraco para meme coins e tokens puramente especulativos.
- Options BTC/ETH: útil para BTC/ETH e contexto macro; fraco para altcoins pequenas.
- Polymarket/ETF/Truth Social: contexto macro, não sinal direto de entrada.
- Token unlocks: risco de oferta, sobretudo para LONG; não é sinal suficiente para SHORT sozinho.
- Macro calendar: bloqueia ou reduz agressividade perto de eventos, não prevê direção sozinho.

## Dados que não devem dominar

- Features temporais puras, como hora/dia da semana: risco alto de overfit em crypto 24/7.
- Preços absolutos de Fibonacci no ML multi-coin: melhor usar distância percentual/posição.
- Indicadores duplicados: Williams%R, Stoch D, MACD raw e CCI tendem a repetir RSI/Stoch/MACD_hist.
- Notícias genéricas de “price prediction”: muito ruído, devem ter peso baixo.
- Qualquer feature com AUC próxima de 0.50 durante várias janelas: deve ficar neutra ou ser podada.

## Política final

- O bot deve usar poucas confluências fortes em vez de dezenas de sinais fracos.
- ML só deve pesar quando AUC/PSI indicam utilidade; se estiver random, votos técnicos decidem.
- Order flow e Profile ajudam a timing de entrada, mas não substituem tendência, risco e SL.
- Guardrails nunca saem: SL, Kelly, VaR, circuit breaker, max exposure e cooldown pós-perda.
- O próximo passo correto é deixar o feature store acumular dados e depois rodar análise de importância/performance por família.
