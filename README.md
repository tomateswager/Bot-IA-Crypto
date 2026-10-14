# Bot de Trading de Criptomoedas

Bot em Python com analise de mercado, dashboard opcional e integracoes opcionais
com Telegram, DeepSeek, Reddit e feeds externos. O modo predefinido e `paper`;
nao uses `live` sem testar e rever cuidadosamente as credenciais e os limites
de risco.

## Requisitos

- Python 3.11 ou superior.
- Dependencias de `requirements.txt`, incluindo TA-Lib. A instalacao do TA-Lib
pode exigir passos adicionais dependendo do sistema operativo.

## Instalar

Na pasta do projeto, cria e ativa um ambiente virtual:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edita `.env` e adiciona apenas as chaves dos servicos que vais utilizar. As
chaves Binance devem ter apenas as permissoes estritamente necessarias; nunca
habilites levantamentos. O ficheiro `.env` e ignorado pelo Git.

## Executar

Inicia o bot:

```powershell
python Bot_Trade_v2.py
```

Opcionalmente, inicia o dashboard num segundo terminal:

```powershell
.\.venv\Scripts\Activate.ps1
python dashboard.py
```

Os ficheiros de saldo virtual, historico e posicoes sao criados/atualizados
localmente durante a execucao e nao devem ser publicados. O dashboard le esses
ficheiros locais.

## Feeds e tarefas opcionais

- `tools/openclaw_x_collector.py` recolhe dados para o feed externo. Os ficheiros
  `openclaw_x_sources.example.json` e `external_x_feed.example.json` sao apenas
  exemplos de configuracao/dados. Configura os caminhos e variaveis de ambiente
  do coletor antes de o executar.
- `tools/bot_training_monitor.py` e os ficheiros de `deploy/` destinam-se a
  tarefas agendadas num servidor Linux. Revê e adapta os caminhos, utilizadores
  e servicos systemd ao teu servidor antes de instalar.
- `ml_rebuild/` contem experiencias e dados de treino/backtest locais; nao e
  necessario para iniciar o bot.

## Seguranca

- Mantem `.env`, chaves `.pem`, tokens, dados de conta, historico de trades e
  ficheiros de estado fora do Git.
- Comeca sempre em `paper` e confirma o comportamento antes de qualquer uso
  com fundos reais.
- Os exemplos de configuracao nao contem credenciais nem dados reais.
