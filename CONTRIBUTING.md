# Contribuindo com o BotCripto

Obrigado pelo interesse em contribuir! Aqui estão as diretrizes.

## Como contribuir

1. Faça um fork do repositório
2. Crie uma branch descritiva: `git checkout -b feat/nome-da-feature`
3. Faça commits pequenos e com mensagens claras (veja convenção abaixo)
4. Abra um Pull Request com descrição do que foi feito e por quê

## Convenção de commits

Usamos [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: nova funcionalidade
fix: correção de bug
docs: atualização de documentação
refactor: refatoração sem mudança de comportamento
perf: melhoria de performance
chore: tarefas de manutenção (deps, config)
```

## Áreas onde contribuições são especialmente bem-vindas

- **Novos indicadores técnicos** — Ichimoku Cloud, VWAP, Pivot Points
- **Suporte a mais exchanges** — Binance, Bybit via API pública
- **Backtesting** — testar historicamente a precisão dos sinais
- **Notificações** — alertas via Telegram, WhatsApp ou email
- **Testes automatizados** — cobertura com pytest

## Estilo de código

- Python 3.11+
- Type hints sempre que possível
- Docstrings em português em funções públicas
- Linhas de no máximo 100 caracteres

## Reportando bugs

Abra uma issue com:
- Descrição do problema
- Passos para reproduzir
- Comportamento esperado vs. observado
- Versão do Python e SO
