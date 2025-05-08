# DeepSeek Coder Chat Server

Sistema de chat baseado no modelo [DeepSeek Coder 7B Instruct v1.5](https://huggingface.co/deepseek-ai/deepseek-coder-7b-instruct-v1.5) executado localmente em formato nativo (FP16) sem quantização.

## Características

- Interface de chat baseada em WebSocket para comunicação em tempo real
- Execução do modelo DeepSeek Coder 7B em formato nativo (FP16)
- Compatível com GPU CUDA para inferência eficiente
- Histórico de conversas com contexto
- Execução como serviço systemd para inicialização automática com o sistema
- Interface web simples e responsiva

## Requisitos de Sistema

- **Hardware**:
  - GPU com pelo menos 14GB de VRAM (recomendado NVIDIA RTX 3090 ou superior)
  - Pelo menos 16GB de RAM do sistema
  - Espaço em disco suficiente para armazenar o modelo (~14GB)

- **Software**:
  - Ubuntu 20.04 ou superior / Outra distribuição Linux
  - Python 3.8+ com ambiente virtual
  - CUDA 11.7 ou superior e drivers NVIDIA
  - Bibliotecas Python: torch, transformers, fastapi, uvicorn, jinja2

## Instalação

### 1. Preparação do Ambiente

```bash
# Criar diretório para o projeto
mkdir -p /home/deivid/deepseek
cd /home/deivid/deepseek

# Criar e ativar ambiente virtual
python -m venv env
source env/bin/activate

# Instalar dependências
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers fastapi uvicorn jinja2
```

### 2. Baixar o Modelo

O modelo será baixado automaticamente na primeira execução, ou você pode pré-baixá-lo:

```bash
# Baixar o modelo antecipadamente (opcional)
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained('deepseek-ai/deepseek-coder-7b-instruct-v1.5', cache_dir='/home/deivid/deepseek', torch_dtype='float16'); AutoTokenizer.from_pretrained('deepseek-ai/deepseek-coder-7b-instruct-v1.5', cache_dir='/home/deivid/deepseek')"
```

### 3. Configuração de Arquivos

1. Salve o arquivo principal do servidor como `/home/deivid/pruna_model.py`
2. Crie a pasta para o template: `mkdir -p /home/deivid/templates`
3. Salve o arquivo de interface web como `/home/deivid/templates/chat.html`

### 4. Configuração do Serviço Systemd

1. Crie o arquivo de serviço systemd:

```bash
sudo nano /etc/systemd/system/deepseek-coder.service
```

2. Adicione o seguinte conteúdo (ajuste os caminhos conforme necessário):

```ini
[Unit]
Description=DeepSeek Coder Chat Service
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=deivid
WorkingDirectory=/home/deivid

# Ajuste para o caminho correto do seu ambiente virtual
ExecStart=/bin/bash -c 'source /home/deivid/deepseek/env/bin/activate && python /home/deivid/pruna_model.py'

Restart=on-failure
RestartSec=10

# Variáveis de ambiente importantes para GPU
Environment=PYTHONUNBUFFERED=1
Environment=CUDA_VISIBLE_DEVICES=0

# Tempo máximo para inicialização do serviço
TimeoutStartSec=300

[Install]
WantedBy=multi-user.target
```

3. Habilite e inicie o serviço:

```bash
sudo systemctl daemon-reload
sudo systemctl enable deepseek-coder.service
sudo systemctl start deepseek-coder.service
```

## Uso

### Acesso à Interface Web

Acesse a interface de chat através do navegador:

```
http://seu-servidor:8080
```

### Comandos de Chat

- `/limpar` ou `/clear` - Limpa o histórico da conversa atual
- `/novo` ou `/new` - Inicia uma nova conversa
- `/reset` - Reinicia o contexto da conversa

### Monitoramento e Logs

Verifique o status do serviço:

```bash
sudo systemctl status deepseek-coder.service
```

Visualize os logs do serviço:

```bash
# Ver todos os logs
sudo journalctl -u deepseek-coder.service

# Ver logs em tempo real
sudo journalctl -u deepseek-coder.service -f

# Ver apenas os 50 logs mais recentes
sudo journalctl -u deepseek-coder.service -n 50
```

Os logs da aplicação também são salvos em:

```
/home/deivid/chat_pruna.log
```

## Resolução de Problemas

### Erros de Memória GPU

Se você encontrar erros relacionados à memória da GPU:

1. Verifique a quantidade de VRAM disponível:
   ```bash
   nvidia-smi
   ```

2. Reduza o valor de `max_new_tokens` no arquivo `pruna_model.py` (de 512 para um valor menor, como 256)

3. Limpe a cache da GPU antes de iniciar:
   ```bash
   python -c "import torch; torch.cuda.empty_cache()"
   ```

### Serviço Não Inicia

1. Verifique os logs para identificar o problema:
   ```bash
   sudo journalctl -u deepseek-coder.service -f
   ```

2. Certifique-se de que os caminhos no arquivo de serviço estão corretos:
   - Caminho do ambiente virtual
   - Caminho do script Python
   - Caminho do diretório de trabalho

3. Teste a execução manual:
   ```bash
   cd /home/deivid
   source /home/deivid/deepseek/env/bin/activate
   python pruna_model.py
   ```

## Personalização

### Modelo

Para usar outra variante do modelo, altere as variáveis `MODEL_NAME` e `TOKENIZER_NAME` no arquivo `pruna_model.py`.

### Configuração de Geração

Ajuste os parâmetros de geração no método `gerar_resposta` no arquivo `pruna_model.py`:

```python
outputs = self.modelo.generate(
    **inputs,
    max_new_tokens=512,  # Número máximo de tokens gerados
    temperature=0.7,     # Controla a aleatoriedade (menor = mais determinístico)
    top_p=0.95,          # Núcleo de probabilidade (menores valores = menos aleatório)
    do_sample=True,      # Usa amostragem estocástica
    repetition_penalty=1.1  # Penaliza repetições
)
```

### Porta do Servidor

Para alterar a porta padrão (8080), modifique a linha no final do arquivo `pruna_model.py`:

```python
uvicorn.run("pruna_model:app", host="0.0.0.0", port=8080, reload=False)
```

## Licença

Este projeto utiliza o modelo DeepSeek Coder que está sob a [licença DeepSeek License Agreement](https://github.com/deepseek-ai/DeepSeek-Coder/blob/main/LICENSE-MODEL). Consulte os termos desta licença antes de usar o modelo para aplicações comerciais.

## Créditos

- [DeepSeek AI](https://github.com/deepseek-ai) pelo desenvolvimento do modelo DeepSeek Coder
- [FastAPI](https://fastapi.tiangolo.com/) para o framework web
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) para a implementação do modelo