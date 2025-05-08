import os
import time
import json
import logging
import traceback
import torch
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
from typing import List, Dict, Any, Tuple

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chat_pruna.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Chat_Modelo")

# Verificar ambiente e configurar
if torch.cuda.is_available():
    DEVICE = "cuda"
    logger.info(f"Usando GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Memória GPU total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    torch.cuda.empty_cache()
else:
    DEVICE = "cpu"
    logger.warning("GPU não disponível, usando CPU. O desempenho será muito lento para este modelo.")

# Modelo e configurações
MODEL_NAME = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
TOKENIZER_NAME = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
MODEL_PATH = "/home/deivid/deepseek/models/"

# Classe para gerenciar o histórico da conversa
class HistoricoConversa:
    def __init__(self, max_mensagens=10):
        self.historico: List[Dict[str, str]] = []
        self.max_mensagens = max_mensagens
        
    def adicionar_mensagem(self, role: str, content: str):
        """Adiciona uma mensagem ao histórico."""
        self.historico.append({"role": role, "content": content})
        
        # Limitar o tamanho do histórico para evitar tokens excessivos
        if len(self.historico) > self.max_mensagens:
            # Mantém a primeira mensagem (sistema) e remove as mais antigas
            self.historico = [self.historico[0]] + self.historico[-(self.max_mensagens-1):]
            
    def obter_historico_formatado(self) -> str:
        """Retorna o histórico formatado para o modelo DeepSeek."""
        resultado = "<｜begin▁of▁sentence｜>\n"
        
        for msg in self.historico:
            if msg["role"] == "system":
                resultado += f"<｜system｜>\n{msg['content']}\n"
            elif msg["role"] == "user":
                resultado += f"<｜user｜>\n{msg['content']}\n"
            elif msg["role"] == "assistant":
                resultado += f"<｜assistant｜>\n{msg['content']}\n"
        
        # Adicionar tag para a próxima resposta do assistente
        resultado += "<｜assistant｜>"
        return resultado
    
    def limpar(self):
        """Limpa o histórico mantendo apenas a mensagem do sistema."""
        if self.historico and self.historico[0]["role"] == "system":
            self.historico = [self.historico[0]]
        else:
            self.historico = []

class ChatModelo:
    def __init__(self):
        self.modelo = None
        self.tokenizer = None
        # Mapa de sessões para históricos de conversa
        self.sessoes: Dict[str, HistoricoConversa] = {}
        
    def carregar_modelo(self):
        """Carrega o modelo completo em formato nativo (FP16)."""
        logger.info(f"Carregando modelo completo em formato nativo: {MODEL_NAME}")
        logger.info(f"Tokenizer: {TOKENIZER_NAME}")
        logger.info(f"Dispositivo: {DEVICE}")

        inicio = time.time()

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Mostrar versão do transformers
            import transformers
            logger.info(f"Usando transformers versão: {transformers.__version__}")

            # Carregar tokenizer
            logger.info("Carregando tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                TOKENIZER_NAME,
                cache_dir=MODEL_PATH,
                use_fast=True
            )

            # Garantir tokens especiais
            if self.tokenizer.eos_token is None:
                self.tokenizer.eos_token = self.tokenizer.pad_token or '</s>'

            # Monitorar memória antes do carregamento
            if DEVICE == "cuda":
                mem_antes = torch.cuda.memory_allocated(0) / 1024**3
                logger.info(f"Memória GPU antes de carregar modelo: {mem_antes:.2f} GB")
                logger.info(f"Memória GPU livre antes de carregar: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3:.2f} GB")

            # Parâmetros de carregamento - modelo nativo em FP16
            logger.info("Carregando modelo em formato completo (FP16) sem quantização")
            load_config = {
                "pretrained_model_name_or_path": MODEL_NAME,
                "cache_dir": MODEL_PATH,
                "trust_remote_code": True,
                "torch_dtype": torch.float16
            }

            # Carregar modelo
            self.modelo = AutoModelForCausalLM.from_pretrained(**load_config)

            # Mover para GPU
            if DEVICE == "cuda":
                logger.info("Movendo modelo para GPU...")
                self.modelo = self.modelo.to(DEVICE)

            # Monitorar memória após carregamento
            if DEVICE == "cuda":
                mem_depois = torch.cuda.memory_allocated(0) / 1024**3
                logger.info(f"Memória GPU após carregamento: {mem_depois:.2f} GB (+{mem_depois - mem_antes:.2f} GB)")
                logger.info(f"Memória GPU livre após carregar: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3:.2f} GB")

            tempo_carregamento = time.time() - inicio
            logger.info(f"✓ Modelo carregado em {tempo_carregamento:.2f} segundos")
            return True

        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            logger.error(traceback.format_exc())
            return False

    
    def obter_ou_criar_sessao(self, id_sessao: str) -> HistoricoConversa:
        """Obtém uma sessão existente ou cria uma nova."""
        if id_sessao not in self.sessoes:
            self.sessoes[id_sessao] = HistoricoConversa()
            # Adicionar mensagem de sistema inicial
            self.sessoes[id_sessao].adicionar_mensagem("system", 
                "Você é um assistente útil e amigável que responde perguntas de forma clara e concisa. " +
                "Você mantém o contexto da conversa anterior para fornecer respostas relevantes."
            )
        return self.sessoes[id_sessao]
    
    def limpar_sessao(self, id_sessao: str):
        """Limpa o histórico de uma sessão."""
        if id_sessao in self.sessoes:
            self.sessoes[id_sessao].limpar()
            return True
        return False
    
    def gerar_resposta(self, mensagem: str, id_sessao: str):
        """Processa a mensagem e gera uma resposta, mantendo o histórico da conversa."""
        if not self.modelo or not self.tokenizer:
            return "Modelo não carregado. Reinicie o servidor."
        
        sessao = self.obter_ou_criar_sessao(id_sessao)
        
        try:
            # Adicionar mensagem do usuário ao histórico
            sessao.adicionar_mensagem("user", mensagem)
            
            # Obter o histórico formatado para o modelo
            prompt_formatado = sessao.obter_historico_formatado()
            
            # Tokenizar o prompt
            inputs = self.tokenizer(prompt_formatado, return_tensors="pt").to(DEVICE)
            
            # Monitorar memória antes da inferência
            if DEVICE == "cuda":
                mem_antes = torch.cuda.memory_allocated(0) / 1024**3
                logger.info(f"Memória GPU antes da inferência: {mem_antes:.2f} GB")
                logger.info(f"Memória GPU livre antes da inferência: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3:.2f} GB")
            
            # Realizar inferência
            inicio = time.time()
            with torch.no_grad():
                outputs = self.modelo.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            tempo_inferencia = time.time() - inicio
            
            # Monitorar memória após inferência
            if DEVICE == "cuda":
                mem_depois = torch.cuda.memory_allocated(0) / 1024**3
                logger.info(f"Memória GPU após inferência: {mem_depois:.2f} GB (+{mem_depois - mem_antes:.2f} GB)")
                logger.info(f"Memória GPU livre após inferência: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3:.2f} GB")
            
            # Decodificar a resposta
            resposta_completa = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extrair apenas a parte da resposta do assistente
            if "<｜assistant｜>" in resposta_completa:
                resposta = resposta_completa.split("<｜assistant｜>")[-1].strip()
            else:
                resposta = resposta_completa
            
            if "<｜end▁of▁sentence｜>" in resposta:
                resposta = resposta.split("<｜end▁of▁sentence｜>")[0].strip()
            
            # Adicionar resposta do assistente ao histórico
            sessao.adicionar_mensagem("assistant", resposta)
            
            logger.info(f"Resposta gerada em {tempo_inferencia:.2f} segundos")
            
            # Limpar cache CUDA após inferência
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
                mem_após_limpeza = torch.cuda.memory_allocated(0) / 1024**3
                logger.info(f"Memória GPU após limpeza: {mem_após_limpeza:.2f} GB")
                logger.info(f"Memória GPU livre após limpeza: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3:.2f} GB")
                
            return resposta
            
        except Exception as e:
            logger.error(f"Erro durante geração de resposta: {e}")
            logger.error(traceback.format_exc())
            return f"Erro ao processar a mensagem: {str(e)}"

# Inicializar o modelo
modelo_chat = ChatModelo()

# Inicializar a aplicação FastAPI
app = FastAPI(title="API de Chat com DeepSeek Coder (Modelo Nativo)")

# Diretório para templates
templates = Jinja2Templates(directory="templates")

# Criar diretório de templates se não existir
os.makedirs("templates", exist_ok=True)

# Rota principal para a página do chat
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

# Endpoint WebSocket para comunicação em tempo real com o chat
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Gerar ID de sessão único para este cliente
    import uuid
    session_id = str(uuid.uuid4())
    logger.info(f"Nova sessão iniciada: {session_id}")
    
    try:
        # Verificar se o modelo está carregado
        if modelo_chat.modelo is None:
            # Enviar mensagem enquanto carrega
            await websocket.send_text(json.dumps({
                "sender": "system",
                "message": "Carregando modelo DeepSeek Coder em formato nativo (FP16)... Isso pode demorar alguns minutos."
            }))
            
            # Carregar o modelo
            if not modelo_chat.carregar_modelo():
                await websocket.send_text(json.dumps({
                    "sender": "system",
                    "message": "Erro ao carregar o modelo. Verifique os logs para mais informações. Certifique-se de ter memória GPU suficiente (~14GB)."
                }))
                return
            
            await websocket.send_text(json.dumps({
                "sender": "system",
                "message": "Modelo carregado com sucesso! Você pode começar a conversar. O modelo vai lembrar do contexto da conversa anterior."
            }))
        
        # Loop para comunicação contínua
        while True:
            # Receber mensagem do cliente
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_message = message_data.get("message", "")
            
            # Verificar se é um comando para limpar o histórico
            if user_message.lower() in ["/limpar", "/clear", "/reset", "/novo", "/new"]:
                modelo_chat.limpar_sessao(session_id)
                await websocket.send_text(json.dumps({
                    "sender": "system",
                    "message": "Histórico da conversa foi limpo. Vamos começar um novo chat!"
                }))
                continue
            
            # Gerar e enviar resposta
            inicio = time.time()
            resposta = modelo_chat.gerar_resposta(user_message, session_id)
            tempo_total = time.time() - inicio
            
            await websocket.send_text(json.dumps({
                "sender": "bot",
                "message": resposta,
                "time": f"{tempo_total:.2f}"
            }))
            
    except WebSocketDisconnect:
        logger.info(f"Cliente desconectado: {session_id}")
    except Exception as e:
        logger.error(f"Erro no WebSocket para sessão {session_id}: {e}")
        try:
            await websocket.send_text(json.dumps({
                "sender": "system",
                "message": f"Ocorreu um erro: {str(e)}"
            }))
        except:
            pass

# Verificar se o arquivo de template existe
if not os.path.exists("templates/chat.html"):
    logger.info("Arquivo de template não encontrado. Certifique-se de criar o arquivo templates/chat.html.")

# Inicializar o servidor se este script for executado diretamente
if __name__ == "__main__":
    # Iniciar servidor
    uvicorn.run("pruna_model:app", host="0.0.0.0", port=8080, reload=False)