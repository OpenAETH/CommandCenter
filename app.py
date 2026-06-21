from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from bson import ObjectId
from datetime import datetime, timedelta
import os, json, re, certifi, unicodedata, time, random

try:
    from groq import Groq as GroqClient
except ImportError:
    GroqClient = None

try:
    from headroom import compress as hr_compress
except ImportError:
    hr_compress = None

app = Flask(__name__)
CORS(app)

# ============================================================================
# MONGODB CONNECTION
# ============================================================================

MONGODB_URI = os.environ.get('MONGODB_URI', '')
MONGODB_DB  = os.environ.get('MONGODB_DB', 'commandcenter')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
# Modelo Groq (env var para poder cambiarlo sin redeploy; Qwen3-32B está en preview).
GROQ_MODEL = os.environ.get('GROQ_MODEL', 'qwen/qwen3-32b')
# Presupuesto de salida. Qwen3 es reasoning: el razonamiento consume este budget junto con
# el JSON de tool_calls. OJO: Groq cuenta input+max_tokens contra el límite de TPM (6000 en
# el tier gratis), así que un max_tokens alto solo dispara 413 'request too large'. Con tools
# de a una y reasoning_format='hidden', 2048 alcanza para una operación por paso sin truncar.
GROQ_MAX_TOKENS = int(os.environ.get('GROQ_MAX_TOKENS', '2048'))
# Reintentos ante fallos transitorios de Groq (rate-limit 429 / 5xx / errores de red).
GROQ_MAX_RETRIES = int(os.environ.get('GROQ_MAX_RETRIES', '3'))
# Pausa (ms) entre pasos del loop de tools, para no acumular requests dentro de la misma
# ventana de TPM (rate-limit por minuto). El Sync 1×1 ejecuta una tool por paso y espera esto.
GROQ_THROTTLE_MS = int(os.environ.get('GROQ_THROTTLE_MS', '350'))
# Techo de TPM del tier de Groq. Groq cuenta input+max_tokens contra este límite por minuto;
# si la suma lo supera devuelve 413 'request too large' (NO es truncamiento). Recortamos el
# request del lado servidor para que SIEMPRE entre, en vez de propagar el 413 al usuario.
GROQ_TPM_LIMIT = int(os.environ.get('GROQ_TPM_LIMIT', '6000'))
# Colchón de seguridad: nuestra cuenta de tokens es una estimación (no tokeniza igual que Groq),
# así que dejamos margen para no rozar el techo por un error de redondeo.
GROQ_TPM_MARGIN = int(os.environ.get('GROQ_TPM_MARGIN', '600'))
# Piso de max_tokens: aunque haya que achicar el budget de salida para que entre el request,
# nunca bajamos de acá (debajo de esto Qwen3 trunca el JSON de tools, ver memoria del bug).
GROQ_MIN_TOKENS = int(os.environ.get('GROQ_MIN_TOKENS', '1024'))
# Compresión de contexto (Headroom) antes de cada llamada a Groq: achica los tool outputs / JSON
# de Mongo que se reinyectan al modelo en cada paso del loop. Se desactiva sola si headroom-ai no
# está instalado; HEADROOM_DISABLE=1 la apaga sin redeploy.
HEADROOM_ENABLED = bool(hr_compress) and os.environ.get('HEADROOM_DISABLE') != '1'
# Modelo SOLO para el conteo interno de tokens de Headroom — NO es el modelo que responde (ese es
# GROQ_MODEL). 'gpt-4o' usa tiktoken (ya viene con headroom-ai); un nombre 'qwen/...' enrutaría al
# tokenizer HuggingFace, que exige el paquete `transformers` (~2GB) y, si falta, deja la compresión
# en no-op silencioso. El conteo queda aproximado para Qwen, pero eso solo afecta decisiones
# internas de Headroom: el gating real de TPM lo sigue haciendo _fit_messages/_estimate_tokens.
HEADROOM_TOKENIZER_MODEL = os.environ.get('HEADROOM_TOKENIZER_MODEL', 'gpt-4o')

client = MongoClient(
    MONGODB_URI,
    server_api=ServerApi("1"),
    tls=True,
    tlsCAFile=certifi.where(),
    serverSelectionTimeoutMS=5000,
)
db = client[MONGODB_DB]

# max_retries=0: desactivamos el retry INTERNO del SDK de Groq. Por default reintenta 2 veces y,
# ante un 429 (TPM), DUERME el 'retry-after' completo (puede ser 30-60s) dentro de create() —ese
# sleep largo choca con el --timeout de gunicorn y mata al worker (SIGKILL) antes de que nuestro
# propio backoff capeado en _groq_create llegue a correr. Manejamos los reintentos nosotros.
groq_client = (GroqClient(api_key=GROQ_API_KEY, max_retries=0)
               if (GroqClient and GROQ_API_KEY) else None)

# ============================================================================
# HELPERS
# ============================================================================

def doc(d):
    if d is None:
        return None
    d = dict(d)
    d['id'] = str(d.pop('_id'))
    for k, v in list(d.items()):
        if hasattr(v, 'isoformat'):
            d[k] = v.isoformat()
    return d

def oid(s):
    return ObjectId(s)

_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)

def strip_think(text):
    """Quita bloques <think>…</think> que emiten los modelos reasoning (Qwen3).
    Red de seguridad por si reasoning_format no los oculta en algún caso."""
    if not text:
        return text
    return _THINK_RE.sub('', text).strip()

def apply_completion(update, status, now):
    """Mantiene done y completed_at coherentes con el status.
    Setea completed_at al pasar a 'done'; lo limpia al salir de 'done'.
    Muta y devuelve el dict `update`."""
    if status == 'done':
        update['done'] = 1
        update['completed_at'] = now
    else:
        update['done'] = 0
        update['completed_at'] = None
    return update

# ============================================================================
# HEALTH
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health():
    try:
        client.admin.command('ping')
        return jsonify({
            'status': 'ok', 'db': 'connected',
            'mongo_db': MONGODB_DB,
            'groq': bool(groq_client),
            'headroom': HEADROOM_ENABLED,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'status': 'error', 'db': str(e)}), 500

# ============================================================================
# PRODUCTS
# ============================================================================

@app.route('/api/products', methods=['GET'])
def get_products():
    try:
        return jsonify([doc(r) for r in db.products.find().sort([('sort_order', 1), ('_id', 1)])])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/products', methods=['POST'])
def create_product():
    d = request.json
    try:
        now = datetime.utcnow()
        product = {
            'name': d.get('name'), 'icon': d.get('icon', '📦'),
            'description': d.get('description', ''), 'status': d.get('status', 'activo'),
            'color': d.get('color', '#00e5ff'),
            'sort_order': db.products.count_documents({}) + 1,
            'created_at': now, 'updated_at': now,
        }
        result = db.products.insert_one(product)
        product['_id'] = result.inserted_id
        return jsonify(doc(product)), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/products/<string:pid>', methods=['PUT'])
def update_product(pid):
    d = request.json
    try:
        update = {
            'name': d.get('name'), 'icon': d.get('icon', '📦'),
            'description': d.get('description', ''), 'status': d.get('status', 'activo'),
            'color': d.get('color', '#00e5ff'), 'updated_at': datetime.utcnow(),
        }
        result = db.products.find_one_and_update(
            {'_id': oid(pid)}, {'$set': update}, return_document=True)
        if not result:
            return jsonify({'error': 'Product not found'}), 404
        return jsonify(doc(result))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/products/<string:pid>', methods=['DELETE'])
def delete_product(pid):
    try:
        db.tasks.delete_many({'product_id': pid})
        db.products.delete_one({'_id': oid(pid)})
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# TASKS
# ============================================================================

@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    try:
        return jsonify([doc(r) for r in db.tasks.find().sort([('product_id', 1), ('module', 1), ('_id', 1)])])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tasks', methods=['POST'])
def create_task():
    d = request.json
    try:
        now = datetime.utcnow()
        status = d.get('status', 'todo')
        task = {
            'product_id': d.get('product_id'),
            'module': d.get('module', 'Backend'),
            'name': d.get('name'), 'description': d.get('description', ''),
            'status': status, 'priority': d.get('priority', 'medio'),
            'impact': d.get('impact', 'medio'),
            'created_at': now, 'updated_at': now,
        }
        apply_completion(task, status, now)
        result = db.tasks.insert_one(task)
        task['_id'] = result.inserted_id
        return jsonify(doc(task)), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tasks/<string:tid>', methods=['PUT'])
def update_task(tid):
    d = request.json
    try:
        now = datetime.utcnow()
        status = 'done' if d.get('done') else d.get('status', 'todo')
        update = {
            'product_id': d.get('product_id'), 'module': d.get('module', 'Backend'),
            'name': d.get('name'), 'description': d.get('description', ''),
            'status': status, 'priority': d.get('priority', 'medio'),
            'impact': d.get('impact', 'medio'),
            'updated_at': now,
        }
        apply_completion(update, status, now)
        result = db.tasks.find_one_and_update(
            {'_id': oid(tid)}, {'$set': update}, return_document=True)
        if not result:
            return jsonify({'error': 'Task not found'}), 404
        return jsonify(doc(result))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tasks/<string:tid>/toggle', methods=['PATCH'])
def toggle_task(tid):
    try:
        current = db.tasks.find_one({'_id': oid(tid)})
        if not current:
            return jsonify({'error': 'Task not found'}), 404
        now = datetime.utcnow()
        new_status = 'doing' if current.get('done') else 'done'
        update = {'status': new_status, 'updated_at': now}
        apply_completion(update, new_status, now)
        result = db.tasks.find_one_and_update(
            {'_id': oid(tid)}, {'$set': update}, return_document=True)
        return jsonify(doc(result))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tasks/<string:tid>', methods=['DELETE'])
def delete_task(tid):
    try:
        db.tasks.delete_one({'_id': oid(tid)})
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# CAMPAÑA — DASHBOARD DE LA DESARROLLADORA (datos derivados del módulo DEV)
# ============================================================================

def compute_dev_metrics():
    """Métricas agregadas del módulo DEV. Compartido por el endpoint /api/dev-metrics
    y el tool get_dev_metrics del asistente. Todo se deriva de products + tasks."""
    products = list(db.products.find())
    tasks = list(db.tasks.find())

    # Conteo por estado (normalizado a todo/doing/done)
    by_status = {'todo': 0, 'doing': 0, 'done': 0}
    for t in tasks:
        st = 'done' if t.get('done') else t.get('status', 'todo')
        by_status[st if st in by_status else 'todo'] += 1

    total = len(tasks)
    done_n = by_status['done']
    pct = round(done_n / total * 100) if total else 0

    # Completadas por día — últimos 7 días (incluye hoy)
    today = datetime.utcnow().date()
    week = [today - timedelta(days=i) for i in range(6, -1, -1)]
    week_counts = {d.isoformat(): 0 for d in week}
    for t in tasks:
        if not t.get('done'):
            continue
        # completed_at es el camino principal; updated_at como fallback robusto
        ts = t.get('completed_at') or t.get('updated_at')
        if not ts:
            continue
        d = ts.date() if hasattr(ts, 'date') else None
        if d and d.isoformat() in week_counts:
            week_counts[d.isoformat()] += 1
    weekly = [
        {'date': d.isoformat(),
         'label': d.strftime('%a'),
         'count': week_counts[d.isoformat()]}
        for d in week
    ]

    # Progreso por producto
    per_product = []
    for p in products:
        pid = str(p['_id'])
        ptasks = [t for t in tasks if t.get('product_id') == pid]
        pdone = sum(1 for t in ptasks if t.get('done'))
        per_product.append({
            'id': pid,
            'name': p.get('name', ''),
            'icon': p.get('icon', '📦'),
            'color': p.get('color', '#00e5ff'),
            'status': p.get('status', 'activo'),
            'total': len(ptasks),
            'done': pdone,
            'pct': round(pdone / len(ptasks) * 100) if ptasks else 0,
        })

    return {
        'products_total':  len(products),
        'products_active': sum(1 for p in products if p.get('status') == 'activo'),
        'tasks_total':     total,
        'tasks_todo':      by_status['todo'],
        'tasks_doing':     by_status['doing'],
        'tasks_done':      done_n,
        'pct_done':        pct,
        'weekly_completed': weekly,
        'week_total':      sum(week_counts.values()),
        'per_product':     per_product,
    }


@app.route('/api/dev-metrics', methods=['GET'])
def dev_metrics():
    """Métricas agregadas del módulo DEV para el panel Campaña."""
    try:
        return jsonify(compute_dev_metrics())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# STRATEGY LOGS
# ============================================================================

@app.route('/api/logs', methods=['GET'])
def get_logs():
    try:
        result = []
        for r in db.strategy_logs.find().sort('created_at', -1):
            d = doc(r)
            if not isinstance(d.get('links'), list):
                d['links'] = []
            result.append(d)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/logs', methods=['POST'])
def create_log():
    d = request.json
    try:
        now = datetime.utcnow()
        log = {
            'type': d.get('type', 'Insight'), 'title': d.get('title', ''),
            'text': d.get('text'), 'links': d.get('links', []),
            'date': d.get('date', datetime.now().strftime('%Y-%m-%d')),
            'created_at': now,
        }
        result = db.strategy_logs.insert_one(log)
        log['_id'] = result.inserted_id
        return jsonify(doc(log)), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/logs/<string:lid>', methods=['PUT'])
def update_log(lid):
    d = request.json
    try:
        update = {
            'type':  d.get('type', 'Insight'),
            'title': d.get('title', ''),
            'text':  d.get('text', ''),
            'links': d.get('links', []),
            'date':  d.get('date', datetime.now().strftime('%Y-%m-%d')),
            'updated_at': datetime.utcnow(),
        }
        result = db.strategy_logs.find_one_and_update(
            {'_id': oid(lid)}, {'$set': update}, return_document=True)
        if not result:
            return jsonify({'error': 'Log not found'}), 404
        return jsonify(doc(result))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/logs/<string:lid>', methods=['DELETE'])
def delete_log(lid):
    try:
        db.strategy_logs.delete_one({'_id': oid(lid)})
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# STATS
# ============================================================================

@app.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        tasks_total = db.tasks.count_documents({})
        tasks_done_n = db.tasks.count_documents({'done': 1})
        return jsonify({
            'tasks_doing':     db.tasks.count_documents({'status': 'doing'}),
            'tasks_todo':      db.tasks.count_documents({'status': 'todo'}),
            'tasks_done':      tasks_done_n,
            'tasks_total':     tasks_total,
            'insights':        db.strategy_logs.count_documents({'type': 'Insight'}),
            'logs_total':      db.strategy_logs.count_documents({}),
            'products_total':  db.products.count_documents({}),
            'products_active': db.products.count_documents({'status': 'activo'}),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# GROQ CHATBOT
# ─────────────────────────────────────────────────────────────────────────────
# Design principle: every tool is SELF-CONTAINED — no ID passing between rounds.
# This eliminates the parallel-call race condition where the LLM calls
# create_task with a hallucinated product_id before create_product returns.
#
# Tool catalogue:
#   create_product_with_tasks  → atomic: product + N tasks in one shot
#   add_tasks_to_product       → find product by name internally, then insert tasks
#   list_products              → list all products
#   list_tasks                 → list tasks (optionally filtered by product name)
#   update_tasks               → bulk-update tasks by name + product name
# ============================================================================

GROQ_TOOLS = [
    # ── 1. Create product + tasks atomically ──────────────────────────────────
    {"type": "function", "function": {
        "name": "create_product_with_tasks",
        "description": (
            "Crea un producto nuevo junto con todas sus tareas en UNA sola operación atómica. "
            "Usá esta herramienta cuando el usuario quiera crear un producto y sus tareas a la vez. "
            "No necesitás buscar product_id antes; esta herramienta lo hace internamente."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "product_name":        {"type": "string", "description": "Nombre del producto"},
                "product_icon":        {"type": "string", "description": "Emoji, ej: 🚀"},
                "product_description": {"type": "string"},
                "product_status":      {"type": "string", "enum": ["activo","idea","pausado","archivado"]},
                "product_color":       {"type": "string", "description": "Hex, ej: #00e5ff"},
                "module":              {"type": "string", "description": "Módulo para todas las tasks, ej: Backend"},
                "tasks": {
                    "type": "array",
                    "description": "Lista de tareas a crear dentro de este producto",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name":        {"type": "string"},
                            "description": {"type": "string"},
                            "priority":    {"type": "string", "enum": ["alto","medio","bajo"]},
                            "impact":      {"type": "string", "enum": ["alto","medio","bajo"]},
                            "status":      {"type": "string", "enum": ["todo","doing","done"]},
                            "module":      {"type": "string", "description": "Override del módulo para esta task específica"},
                        },
                        "required": ["name"]
                    }
                }
            },
            "required": ["product_name", "tasks"]
        }
    }},

    # ── 2. Add tasks to an EXISTING product (by name) ─────────────────────────
    {"type": "function", "function": {
        "name": "add_tasks_to_product",
        "description": (
            "Agrega tareas a un producto que YA EXISTE. "
            "Busca el producto por nombre internamente (no necesitás product_id). "
            "Usá esta herramienta cuando el producto ya existe y sólo hay que agregar tareas."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "product_name": {"type": "string", "description": "Nombre del producto (parcial, case-insensitive)"},
                "module":       {"type": "string", "description": "Módulo por defecto para las tasks"},
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name":        {"type": "string"},
                            "description": {"type": "string"},
                            "priority":    {"type": "string", "enum": ["alto","medio","bajo"]},
                            "impact":      {"type": "string", "enum": ["alto","medio","bajo"]},
                            "status":      {"type": "string", "enum": ["todo","doing","done"]},
                            "module":      {"type": "string"},
                        },
                        "required": ["name"]
                    }
                }
            },
            "required": ["product_name", "tasks"]
        }
    }},

    # ── 3. List products ───────────────────────────────────────────────────────
    {"type": "function", "function": {
        "name": "list_products",
        "description": "Lista todos los productos con su id, nombre y estado.",
        "parameters": {"type": "object", "properties": {}, "required": []}
    }},

    # ── 4. List tasks ──────────────────────────────────────────────────────────
    {"type": "function", "function": {
        "name": "list_tasks",
        "description": (
            "Lista tareas. Filtros opcionales: product_name (producto, por nombre parcial) y "
            "module (módulo dentro de ese producto, ej: 'IA', 'Backend'; ignora acentos y "
            "mayúsculas). Para '¿qué tareas hay en el módulo X de Y?' pasá product_name=Y y "
            "module=X en una sola llamada; no traigas todo para filtrar a mano."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "product_name": {"type": "string", "description": "Nombre del producto para filtrar (opcional)"},
                "module":       {"type": "string", "description": "Módulo para filtrar dentro del producto (opcional)"}
            }
        }
    }},

    # ── 4b. List modules of a product ─────────────────────────────────────────
    {"type": "function", "function": {
        "name": "list_modules",
        "description": (
            "Lista los módulos que YA EXISTEN dentro de un producto (busca por nombre parcial). "
            "Usala antes de agregar tareas para reutilizar un módulo existente en vez de crear "
            "variantes duplicadas (Backend vs backend)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "product_name": {"type": "string", "description": "Nombre del producto (parcial)"}
            },
            "required": ["product_name"]
        }
    }},

    # ── 5. Update tasks by name + product ─────────────────────────────────────
    {"type": "function", "function": {
        "name": "update_tasks",
        "description": (
            "Actualiza una o varias tareas buscándolas por nombre (parcial) y producto (nombre parcial). "
            "No necesitás el ID de la task."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "product_name": {"type": "string", "description": "Nombre del producto (parcial)"},
                "updates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task_name":   {"type": "string", "description": "Nombre parcial de la task a actualizar"},
                            "status":      {"type": "string", "enum": ["todo","doing","done"]},
                            "priority":    {"type": "string", "enum": ["alto","medio","bajo"]},
                            "impact":      {"type": "string", "enum": ["alto","medio","bajo"]},
                            "module":      {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["task_name"]
                    }
                }
            },
            "required": ["updates"]
        }
    }},

    # ── 6. Update product (by name) ───────────────────────────────────────────
    {"type": "function", "function": {
        "name": "update_product",
        "description": (
            "Modifica un producto existente buscándolo por nombre (parcial). "
            "Sólo cambia los campos que se pasen. No necesitás el ID."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "product_name": {"type": "string", "description": "Nombre actual del producto (parcial)"},
                "new_name":     {"type": "string", "description": "Nuevo nombre (opcional)"},
                "icon":         {"type": "string"},
                "description":  {"type": "string"},
                "status":       {"type": "string", "enum": ["activo","idea","pausado","archivado"]},
                "color":        {"type": "string", "description": "Hex, ej: #00e5ff"},
            },
            "required": ["product_name"]
        }
    }},

    # ── 7. Move tasks to another product/module ───────────────────────────────
    {"type": "function", "function": {
        "name": "move_tasks",
        "description": (
            "Mueve tareas a otro producto y/o cambia su módulo. "
            "Busca tareas por nombre (parcial) dentro del producto origen."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "from_product":   {"type": "string", "description": "Producto origen (parcial). Opcional si los nombres de task son únicos."},
                "to_product":     {"type": "string", "description": "Producto destino (parcial)."},
                "new_module":     {"type": "string", "description": "Nuevo módulo para las tareas movidas (opcional)."},
                "task_names":     {"type": "array", "items": {"type": "string"},
                                   "description": "Nombres (parciales) de las tareas a mover."},
            },
            "required": ["to_product", "task_names"]
        }
    }},

    # ── 8. Delete tasks (by name + product) ───────────────────────────────────
    {"type": "function", "function": {
        "name": "delete_tasks",
        "description": (
            "Elimina una o varias tareas buscándolas por nombre (parcial) y producto (parcial). "
            "Acción destructiva: usala sólo cuando el usuario lo pida explícitamente."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "product_name": {"type": "string", "description": "Producto (parcial), para acotar la búsqueda."},
                "task_names":   {"type": "array", "items": {"type": "string"},
                                 "description": "Nombres (parciales) de las tareas a eliminar."},
            },
            "required": ["task_names"]
        }
    }},

    # ── 9. Delete product (and its tasks) ─────────────────────────────────────
    {"type": "function", "function": {
        "name": "delete_product",
        "description": (
            "Elimina un producto y TODAS sus tareas. Acción destructiva: "
            "usala sólo cuando el usuario lo pida explícitamente y sin ambigüedad."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "product_name": {"type": "string", "description": "Nombre del producto (parcial)."},
            },
            "required": ["product_name"]
        }
    }},

    # ── 10. Get dev metrics (developer dashboard) ─────────────────────────────
    {"type": "function", "function": {
        "name": "get_dev_metrics",
        "description": (
            "Devuelve métricas agregadas de la desarrolladora: cantidad de productos, "
            "tareas por estado (todo/doing/done), % completado, completadas esta semana "
            "y progreso por producto. Usalo para responder preguntas de avance o reportes."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []}
    }},

    # ── 11. Create strategy log ───────────────────────────────────────────────
    {"type": "function", "function": {
        "name": "create_log",
        "description": (
            "Registra una entrada en Estrategia: decisión, insight, riesgo, oportunidad, "
            "aprendizaje, objetivo, hipótesis o hito que surja de la conversación. Útil para capturar aprendizajes. "
            "Guardá el contenido COMPLETO y literal: no resumas ni trunques."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "type":  {"type": "string", "enum": ["Decision","Insight","Riesgo","Oportunidad","Aprendizaje","Objetivo","Hipótesis","Hito"]},
                "title": {"type": "string", "description": "Título breve."},
                "text":  {"type": "string", "description": "Contenido COMPLETO y literal del log. No resumas ni trunques; no agregues marcadores tipo '[...]' ni 'ver detalles completos'. Incluí el texto íntegro con todo el contexto."},
                "links": {"type": "array", "items": {"type": "string"},
                          "description": "Conexiones opcionales, ej: 'DEV→Backend'."},
            },
            "required": ["type", "text"]
        }
    }},
]


# Índice por nombre para armar subconjuntos sin recorrer la lista entera.
GROQ_TOOLS_BY_NAME = {t["function"]["name"]: t for t in GROQ_TOOLS}

# Grupos de herramientas por tipo de acción. Las de lectura son baratas (~330 tk juntas) y
# SEGURAS (idempotentes): se incluyen SIEMPRE, porque un falso negativo en lectura es lo peor
# que puede pasar — el modelo, sin la tool, inventa los datos en vez de consultarlos (alucina).
# Las escrituras/destructivas son los schemas pesados; ésas sí se gatean por intención para
# achicar el request contra el TPM de Groq. Un falso positivo de escritura sólo cuesta tokens;
# tool_choice='auto' hace que el modelo no la use si el mensaje no la pide.
_TOOLS_READ    = ["list_products", "list_tasks", "list_modules", "get_dev_metrics"]
_TOOLS_CREATE  = ["create_product_with_tasks", "add_tasks_to_product"]
_TOOLS_UPDATE  = ["update_tasks", "update_product", "move_tasks"]
_TOOLS_DELETE  = ["delete_tasks", "delete_product"]
_TOOLS_LOG     = ["create_log"]

# Disparadores normalizados (sin acentos, lower) para sumar los grupos de ESCRITURA. Ante la
# duda incluímos de más; sólo las destructivas exigen una señal explícita de borrado.
_KW_CREATE = ("crear", "creá", "crea ", "nuevo", "nueva", "agrega", "agregá", "añad",
              "sumar", "sumá", "volca", "volcá", "anota", "anotá", "necesito hacer",
              "tengo en la cabeza", "armar", "armá", "dar de alta")
_KW_UPDATE = ("actualiz", "cambia", "cambiá", "modific", "marca", "marcá", "mover", "mové",
              "mueve", "pasar a", "pasá a", "termin", "avanc", "complet", "hecho", "listo",
              "done", "doing", "prioridad", "impacto", "renombr", "pausar", "archivar")
_KW_DELETE = ("elimina", "eliminá", "borra", "borrá", "quita", "quitá", "elimin", "borrar",
              "remover", "remové", "deshacer")
_KW_LOG    = ("decid", "aprend", "insight", "riesgo", "oportunidad", "objetivo", "hipotesis",
              "hito", "registr", "log ", "bitacora", "me di cuenta", "anotar decision",
              "nota estrategica", "elegi", "elegí", "descart", "vamos con")


def _last_user(messages) -> str:
    """Texto del último mensaje del usuario (vacío si no hay)."""
    for m in reversed(messages):
        if m.get("role") == "user" and m.get("content"):
            return m["content"]
    return ""


# Señales de que el último mensaje es una ORDEN AUTÓNOMA: trae todo lo necesario para ejecutar
# la acción (un log con sus campos, un volcado de tareas) y NO depende del historial. En estos
# casos mandamos sólo ese mensaje: el historial es puro overhead que dispara el 413 de TPM.
_KW_SELFCONTAINED = ("tipo:", "titulo:", "título:", "descripcion:", "descripción:",
                     "conexiones:", "volca:", "volcá:", "anota:", "anotá:", "registra:",
                     "registrá:", "log tipo", "nuevo log")


def _is_self_contained(messages) -> bool:
    """True si el último mensaje del usuario es una orden completa que se ejecuta tal cual, sin
    necesitar la conversación previa (p.ej. un log con 'Tipo:/Título:/Descripción:/Conexiones:').
    El usuario lo confirmó: el log debe registrarse literal, el historial sólo estorba y desborda
    el TPM. Pedimos al menos 2 marcadores para no descartar historial por un 'tipo:' suelto."""
    txt = unicodedata.normalize('NFKD', _last_user(messages)).encode('ascii', 'ignore').decode().lower()
    hits = sum(1 for k in _KW_SELFCONTAINED
               if unicodedata.normalize('NFKD', k).encode('ascii', 'ignore').decode().lower() in txt)
    return hits >= 2


def _select_tools(messages):
    """Devuelve el subconjunto de GROQ_TOOLS para el último mensaje del usuario.
    Las lecturas van SIEMPRE (baratas y seguras: evitan que el modelo alucine datos que no
    pudo consultar). Las escrituras se suman por intención, para no inflar el request contra
    el límite de TPM de Groq. Nunca devuelve None: como mínimo van las 4 lecturas (~331 tk)."""
    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user" and m.get("content"):
            last_user = m["content"]
            break
    txt = unicodedata.normalize('NFKD', last_user).encode('ascii', 'ignore').decode().lower()

    def _strip(k):
        return unicodedata.normalize('NFKD', k).encode('ascii', 'ignore').decode().lower()

    def has(kws):
        # Normalizamos también las keywords: txt va sin acentos, así 'mové'/'creá' matchean.
        return any(_strip(k) in txt for k in kws)

    names = list(_TOOLS_READ)  # lecturas siempre
    if has(_KW_CREATE):
        names += _TOOLS_CREATE
    if has(_KW_UPDATE):
        names += _TOOLS_UPDATE
    if has(_KW_DELETE):
        names += _TOOLS_DELETE
    if has(_KW_LOG):
        names += _TOOLS_LOG

    # Dedup preservando orden, y materializar los schemas.
    seen, ordered = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            ordered.append(GROQ_TOOLS_BY_NAME[n])
    return ordered


def _log_tool_call(tool: str, args, result, finish_reason: str = None):
    """Audit trail liviano de cada tool call. Nunca debe romper el flujo del chat."""
    try:
        db.tool_logs.insert_one({
            "ts": datetime.utcnow(),
            "tool": tool,
            "args": args,
            "result": result,
            "ok": not (isinstance(result, dict) and "error" in result),
            "finish_reason": finish_reason,
        })
    except Exception:
        pass


def _norm_module(s: str) -> str:
    """Normaliza un nombre de módulo para comparar: sin acentos, lower, espacios colapsados."""
    s = unicodedata.normalize('NFKD', s or '').encode('ascii', 'ignore').decode()
    return ' '.join(s.lower().split())


def _resolve_module(product_id_str: str, requested: str, default: str = "General") -> str:
    """Si el módulo pedido matchea (normalizado) uno ya existente en el producto, devuelve el
    nombre canónico existente (evita duplicados por mayúsculas/acentos). Si ninguno encaja,
    devuelve el pedido tal cual — permite crear módulos nuevos."""
    name = (requested or default or "General").strip()
    try:
        existing = db.tasks.distinct("module", {"product_id": product_id_str})
    except Exception:
        existing = []
    nmap = {_norm_module(m): m for m in existing if m}
    return nmap.get(_norm_module(name), name)


def _insert_tasks(product_id_str: str, default_module: str, tasks: list) -> list:
    """Insert a list of task dicts under the given product_id. Returns created task summaries."""
    now = datetime.utcnow()
    created = []
    for t in tasks:
        module = _resolve_module(product_id_str, t.get("module") or default_module, "General")
        status = t.get("status", "todo")
        task_doc = {
            "product_id":  product_id_str,
            "module":      module,
            "name":        t.get("name"),
            "description": t.get("description", ""),
            "status":      status,
            "priority":    t.get("priority", "medio"),
            "impact":      t.get("impact", "medio"),
            "created_at":  now,
            "updated_at":  now,
        }
        apply_completion(task_doc, status, now)
        result = db.tasks.insert_one(task_doc)
        created.append({"id": str(result.inserted_id), "name": task_doc["name"]})
    return created


def execute_tool(name: str, args: dict):
    """Execute a tool call. All tools are self-contained — no ID passing needed."""
    try:
        # ── create_product_with_tasks ──────────────────────────────────────────
        if name == "create_product_with_tasks":
            # Validación previa: no escribir en DB con datos incompletos (evita productos vacíos).
            if not (args.get("product_name") or "").strip():
                return {"error": "Falta el nombre del producto."}
            tasks_raw = args.get("tasks", [])
            if not isinstance(tasks_raw, list) or not tasks_raw:
                return {"error": "Hay que crear al menos una tarea junto con el producto."}
            if any(not (t.get("name") or "").strip() for t in tasks_raw):
                return {"error": "Todas las tareas deben tener nombre."}
            now = datetime.utcnow()
            product = {
                "name":        args.get("product_name"),
                "icon":        args.get("product_icon", "📦"),
                "description": args.get("product_description", ""),
                "status":      args.get("product_status", "activo"),
                "color":       args.get("product_color", "#00e5ff"),
                "sort_order":  db.products.count_documents({}) + 1,
                "created_at":  now,
                "updated_at":  now,
            }
            prod_result = db.products.insert_one(product)
            product_id_str = str(prod_result.inserted_id)

            default_module = args.get("module", "General")
            created_tasks = _insert_tasks(product_id_str, default_module, tasks_raw)

            return {
                "product": {"id": product_id_str, "name": product["name"]},
                "tasks_created": len(created_tasks),
                "tasks": created_tasks,
            }

        # ── add_tasks_to_product ───────────────────────────────────────────────
        elif name == "add_tasks_to_product":
            q = args.get("product_name", "")
            product = db.products.find_one({"name": {"$regex": q, "$options": "i"}})
            if not product:
                return {"error": f"No se encontró el producto '{q}'. Verificá el nombre o crealo primero."}
            product_id_str = str(product["_id"])

            tasks_raw = args.get("tasks", [])
            if not isinstance(tasks_raw, list) or not tasks_raw:
                return {"error": "No se especificó ninguna tarea para agregar."}
            if any(not (t.get("name") or "").strip() for t in tasks_raw):
                return {"error": "Todas las tareas deben tener nombre."}
            default_module = args.get("module", "General")
            created_tasks = _insert_tasks(product_id_str, default_module, tasks_raw)

            return {
                "product": {"id": product_id_str, "name": product["name"]},
                "tasks_created": len(created_tasks),
                "tasks": created_tasks,
            }

        # ── list_products ──────────────────────────────────────────────────────
        elif name == "list_products":
            products = list(db.products.find().sort("sort_order", 1))
            return [{"id": str(p["_id"]), "name": p["name"], "status": p.get("status", "")} for p in products]

        # ── list_tasks ─────────────────────────────────────────────────────────
        elif name == "list_tasks":
            q = args.get("product_name", "")
            q_module = (args.get("module") or "").strip()
            query = {}
            product = None
            if q:
                product = db.products.find_one({"name": {"$regex": q, "$options": "i"}})
                if not product:
                    # Avisar explícitamente en vez de devolver lista vacía (que el modelo
                    # podría malinterpretar como "no hay tareas" e inventar una respuesta).
                    return {"error": f"No se encontró el producto '{q}'. Usá list_products "
                                     f"para ver los nombres exactos."}
                query["product_id"] = str(product["_id"])

            # Filtro por módulo en servidor: comparación normalizada (sin acentos, lower),
            # así 'IA' = 'ia', 'Backend' = 'backend'. Evita traer todo y filtrar a mano.
            if q_module:
                norm = _norm_module(q_module)
                scope = {"product_id": query["product_id"]} if "product_id" in query else {}
                existing = [m for m in db.tasks.distinct("module", scope) if m]
                matches = [m for m in existing if _norm_module(m) == norm]
                if not matches:
                    return {"error": f"No se encontró el módulo '{q_module}'"
                                     + (f" en '{product['name']}'" if product else "")
                                     + ". Módulos disponibles: "
                                     + (", ".join(sorted(existing)) or "ninguno") + "."}
                query["module"] = {"$in": matches}

            LIMIT = 200
            total = db.tasks.count_documents(query)
            tasks = list(db.tasks.find(query).limit(LIMIT))
            return {
                "total": total,
                "shown": len(tasks),
                "truncated": total > len(tasks),  # si True, avisá que hay más de las mostradas
                "tasks": [{
                    "id": str(t["_id"]), "name": t["name"],
                    "status": t.get("status"), "module": t.get("module"),
                    "priority": t.get("priority"),
                } for t in tasks],
            }

        # ── list_modules ───────────────────────────────────────────────────────
        elif name == "list_modules":
            q = args.get("product_name", "")
            product = db.products.find_one({"name": {"$regex": q, "$options": "i"}})
            if not product:
                return {"error": f"No se encontró el producto '{q}'."}
            mods = [m for m in db.tasks.distinct("module", {"product_id": str(product["_id"])}) if m]
            return {"product": product["name"], "modules": sorted(mods)}

        # ── update_tasks ───────────────────────────────────────────────────────
        elif name == "update_tasks":
            q_prod = args.get("product_name", "")
            product_id_str = None
            if q_prod:
                product = db.products.find_one({"name": {"$regex": q_prod, "$options": "i"}})
                if product:
                    product_id_str = str(product["_id"])

            results = []
            for upd in args.get("updates", []):
                task_name = upd.pop("task_name", "")
                query = {"name": {"$regex": task_name, "$options": "i"}}
                if product_id_str:
                    query["product_id"] = product_id_str
                now = datetime.utcnow()
                fields = {k: v for k, v in upd.items() if v is not None}
                if "status" in fields:
                    apply_completion(fields, fields["status"], now)
                fields["updated_at"] = now
                res = db.tasks.find_one_and_update(
                    query, {"$set": fields}, return_document=True)
                results.append({
                    "task": task_name,
                    "updated": bool(res),
                    "name": res.get("name") if res else None,
                })
            return {"results": results}

        # ── update_product ─────────────────────────────────────────────────────
        elif name == "update_product":
            q = args.get("product_name", "")
            product = db.products.find_one({"name": {"$regex": q, "$options": "i"}})
            if not product:
                return {"error": f"No se encontró el producto '{q}'."}
            fields = {}
            if args.get("new_name"):    fields["name"] = args["new_name"]
            if args.get("icon"):        fields["icon"] = args["icon"]
            if args.get("description") is not None: fields["description"] = args["description"]
            if args.get("status"):      fields["status"] = args["status"]
            if args.get("color"):       fields["color"] = args["color"]
            if not fields:
                return {"error": "No se especificó ningún cambio."}
            fields["updated_at"] = datetime.utcnow()
            res = db.products.find_one_and_update(
                {"_id": product["_id"]}, {"$set": fields}, return_document=True)
            return {"updated": True, "product": res.get("name"),
                    "changed": [k for k in fields if k != "updated_at"]}

        # ── move_tasks ─────────────────────────────────────────────────────────
        elif name == "move_tasks":
            dest = db.products.find_one({"name": {"$regex": args.get("to_product", ""), "$options": "i"}})
            if not dest:
                return {"error": f"No se encontró el producto destino '{args.get('to_product','')}'."}
            dest_id = str(dest["_id"])
            from_id = None
            if args.get("from_product"):
                src = db.products.find_one({"name": {"$regex": args["from_product"], "$options": "i"}})
                if src:
                    from_id = str(src["_id"])
            new_module = args.get("new_module")
            now = datetime.utcnow()
            results = []
            for tn in args.get("task_names", []):
                query = {"name": {"$regex": tn, "$options": "i"}}
                if from_id:
                    query["product_id"] = from_id
                fields = {"product_id": dest_id, "updated_at": now}
                if new_module:
                    fields["module"] = new_module
                res = db.tasks.find_one_and_update(
                    query, {"$set": fields}, return_document=True)
                results.append({"task": tn, "moved": bool(res),
                                "name": res.get("name") if res else None})
            return {"to_product": dest.get("name"), "new_module": new_module, "results": results}

        # ── delete_tasks ───────────────────────────────────────────────────────
        elif name == "delete_tasks":
            product_id_str = None
            if args.get("product_name"):
                product = db.products.find_one({"name": {"$regex": args["product_name"], "$options": "i"}})
                if product:
                    product_id_str = str(product["_id"])
            results = []
            for tn in args.get("task_names", []):
                query = {"name": {"$regex": tn, "$options": "i"}}
                if product_id_str:
                    query["product_id"] = product_id_str
                target = db.tasks.find_one(query)
                if target:
                    db.tasks.delete_one({"_id": target["_id"]})
                    results.append({"task": tn, "deleted": True, "name": target.get("name")})
                else:
                    results.append({"task": tn, "deleted": False})
            return {"results": results}

        # ── delete_product ─────────────────────────────────────────────────────
        elif name == "delete_product":
            q = args.get("product_name", "")
            product = db.products.find_one({"name": {"$regex": q, "$options": "i"}})
            if not product:
                return {"error": f"No se encontró el producto '{q}'."}
            pid = str(product["_id"])
            tasks_deleted = db.tasks.delete_many({"product_id": pid}).deleted_count
            db.products.delete_one({"_id": product["_id"]})
            return {"deleted": True, "product": product.get("name"),
                    "tasks_deleted": tasks_deleted}

        # ── get_dev_metrics ────────────────────────────────────────────────────
        elif name == "get_dev_metrics":
            m = compute_dev_metrics()
            # Resumen compacto para el LLM (sin el detalle diario verboso)
            return {
                "products_total": m["products_total"],
                "products_active": m["products_active"],
                "tasks_total": m["tasks_total"],
                "tasks_todo": m["tasks_todo"],
                "tasks_doing": m["tasks_doing"],
                "tasks_done": m["tasks_done"],
                "pct_done": m["pct_done"],
                "completed_this_week": m["week_total"],
                "per_product": [
                    {"name": p["name"], "done": p["done"], "total": p["total"], "pct": p["pct"]}
                    for p in m["per_product"]
                ],
            }

        # ── create_log ─────────────────────────────────────────────────────────
        elif name == "create_log":
            if not (args.get("text") or "").strip():
                return {"error": "El log necesita contenido (text) no vacío."}
            now = datetime.utcnow()
            log = {
                "type":  args.get("type", "Insight"),
                "title": args.get("title", ""),
                "text":  args.get("text", ""),
                "links": args.get("links", []),
                "date":  datetime.now().strftime("%Y-%m-%d"),
                "created_at": now,
            }
            result = db.strategy_logs.insert_one(log)
            return {"created": True, "id": str(result.inserted_id),
                    "type": log["type"], "title": log["title"]}

        return {"error": f"Herramienta desconocida: {name}"}

    except Exception as e:
        return {"error": str(e)}


def _is_transient_groq_error(e) -> bool:
    """True si conviene reintentar: rate-limit (429), errores de servidor (5xx) o de red.
    Los errores de cliente (400/401/404, etc.) NO son transitorios: reintentar no ayuda."""
    status = getattr(e, "status_code", None) or getattr(e, "status", None)
    if status is None:
        # Sin status (timeouts, conexión cortada, DNS) → tratar como transitorio.
        return True
    return status == 429 or 500 <= status < 600


def _is_413(e) -> bool:
    """True si el error de Groq es 413 'request too large' (TPM excedido). Lo distinguimos del
    429 porque el 413 NO se arregla esperando: hay que achicar el request y reintentar."""
    status = getattr(e, "status_code", None) or getattr(e, "status", None)
    if status == 413:
        return True
    # Algunos SDK no exponen status; caemos al texto del error.
    return "request too large" in str(e).lower() or "413" in str(e)[:8]


def _toklen(s: str) -> int:
    """Estimación de tokens de un string. Medimos sobre BYTES UTF-8, no chars: las flechas '→' y
    los acentos de los logs son multibyte (2-3 bytes) y Groq los cuenta como ~1 token cada uno.
    Contar chars//3 los subestimaba y por eso pasaban requests que igual reventaban en 413."""
    return len((s or "").encode('utf-8')) // 3


def _estimate_tokens(messages, tools=None) -> int:
    """Estimación barata de tokens del request (no tokeniza igual que Groq, pero alcanza para
    decidir cuánto recortar). Tiramos a SOBREESTIMAR: preferimos recortar de más a comer un 413."""
    total = 8  # overhead base del request
    for m in messages:
        total += 4  # role + framing por mensaje
        total += _toklen(m.get("content") or "")
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {})
            total += _toklen(fn.get("name", "")) + _toklen(fn.get("arguments", "")) + 8
    for t in (tools or []):
        # El schema de cada tool va serializado en el request; lo medimos sobre su JSON.
        total += _toklen(json.dumps(t, ensure_ascii=False))
    return total


def _fit_messages(system_msg, history, tools, max_tokens):
    """Recorta los turnos más viejos del historial hasta que (estimado de entrada + max_tokens)
    entre bajo el techo de TPM con margen. SIEMPRE conserva el system prompt y el último mensaje
    del usuario (sin ellos no hay nada que responder). Devuelve (lista_de_mensajes, dropped)."""
    budget = GROQ_TPM_LIMIT - GROQ_TPM_MARGIN - max_tokens
    # Conservamos el system y el último user pase lo que pase; recortamos del medio hacia atrás.
    last_user_idx = next((i for i in range(len(history) - 1, -1, -1)
                          if history[i].get("role") == "user"), None)
    dropped = 0
    while True:
        msgs = [system_msg] + history
        if _estimate_tokens(msgs, tools) <= budget:
            return msgs, dropped
        # Buscar el turno más viejo que se pueda tirar (no el último user, no el system).
        drop_at = next((i for i in range(len(history))
                        if i != last_user_idx), None)
        if drop_at is None:
            # Ya no queda nada que recortar salvo system + último user: devolvemos lo mínimo.
            return [system_msg] + ([history[last_user_idx]] if last_user_idx is not None else []), dropped
        history = history[:drop_at] + history[drop_at + 1:]
        if last_user_idx is not None and drop_at < last_user_idx:
            last_user_idx -= 1
        dropped += 1


def _groq_retry_after(e):
    """Segundos a esperar según el header 'retry-after' del 429, si viene. None si no."""
    resp = getattr(e, "response", None)
    headers = getattr(resp, "headers", None) or {}
    val = headers.get("retry-after") or headers.get("Retry-After")
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _groq_create(**kwargs):
    """Llama a Groq con backoff exponencial + jitter ante fallos transitorios.
    Reintenta hasta GROQ_MAX_RETRIES; re-lanza de inmediato los errores no transitorios.
    En rate-limit (429) respeta el header 'retry-after' de Groq si está presente."""

    # Compresión de contexto justo antes de pegarle a Groq. El try/except es a propósito: si la
    # compresión falla por cualquier motivo, seguimos con los mensajes sin comprimir en vez de
    # romper el chat. NUNCA debe ser un punto de falla del chatbot.
    if HEADROOM_ENABLED and kwargs.get('messages'):
        try:
            result = hr_compress(
                kwargs['messages'],
                model=HEADROOM_TOKENIZER_MODEL,  # solo para contar tokens; el LLM real es GROQ_MODEL
                model_limit=131072,              # ventana de contexto de qwen3-32b en Groq
                compress_user_messages=False,    # no tocar lo que escribe el usuario, solo tool outputs
            )
            kwargs['messages'] = result.messages
            if result.tokens_saved > 0:
                print(f"[headroom] {result.tokens_saved} tokens ahorrados "
                      f"({result.compression_ratio:.0%}) — {result.transforms_applied}")
        except Exception as e:
            print(f"[headroom] compresión falló, sigo sin comprimir: {e}")

    last_exc = None
    for attempt in range(GROQ_MAX_RETRIES + 1):
        try:
            return groq_client.chat.completions.create(**kwargs)
        except Exception as e:
            last_exc = e
            if attempt >= GROQ_MAX_RETRIES or not _is_transient_groq_error(e):
                raise
            # Groq nos dice cuánto esperar en un 429; respetarlo evita reintentos que rebotan.
            retry_after = _groq_retry_after(e)
            if retry_after is not None:
                delay = min(retry_after, 10) + random.uniform(0, 0.25)
            else:
                # backoff exponencial (0.5, 1, 2, …) + jitter para evitar thundering herd
                delay = 0.5 * (2 ** attempt) + random.uniform(0, 0.25)
            time.sleep(delay)
    raise last_exc  # inalcanzable, por completitud


@app.route('/api/chat', methods=['POST'])
def chat():
    if not groq_client:
        msg = ('Chatbot no disponible: configurá GROQ_API_KEY en Render.'
               if GroqClient else
               'Chatbot no disponible: instalá la librería groq (pip install groq).')
        return jsonify({'response': msg, 'refresh': False})

    data = request.json or {}
    messages = data.get('messages', [])

    system_msg = {
        "role": "system",
        "content": (
            "Sos AETHY, el asistente IA de OpenAETH CommandCenter. Gestionás el módulo DEV "
            "(productos y tareas) y respondés preguntas sobre el avance de la desarrolladora. "
            "Respondé siempre en español, en tono claro y conciso.\n\n"
            "HERRAMIENTAS DISPONIBLES:\n"
            "• Crear: create_product_with_tasks (producto + sus tareas en una sola llamada), "
            "add_tasks_to_product (tareas a un producto existente).\n"
            "• Modificar: update_product (nombre/estado/ícono/color), update_tasks (estado, "
            "prioridad, impacto, etc.), move_tasks (mover tareas a otro producto/módulo).\n"
            "• Eliminar: delete_tasks, delete_product (destructivas).\n"
            "• Consultar: list_products, list_tasks, list_modules (módulos de un producto), "
            "get_dev_metrics (avance, % completado, tareas por estado, esta semana, progreso por producto).\n"
            "• Estrategia: create_log (registrar decisión/insight/riesgo/oportunidad/"
            "aprendizaje/objetivo/hipótesis/hito).\n\n"
            "FORMULARIOS Y SUS CAMPOS (mapeá lo que dice el usuario a estos campos; en los "
            "desplegables elegí SIEMPRE uno de los valores listados, traduciendo del lenguaje "
            "natural; los campos de texto son libres):\n"
            "• PRODUCTO (create_product_with_tasks / update_product):\n"
            "   - product_name: texto libre (requerido).\n"
            "   - product_icon: un emoji (ej: 🚀). Si no lo dan, dejá 📦.\n"
            "   - product_description: texto libre.\n"
            "   - product_status [desplegable]: activo | idea | pausado | archivado. "
            "('en pausa'→pausado, 'guardado/terminado'→archivado, 'sólo una idea'→idea).\n"
            "   - product_color [desplegable, hex]: #00e5ff (cyan) | #39ff14 (verde) | "
            "#ff9f43 (naranja) | #a87fff (púrpura) | #ff6b35 (rojo-naranja) | #ffd700 "
            "(amarillo) | #ff3e5e (rojo). Traducí el color por nombre al hex.\n"
            "• TAREA (dentro de create_product_with_tasks/add_tasks_to_product, o update_tasks):\n"
            "   - name: texto libre (requerido).\n"
            "   - description: texto libre.\n"
            "   - module: texto libre, pero reutilizá los módulos existentes del producto "
            "(ver regla 3).\n"
            "   - status [desplegable]: todo | doing | done. "
            "('pendiente/por hacer'→todo, 'en curso/haciendo/en progreso'→doing, "
            "'lista/hecha/completada'→done).\n"
            "   - priority [desplegable]: alto | medio | bajo (prioridad).\n"
            "   - impact [desplegable]: alto | medio | bajo (impacto).\n"
            "• LOG ESTRATÉGICO (create_log):\n"
            "   - type [desplegable]: Decision | Insight | Riesgo | Oportunidad | Aprendizaje "
            "| Objetivo | Hipótesis | Hito. Inferí el tipo por el contenido "
            "('decidí'→Decision, 'me di cuenta/aprendí'→Aprendizaje/Insight, "
            "'meta/quiero lograr'→Objetivo, 'creo que/probemos si'→Hipótesis, "
            "'logramos/alcanzamos'→Hito, 'peligro/podría fallar'→Riesgo).\n"
            "   - title: texto libre breve.\n"
            "   - text: texto libre, COMPLETO y literal (requerido).\n"
            "   - links: lista de strings tipo 'DEV→Backend', 'Campaña→Landing'.\n"
            "Si un campo desplegable es ambiguo y no podés inferirlo con confianza, preguntá "
            "antes de asumir; no inventes valores fuera de las listas.\n\n"
            "ARTEFACTOS DE CONVERSACIÓN (patrones recurrentes; reconocelos por INTENCIÓN, sin "
            "exigir sintaxis especial; aplicá uno sólo si el match es claro, si no tratá el "
            "mensaje como pedido normal):\n"
            "• VOLCADO — el usuario vuelca varias cosas de la cabeza y vos las estructurás en "
            "tareas. Gatillos: 'tengo en la cabeza…', 'anotá esto', 'volcá:', 'necesito hacer: "
            "a, b, c'. Acción: partí el texto en N tareas; si nombra un producto existente usá "
            "add_tasks_to_product, si es nuevo create_product_with_tasks; inferí módulo "
            "(reutilizando los existentes), prioridad e impacto. Preguntá SÓLO si no queda "
            "claro a qué producto van. Respondé con el resultado limpio: cuántas tareas y dónde.\n"
            "• DECISIÓN — el usuario comunica una decisión. Gatillos: 'decidí…', 'elegí X sobre "
            "Y', 'vamos con…', 'descartamos…'. Acción: create_log type=Decision con el texto "
            "COMPLETO y literal (incluí el porqué y lo descartado si los menciona); si refiere a "
            "un producto/módulo agregá links ('DEV→Backend'). Respondé confirmando brevemente.\n"
            "• BITÁCORA DEL DÍA — cierre de jornada con varias acciones a la vez. Gatillos: 'hoy "
            "avancé en…', 'lo de hoy:', 'resumen del día', 'me trabé con…'. Mapeo: 'avancé/"
            "terminé X'→update_tasks (doing/done); 'me trabé con Y'→log Riesgo; 'aprendí Z'→log "
            "Aprendizaje. IMPORTANTE: este artefacto es compuesto y puede marcar tareas como "
            "done, así que NO ejecutes de una: primero PRESENTÁ el plan ('voy a: marcar X done, "
            "crear log Y…') y esperá la confirmación del usuario; recién con su OK ejecutás. "
            "Preguntá si una tarea mencionada no matchea ninguna existente.\n\n"
            "REGLAS:\n"
            "1. Para crear producto + tareas usá SIEMPRE create_product_with_tasks (nunca por separado).\n"
            "2. Buscás productos y tareas por nombre; no inventes IDs.\n"
            "3. Antes de agregar tareas a un producto existente, consultá sus módulos con "
            "list_modules y REUTILIZÁ los que ya existen (no crees variantes por mayúsculas o "
            "acentos: 'Backend' y 'backend' son el mismo). Creá un módulo nuevo sólo si el "
            "usuario lo pide explícitamente o si ninguno encaja.\n"
            "4. Para preguntas de avance/estado/reportes usá get_dev_metrics, no adivines números.\n"
            "5. Las acciones destructivas (delete_*) sólo si el usuario lo pide explícitamente; "
            "si hay ambigüedad, preguntá antes.\n"
            "6. Confirmá siempre con un resumen breve de lo que hiciste.\n"
            "7. Al usar create_log, guardá el texto COMPLETO y literal: nunca lo resumas, "
            "trunques ni agregues marcadores como '[...]' o 'ver detalles completos'.\n"
            "8. Ejecutá UNA herramienta por paso (Sync 1×1): pedí una sola tool, esperá su "
            "resultado y recién entonces decidí la siguiente. No agrupes varias tool_calls en "
            "un mismo turno; si una operación necesita varios pasos, hacelos de a uno.\n"
            "9. NUNCA inventes datos. Para listar productos/tareas, métricas o estados SIEMPRE "
            "llamá la herramienta correspondiente (list_products, list_tasks, get_dev_metrics) "
            "y respondé SÓLO con lo que devuelva. No anuncies 'Ejecutando…' ni muestres un "
            "'Resultado:' sin haber llamado realmente la tool. Si la herramienta no está "
            "disponible o falla, decílo con franqueza; jamás fabriques nombres ni números."
        )
    }

    # Rebuild conversation history — include tool messages from this request only
    # (client sends only user/assistant content turns)
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]

    did_tool_calls = False

    # Subconjunto de tools relevante al pedido: baja el tamaño del request para no chocar con
    # el límite de TPM de Groq. None = mensaje conversacional, no mandamos tools.
    active_tools = _select_tools(messages)

    # Órdenes autónomas (un log con sus campos, un volcado): el mensaje trae TODO lo necesario y
    # debe registrarse literal. El historial no aporta y es justo lo que desbordaba el TPM (el
    # usuario confirmó que al vaciar el chat dejaba de fallar). Para estas mandamos sólo el último
    # mensaje del usuario; para el resto, recorte preventivo de los turnos más viejos hasta entrar.
    if _is_self_contained(messages):
        last = _last_user(messages)
        history = [{"role": "user", "content": last}] if last else []

    all_messages, _dropped = _fit_messages(system_msg, history, active_tools, GROQ_MAX_TOKENS)

    for step in range(8):
        # Pacing entre pasos del Sync 1×1: separa los requests dentro de la ventana de TPM.
        if step > 0 and GROQ_THROTTLE_MS > 0:
            time.sleep(GROQ_THROTTLE_MS / 1000.0)

        # Budget de salida adaptado a lo que ya ocupa la entrada: a medida que se acumulan
        # resultados de tools el input crece, así que bajamos max_tokens (sin pasar del piso)
        # para que input+max_tokens siga entrando bajo el techo de TPM y no dispare un 413.
        in_est = _estimate_tokens(all_messages, active_tools)
        step_max = max(GROQ_MIN_TOKENS,
                       min(GROQ_MAX_TOKENS, GROQ_TPM_LIMIT - GROQ_TPM_MARGIN - in_est))

        try:
            response = None
            # Reintento defensivo ante 413: si pese al recorte previo Groq todavía considera el
            # request demasiado grande, achicamos (más historial fuera + menos max_tokens) y
            # reintentamos en vez de devolverle el error crudo al usuario.
            for shrink in range(3):
                create_kwargs = dict(
                    model=GROQ_MODEL,
                    messages=all_messages,
                    max_tokens=step_max,
                    reasoning_format="hidden",
                )
                if active_tools:
                    create_kwargs["tools"] = active_tools
                    create_kwargs["tool_choice"] = "auto"
                    # Una sola tool por respuesta, forzado por la API (no depende de que el
                    # modelo respete la regla 8). No cuesta tokens de prompt.
                    create_kwargs["parallel_tool_calls"] = False
                try:
                    response = _groq_create(**create_kwargs)
                    break
                except Exception as e:
                    if not _is_413(e) or shrink == 2:
                        raise
                    # Achicamos para el próximo intento: primero soltamos más historial, y si
                    # ya no hay historial que soltar, recortamos el budget de salida.
                    all_messages, _d = _fit_messages(
                        system_msg, all_messages[1:], active_tools,
                        step_max + max(400, step_max // 3))
                    step_max = max(GROQ_MIN_TOKENS, step_max - max(400, step_max // 3))
        except Exception as e:
            if did_tool_calls:
                # La(s) tool(s) de este turno ya se ejecutaron con éxito; el error es sólo del
                # paso de redactar la confirmación. No es justo mostrarle "Error" al usuario por
                # algo que sí se guardó.
                return jsonify({
                    'response': '✅ Listo, se guardó. (No pude redactar la confirmación por '
                                'límite de uso de Groq — si no ves el cambio reflejado, '
                                'refrescá la página.)',
                    'refresh': True
                })
            return jsonify({'response': f'Error Groq: {str(e)}', 'refresh': did_tool_calls})

        choice = response.choices[0]
        message = choice.message
        finish_reason = choice.finish_reason

        # Defensa por si el modelo igual manda dos tools (no debería con parallel_tool_calls=False):
        # nos quedamos con la primera para sostener el Sync 1×1.
        if message.tool_calls and len(message.tool_calls) > 1:
            message.tool_calls = message.tool_calls[:1]

        # Append assistant message (with tool_calls if present)
        asst_msg = {"role": "assistant", "content": strip_think(message.content) or ""}
        if message.tool_calls:
            asst_msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in message.tool_calls
            ]
        all_messages.append(asst_msg)

        if finish_reason == "tool_calls" and message.tool_calls:
            did_tool_calls = True
            tool_results = []
            for tc in message.tool_calls:
                try:
                    tool_args = json.loads(tc.function.arguments)
                except Exception:
                    # JSON de argumentos truncado/inválido (suele pasar al cortar por límite
                    # de tokens). NO ejecutamos con {}: eso crea productos vacíos. Devolvemos
                    # el error al modelo para que reintente fragmentando la operación.
                    err = {"error": "Argumentos truncados o inválidos. Reintentá la operación "
                                     "dividiéndola en llamadas más chicas (menos tareas por vez)."}
                    all_messages.append({
                        "role": "tool", "tool_call_id": tc.id,
                        "content": json.dumps(err, ensure_ascii=False)
                    })
                    _log_tool_call(tc.function.name, None, err, finish_reason)
                    continue
                result = execute_tool(tc.function.name, tool_args)
                _log_tool_call(tc.function.name, tool_args, result, finish_reason)
                tool_results.append((tc.function.name, result))
                all_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False, default=str)
                })

            # Atajo determinístico: create_log solo y sin error → confirmamos en Python y NO
            # volvemos a llamar a Groq. Elimina la segunda llamada que causaba el bug reportado
            # (el log se guardaba pero el paso de wrap-up chocaba con el TPM y mostraba "Error").
            if (len(tool_results) == 1 and tool_results[0][0] == "create_log"
                    and "error" not in tool_results[0][1]):
                r = tool_results[0][1]
                titulo = f": «{r['title']}»" if r.get("title") else "."
                return jsonify({
                    "response": f"✅ Log de tipo **{r.get('type', '')}** guardado{titulo}",
                    "refresh": True
                })
        elif finish_reason == "length":
            # La respuesta se cortó por límite de tokens (no por terminar). Avisamos en vez de
            # devolver texto parcial silenciosamente. Preservamos lo ya creado vía refresh.
            partial = strip_think(message.content) or ""
            note = ("⚠️ La respuesta se truncó por longitud. "
                    "Si pediste varias tareas a la vez, probá dividirlo en partes más chicas.")
            return jsonify({
                "response": (partial + "\n\n" + note) if partial else note,
                "refresh": did_tool_calls
            })
        else:
            return jsonify({
                "response": strip_think(message.content) or "Acción completada.",
                "refresh": did_tool_calls
            })

    return jsonify({
        "response": "No pude completar la operación en los pasos permitidos.",
        "refresh": did_tool_calls
    })

# ============================================================================
# FRONTEND
# ============================================================================

HTML = r"""


<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>OpenAETH — Command Core</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script>/**
 * marked v15.0.0 - a markdown parser
 * Copyright (c) 2011-2024, Christopher Jeffrey. (MIT Licensed)
 * https://github.com/markedjs/marked
 */
!function(e,t){"object"==typeof exports&&"undefined"!=typeof module?t(exports):"function"==typeof define&&define.amd?define(["exports"],t):t((e="undefined"!=typeof globalThis?globalThis:e||self).marked={})}(this,(function(e){"use strict";function t(){return{async:!1,breaks:!1,extensions:null,gfm:!0,hooks:null,pedantic:!1,renderer:null,silent:!1,tokenizer:null,walkTokens:null}}function n(t){e.defaults=t}e.defaults={async:!1,breaks:!1,extensions:null,gfm:!0,hooks:null,pedantic:!1,renderer:null,silent:!1,tokenizer:null,walkTokens:null};const s={exec:()=>null};function r(e,t=""){let n="string"==typeof e?e:e.source;const s={replace:(e,t)=>{let r="string"==typeof t?t:t.source;return r=r.replace(i.caret,"$1"),n=n.replace(e,r),s},getRegex:()=>new RegExp(n,t)};return s}const i={codeRemoveIndent:/^(?: {1,4}| {0,3}\t)/gm,outputLinkReplace:/\\([\[\]])/g,indentCodeCompensation:/^(\s+)(?:```)/,beginningSpace:/^\s+/,endingHash:/#$/,startingSpaceChar:/^ /,endingSpaceChar:/ $/,nonSpaceChar:/[^ ]/,newLineCharGlobal:/\n/g,tabCharGlobal:/\t/g,multipleSpaceGlobal:/\s+/g,blankLine:/^[ \t]*$/,doubleBlankLine:/\n[ \t]*\n[ \t]*$/,blockquoteStart:/^ {0,3}>/,blockquoteSetextReplace:/\n {0,3}((?:=+|-+) *)(?=\n|$)/g,blockquoteSetextReplace2:/^ {0,3}>[ \t]?/gm,listReplaceTabs:/^\t+/,listReplaceNesting:/^ {1,4}(?=( {4})*[^ ])/g,listIsTask:/^\[[ xX]\] /,listReplaceTask:/^\[[ xX]\] +/,anyLine:/\n.*\n/,hrefBrackets:/^<(.*)>$/,tableDelimiter:/[:|]/,tableAlignChars:/^\||\| *$/g,tableRowBlankLine:/\n[ \t]*$/,tableAlignRight:/^ *-+: *$/,tableAlignCenter:/^ *:-+: *$/,tableAlignLeft:/^ *:-+ *$/,startATag:/^<a /i,endATag:/^<\/a>/i,startPreScriptTag:/^<(pre|code|kbd|script)(\s|>)/i,endPreScriptTag:/^<\/(pre|code|kbd|script)(\s|>)/i,startAngleBracket:/^</,endAngleBracket:/>$/,pedanticHrefTitle:/^([^'"]*[^\s])\s+(['"])(.*)\2/,unicodeAlphaNumeric:/[\p{L}\p{N}]/u,escapeTest:/[&<>"']/,escapeReplace:/[&<>"']/g,escapeTestNoEncode:/[<>"']|&(?!(#\d{1,7}|#[Xx][a-fA-F0-9]{1,6}|\w+);)/,escapeReplaceNoEncode:/[<>"']|&(?!(#\d{1,7}|#[Xx][a-fA-F0-9]{1,6}|\w+);)/g,unescapeTest:/&(#(?:\d+)|(?:#x[0-9A-Fa-f]+)|(?:\w+));?/gi,caret:/(^|[^\[])\^/g,percentDecode:/%25/g,findPipe:/\|/g,splitPipe:/ \|/,slashPipe:/\\\|/g,carriageReturn:/\r\n|\r/g,spaceLine:/^ +$/gm,notSpaceStart:/^\S*/,endingNewline:/\n$/,listItemRegex:e=>new RegExp(`^( {0,3}${e})((?:[\t ][^\\n]*)?(?:\\n|$))`),nextBulletRegex:e=>new RegExp(`^ {0,${Math.min(3,e-1)}}(?:[*+-]|\\d{1,9}[.)])((?:[ \t][^\\n]*)?(?:\\n|$))`),hrRegex:e=>new RegExp(`^ {0,${Math.min(3,e-1)}}((?:- *){3,}|(?:_ *){3,}|(?:\\* *){3,})(?:\\n+|$)`),fencesBeginRegex:e=>new RegExp(`^ {0,${Math.min(3,e-1)}}(?:\`\`\`|~~~)`),headingBeginRegex:e=>new RegExp(`^ {0,${Math.min(3,e-1)}}#`),htmlBeginRegex:e=>new RegExp(`^ {0,${Math.min(3,e-1)}}<(?:[a-z].*>|!--)`,"i")},l=/^ {0,3}((?:-[\t ]*){3,}|(?:_[ \t]*){3,}|(?:\*[ \t]*){3,})(?:\n+|$)/,o=/(?:[*+-]|\d{1,9}[.)])/,a=r(/^(?!bull |blockCode|fences|blockquote|heading|html)((?:.|\n(?!\s*?\n|bull |blockCode|fences|blockquote|heading|html))+?)\n {0,3}(=+|-+) *(?:\n+|$)/).replace(/bull/g,o).replace(/blockCode/g,/(?: {4}| {0,3}\t)/).replace(/fences/g,/ {0,3}(?:`{3,}|~{3,})/).replace(/blockquote/g,/ {0,3}>/).replace(/heading/g,/ {0,3}#{1,6}/).replace(/html/g,/ {0,3}<[^\n>]+>\n/).getRegex(),c=/^([^\n]+(?:\n(?!hr|heading|lheading|blockquote|fences|list|html|table| +\n)[^\n]+)*)/,h=/(?!\s*\])(?:\\.|[^\[\]\\])+/,p=r(/^ {0,3}\[(label)\]: *(?:\n[ \t]*)?([^<\s][^\s]*|<.*?>)(?:(?: +(?:\n[ \t]*)?| *\n[ \t]*)(title))? *(?:\n+|$)/).replace("label",h).replace("title",/(?:"(?:\\"?|[^"\\])*"|'[^'\n]*(?:\n[^'\n]+)*\n?'|\([^()]*\))/).getRegex(),u=r(/^( {0,3}bull)([ \t][^\n]+?)?(?:\n|$)/).replace(/bull/g,o).getRegex(),g="address|article|aside|base|basefont|blockquote|body|caption|center|col|colgroup|dd|details|dialog|dir|div|dl|dt|fieldset|figcaption|figure|footer|form|frame|frameset|h[1-6]|head|header|hr|html|iframe|legend|li|link|main|menu|menuitem|meta|nav|noframes|ol|optgroup|option|p|param|search|section|summary|table|tbody|td|tfoot|th|thead|title|tr|track|ul",k=/<!--(?:-?>|[\s\S]*?(?:-->|$))/,d=r("^ {0,3}(?:<(script|pre|style|textarea)[\\s>][\\s\\S]*?(?:</\\1>[^\\n]*\\n+|$)|comment[^\\n]*(\\n+|$)|<\\?[\\s\\S]*?(?:\\?>\\n*|$)|<![A-Z][\\s\\S]*?(?:>\\n*|$)|<!\\[CDATA\\[[\\s\\S]*?(?:\\]\\]>\\n*|$)|</?(tag)(?: +|\\n|/?>)[\\s\\S]*?(?:(?:\\n[ \t]*)+\\n|$)|<(?!script|pre|style|textarea)([a-z][\\w-]*)(?:attribute)*? */?>(?=[ \\t]*(?:\\n|$))[\\s\\S]*?(?:(?:\\n[ \t]*)+\\n|$)|</(?!script|pre|style|textarea)[a-z][\\w-]*\\s*>(?=[ \\t]*(?:\\n|$))[\\s\\S]*?(?:(?:\\n[ \t]*)+\\n|$))","i").replace("comment",k).replace("tag",g).replace("attribute",/ +[a-zA-Z:_][\w.:-]*(?: *= *"[^"\n]*"| *= *'[^'\n]*'| *= *[^\s"'=<>`]+)?/).getRegex(),f=r(c).replace("hr",l).replace("heading"," {0,3}#{1,6}(?:\\s|$)").replace("|lheading","").replace("|table","").replace("blockquote"," {0,3}>").replace("fences"," {0,3}(?:`{3,}(?=[^`\\n]*\\n)|~{3,})[^\\n]*\\n").replace("list"," {0,3}(?:[*+-]|1[.)]) ").replace("html","</?(?:tag)(?: +|\\n|/?>)|<(?:script|pre|style|textarea|!--)").replace("tag",g).getRegex(),x={blockquote:r(/^( {0,3}> ?(paragraph|[^\n]*)(?:\n|$))+/).replace("paragraph",f).getRegex(),code:/^((?: {4}| {0,3}\t)[^\n]+(?:\n(?:[ \t]*(?:\n|$))*)?)+/,def:p,fences:/^ {0,3}(`{3,}(?=[^`\n]*(?:\n|$))|~{3,})([^\n]*)(?:\n|$)(?:|([\s\S]*?)(?:\n|$))(?: {0,3}\1[~`]* *(?=\n|$)|$)/,heading:/^ {0,3}(#{1,6})(?=\s|$)(.*)(?:\n+|$)/,hr:l,html:d,lheading:a,list:u,newline:/^(?:[ \t]*(?:\n|$))+/,paragraph:f,table:s,text:/^[^\n]+/},b=r("^ *([^\\n ].*)\\n {0,3}((?:\\| *)?:?-+:? *(?:\\| *:?-+:? *)*(?:\\| *)?)(?:\\n((?:(?! *\\n|hr|heading|blockquote|code|fences|list|html).*(?:\\n|$))*)\\n*|$)").replace("hr",l).replace("heading"," {0,3}#{1,6}(?:\\s|$)").replace("blockquote"," {0,3}>").replace("code","(?: {4}| {0,3}\t)[^\\n]").replace("fences"," {0,3}(?:`{3,}(?=[^`\\n]*\\n)|~{3,})[^\\n]*\\n").replace("list"," {0,3}(?:[*+-]|1[.)]) ").replace("html","</?(?:tag)(?: +|\\n|/?>)|<(?:script|pre|style|textarea|!--)").replace("tag",g).getRegex(),m={...x,table:b,paragraph:r(c).replace("hr",l).replace("heading"," {0,3}#{1,6}(?:\\s|$)").replace("|lheading","").replace("table",b).replace("blockquote"," {0,3}>").replace("fences"," {0,3}(?:`{3,}(?=[^`\\n]*\\n)|~{3,})[^\\n]*\\n").replace("list"," {0,3}(?:[*+-]|1[.)]) ").replace("html","</?(?:tag)(?: +|\\n|/?>)|<(?:script|pre|style|textarea|!--)").replace("tag",g).getRegex()},w={...x,html:r("^ *(?:comment *(?:\\n|\\s*$)|<(tag)[\\s\\S]+?</\\1> *(?:\\n{2,}|\\s*$)|<tag(?:\"[^\"]*\"|'[^']*'|\\s[^'\"/>\\s]*)*?/?> *(?:\\n{2,}|\\s*$))").replace("comment",k).replace(/tag/g,"(?!(?:a|em|strong|small|s|cite|q|dfn|abbr|data|time|code|var|samp|kbd|sub|sup|i|b|u|mark|ruby|rt|rp|bdi|bdo|span|br|wbr|ins|del|img)\\b)\\w+(?!:|[^\\w\\s@]*@)\\b").getRegex(),def:/^ *\[([^\]]+)\]: *<?([^\s>]+)>?(?: +(["(][^\n]+[")]))? *(?:\n+|$)/,heading:/^(#{1,6})(.*)(?:\n+|$)/,fences:s,lheading:/^(.+?)\n {0,3}(=+|-+) *(?:\n+|$)/,paragraph:r(c).replace("hr",l).replace("heading"," *#{1,6} *[^\n]").replace("lheading",a).replace("|table","").replace("blockquote"," {0,3}>").replace("|fences","").replace("|list","").replace("|html","").replace("|tag","").getRegex()},y=/^\\([!"#$%&'()*+,\-./:;<=>?@\[\]\\^_`{|}~])/,$=/^( {2,}|\\)\n(?!\s*$)/,R="\\p{P}\\p{S}",T=r(/^((?![*_])[\spunctuation])/,"u").replace(/punctuation/g,R).getRegex(),z=r(/^(?:\*+(?:((?!\*)[punct])|[^\s*]))|^_+(?:((?!_)[punct])|([^\s_]))/,"u").replace(/punct/g,R).getRegex(),A=r("^[^_*]*?__[^_*]*?\\*[^_*]*?(?=__)|[^*]+(?=[^*])|(?!\\*)[punct](\\*+)(?=[\\s]|$)|[^punct\\s](\\*+)(?!\\*)(?=[punct\\s]|$)|(?!\\*)[punct\\s](\\*+)(?=[^punct\\s])|[\\s](\\*+)(?!\\*)(?=[punct])|(?!\\*)[punct](\\*+)(?!\\*)(?=[punct])|[^punct\\s](\\*+)(?=[^punct\\s])","gu").replace(/punct/g,R).getRegex(),S=r("^[^_*]*?\\*\\*[^_*]*?_[^_*]*?(?=\\*\\*)|[^_]+(?=[^_])|(?!_)[punct](_+)(?=[\\s]|$)|[^punct\\s](_+)(?!_)(?=[punct\\s]|$)|(?!_)[punct\\s](_+)(?=[^punct\\s])|[\\s](_+)(?!_)(?=[punct])|(?!_)[punct](_+)(?!_)(?=[punct])","gu").replace(/punct/g,R).getRegex(),_=r(/\\([punct])/,"gu").replace(/punct/g,R).getRegex(),I=r(/^<(scheme:[^\s\x00-\x1f<>]*|email)>/).replace("scheme",/[a-zA-Z][a-zA-Z0-9+.-]{1,31}/).replace("email",/[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+(@)[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+(?![-_])/).getRegex(),L=r(k).replace("(?:--\x3e|$)","--\x3e").getRegex(),B=r("^comment|^</[a-zA-Z][\\w:-]*\\s*>|^<[a-zA-Z][\\w-]*(?:attribute)*?\\s*/?>|^<\\?[\\s\\S]*?\\?>|^<![a-zA-Z]+\\s[\\s\\S]*?>|^<!\\[CDATA\\[[\\s\\S]*?\\]\\]>").replace("comment",L).replace("attribute",/\s+[a-zA-Z:_][\w.:-]*(?:\s*=\s*"[^"]*"|\s*=\s*'[^']*'|\s*=\s*[^\s"'=<>`]+)?/).getRegex(),C=/(?:\[(?:\\.|[^\[\]\\])*\]|\\.|`[^`]*`|[^\[\]\\`])*?/,E=r(/^!?\[(label)\]\(\s*(href)(?:\s+(title))?\s*\)/).replace("label",C).replace("href",/<(?:\\.|[^\n<>\\])+>|[^\s\x00-\x1f]*/).replace("title",/"(?:\\"?|[^"\\])*"|'(?:\\'?|[^'\\])*'|\((?:\\\)?|[^)\\])*\)/).getRegex(),q=r(/^!?\[(label)\]\[(ref)\]/).replace("label",C).replace("ref",h).getRegex(),P=r(/^!?\[(ref)\](?:\[\])?/).replace("ref",h).getRegex(),Z={_backpedal:s,anyPunctuation:_,autolink:I,blockSkip:/\[[^[\]]*?\]\((?:\\.|[^\\\(\)]|\((?:\\.|[^\\\(\)])*\))*\)|`[^`]*?`|<[^<>]*?>/g,br:$,code:/^(`+)([^`]|[^`][\s\S]*?[^`])\1(?!`)/,del:s,emStrongLDelim:z,emStrongRDelimAst:A,emStrongRDelimUnd:S,escape:y,link:E,nolink:P,punctuation:T,reflink:q,reflinkSearch:r("reflink|nolink(?!\\()","g").replace("reflink",q).replace("nolink",P).getRegex(),tag:B,text:/^(`+|[^`])(?:(?= {2,}\n)|[\s\S]*?(?:(?=[\\<!\[`*_]|\b_|$)|[^ ](?= {2,}\n)))/,url:s},v={...Z,link:r(/^!?\[(label)\]\((.*?)\)/).replace("label",C).getRegex(),reflink:r(/^!?\[(label)\]\s*\[([^\]]*)\]/).replace("label",C).getRegex()},Q={...Z,escape:r(y).replace("])","~|])").getRegex(),url:r(/^((?:ftp|https?):\/\/|www\.)(?:[a-zA-Z0-9\-]+\.?)+[^\s<]*|^email/,"i").replace("email",/[A-Za-z0-9._+-]+(@)[a-zA-Z0-9-_]+(?:\.[a-zA-Z0-9-_]*[a-zA-Z0-9])+(?![-_])/).getRegex(),_backpedal:/(?:[^?!.,:;*_'"~()&]+|\([^)]*\)|&(?![a-zA-Z0-9]+;$)|[?!.,:;*_'"~)]+(?!$))+/,del:/^(~~?)(?=[^\s~])((?:\\.|[^\\])*?(?:\\.|[^\s~\\]))\1(?=[^~]|$)/,text:/^([`~]+|[^`~])(?:(?= {2,}\n)|(?=[a-zA-Z0-9.!#$%&'*+\/=?_`{\|}~-]+@)|[\s\S]*?(?:(?=[\\<!\[`*~_]|\b_|https?:\/\/|ftp:\/\/|www\.|$)|[^ ](?= {2,}\n)|[^a-zA-Z0-9.!#$%&'*+\/=?_`{\|}~-](?=[a-zA-Z0-9.!#$%&'*+\/=?_`{\|}~-]+@)))/},D={...Q,br:r($).replace("{2,}","*").getRegex(),text:r(Q.text).replace("\\b_","\\b_| {2,}\\n").replace(/\{2,\}/g,"*").getRegex()},M={normal:x,gfm:m,pedantic:w},O={normal:Z,gfm:Q,breaks:D,pedantic:v},j={"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"},N=e=>j[e];function G(e,t){if(t){if(i.escapeTest.test(e))return e.replace(i.escapeReplace,N)}else if(i.escapeTestNoEncode.test(e))return e.replace(i.escapeReplaceNoEncode,N);return e}function H(e){try{e=encodeURI(e).replace(i.percentDecode,"%")}catch{return null}return e}function X(e,t){const n=e.replace(i.findPipe,((e,t,n)=>{let s=!1,r=t;for(;--r>=0&&"\\"===n[r];)s=!s;return s?"|":" |"})).split(i.splitPipe);let s=0;if(n[0].trim()||n.shift(),n.length>0&&!n[n.length-1].trim()&&n.pop(),t)if(n.length>t)n.splice(t);else for(;n.length<t;)n.push("");for(;s<n.length;s++)n[s]=n[s].trim().replace(i.slashPipe,"|");return n}function F(e,t,n){const s=e.length;if(0===s)return"";let r=0;for(;r<s;){const i=e.charAt(s-r-1);if(i!==t||n){if(i===t||!n)break;r++}else r++}return e.slice(0,s-r)}function U(e,t,n,s,r){const i=t.href,l=t.title||null,o=e[1].replace(r.other.outputLinkReplace,"$1");if("!"!==e[0].charAt(0)){s.state.inLink=!0;const e={type:"link",raw:n,href:i,title:l,text:o,tokens:s.inlineTokens(o)};return s.state.inLink=!1,e}return{type:"image",raw:n,href:i,title:l,text:o}}class J{options;rules;lexer;constructor(t){this.options=t||e.defaults}space(e){const t=this.rules.block.newline.exec(e);if(t&&t[0].length>0)return{type:"space",raw:t[0]}}code(e){const t=this.rules.block.code.exec(e);if(t){const e=t[0].replace(this.rules.other.codeRemoveIndent,"");return{type:"code",raw:t[0],codeBlockStyle:"indented",text:this.options.pedantic?e:F(e,"\n")}}}fences(e){const t=this.rules.block.fences.exec(e);if(t){const e=t[0],n=function(e,t,n){const s=e.match(n.other.indentCodeCompensation);if(null===s)return t;const r=s[1];return t.split("\n").map((e=>{const t=e.match(n.other.beginningSpace);if(null===t)return e;const[s]=t;return s.length>=r.length?e.slice(r.length):e})).join("\n")}(e,t[3]||"",this.rules);return{type:"code",raw:e,lang:t[2]?t[2].trim().replace(this.rules.inline.anyPunctuation,"$1"):t[2],text:n}}}heading(e){const t=this.rules.block.heading.exec(e);if(t){let e=t[2].trim();if(this.rules.other.endingHash.test(e)){const t=F(e,"#");this.options.pedantic?e=t.trim():t&&!this.rules.other.endingSpaceChar.test(t)||(e=t.trim())}return{type:"heading",raw:t[0],depth:t[1].length,text:e,tokens:this.lexer.inline(e)}}}hr(e){const t=this.rules.block.hr.exec(e);if(t)return{type:"hr",raw:F(t[0],"\n")}}blockquote(e){const t=this.rules.block.blockquote.exec(e);if(t){let e=F(t[0],"\n").split("\n"),n="",s="";const r=[];for(;e.length>0;){let t=!1;const i=[];let l;for(l=0;l<e.length;l++)if(this.rules.other.blockquoteStart.test(e[l]))i.push(e[l]),t=!0;else{if(t)break;i.push(e[l])}e=e.slice(l);const o=i.join("\n"),a=o.replace(this.rules.other.blockquoteSetextReplace,"\n    $1").replace(this.rules.other.blockquoteSetextReplace2,"");n=n?`${n}\n${o}`:o,s=s?`${s}\n${a}`:a;const c=this.lexer.state.top;if(this.lexer.state.top=!0,this.lexer.blockTokens(a,r,!0),this.lexer.state.top=c,0===e.length)break;const h=r[r.length-1];if("code"===h?.type)break;if("blockquote"===h?.type){const t=h,i=t.raw+"\n"+e.join("\n"),l=this.blockquote(i);r[r.length-1]=l,n=n.substring(0,n.length-t.raw.length)+l.raw,s=s.substring(0,s.length-t.text.length)+l.text;break}if("list"!==h?.type);else{const t=h,i=t.raw+"\n"+e.join("\n"),l=this.list(i);r[r.length-1]=l,n=n.substring(0,n.length-h.raw.length)+l.raw,s=s.substring(0,s.length-t.raw.length)+l.raw,e=i.substring(r[r.length-1].raw.length).split("\n")}}return{type:"blockquote",raw:n,tokens:r,text:s}}}list(e){let t=this.rules.block.list.exec(e);if(t){let n=t[1].trim();const s=n.length>1,r={type:"list",raw:"",ordered:s,start:s?+n.slice(0,-1):"",loose:!1,items:[]};n=s?`\\d{1,9}\\${n.slice(-1)}`:`\\${n}`,this.options.pedantic&&(n=s?n:"[*+-]");const i=this.rules.other.listItemRegex(n);let l=!1;for(;e;){let n=!1,s="",o="";if(!(t=i.exec(e)))break;if(this.rules.block.hr.test(e))break;s=t[0],e=e.substring(s.length);let a=t[2].split("\n",1)[0].replace(this.rules.other.listReplaceTabs,(e=>" ".repeat(3*e.length))),c=e.split("\n",1)[0],h=!a.trim(),p=0;if(this.options.pedantic?(p=2,o=a.trimStart()):h?p=t[1].length+1:(p=t[2].search(this.rules.other.nonSpaceChar),p=p>4?1:p,o=a.slice(p),p+=t[1].length),h&&this.rules.other.blankLine.test(c)&&(s+=c+"\n",e=e.substring(c.length+1),n=!0),!n){const t=this.rules.other.nextBulletRegex(p),n=this.rules.other.hrRegex(p),r=this.rules.other.fencesBeginRegex(p),i=this.rules.other.headingBeginRegex(p),l=this.rules.other.htmlBeginRegex(p);for(;e;){const u=e.split("\n",1)[0];let g;if(c=u,this.options.pedantic?(c=c.replace(this.rules.other.listReplaceNesting,"  "),g=c):g=c.replace(this.rules.other.tabCharGlobal,"    "),r.test(c))break;if(i.test(c))break;if(l.test(c))break;if(t.test(c))break;if(n.test(c))break;if(g.search(this.rules.other.nonSpaceChar)>=p||!c.trim())o+="\n"+g.slice(p);else{if(h)break;if(a.replace(this.rules.other.tabCharGlobal,"    ").search(this.rules.other.nonSpaceChar)>=4)break;if(r.test(a))break;if(i.test(a))break;if(n.test(a))break;o+="\n"+c}h||c.trim()||(h=!0),s+=u+"\n",e=e.substring(u.length+1),a=g.slice(p)}}r.loose||(l?r.loose=!0:this.rules.other.doubleBlankLine.test(s)&&(l=!0));let u,g=null;this.options.gfm&&(g=this.rules.other.listIsTask.exec(o),g&&(u="[ ] "!==g[0],o=o.replace(this.rules.other.listReplaceTask,""))),r.items.push({type:"list_item",raw:s,task:!!g,checked:u,loose:!1,text:o,tokens:[]}),r.raw+=s}r.items[r.items.length-1].raw=r.items[r.items.length-1].raw.trimEnd(),r.items[r.items.length-1].text=r.items[r.items.length-1].text.trimEnd(),r.raw=r.raw.trimEnd();for(let e=0;e<r.items.length;e++)if(this.lexer.state.top=!1,r.items[e].tokens=this.lexer.blockTokens(r.items[e].text,[]),!r.loose){const t=r.items[e].tokens.filter((e=>"space"===e.type)),n=t.length>0&&t.some((e=>this.rules.other.anyLine.test(e.raw)));r.loose=n}if(r.loose)for(let e=0;e<r.items.length;e++)r.items[e].loose=!0;return r}}html(e){const t=this.rules.block.html.exec(e);if(t){return{type:"html",block:!0,raw:t[0],pre:"pre"===t[1]||"script"===t[1]||"style"===t[1],text:t[0]}}}def(e){const t=this.rules.block.def.exec(e);if(t){const e=t[1].toLowerCase().replace(this.rules.other.multipleSpaceGlobal," "),n=t[2]?t[2].replace(this.rules.other.hrefBrackets,"$1").replace(this.rules.inline.anyPunctuation,"$1"):"",s=t[3]?t[3].substring(1,t[3].length-1).replace(this.rules.inline.anyPunctuation,"$1"):t[3];return{type:"def",tag:e,raw:t[0],href:n,title:s}}}table(e){const t=this.rules.block.table.exec(e);if(!t)return;if(!this.rules.other.tableDelimiter.test(t[2]))return;const n=X(t[1]),s=t[2].replace(this.rules.other.tableAlignChars,"").split("|"),r=t[3]&&t[3].trim()?t[3].replace(this.rules.other.tableRowBlankLine,"").split("\n"):[],i={type:"table",raw:t[0],header:[],align:[],rows:[]};if(n.length===s.length){for(const e of s)this.rules.other.tableAlignRight.test(e)?i.align.push("right"):this.rules.other.tableAlignCenter.test(e)?i.align.push("center"):this.rules.other.tableAlignLeft.test(e)?i.align.push("left"):i.align.push(null);for(let e=0;e<n.length;e++)i.header.push({text:n[e],tokens:this.lexer.inline(n[e]),header:!0,align:i.align[e]});for(const e of r)i.rows.push(X(e,i.header.length).map(((e,t)=>({text:e,tokens:this.lexer.inline(e),header:!1,align:i.align[t]}))));return i}}lheading(e){const t=this.rules.block.lheading.exec(e);if(t)return{type:"heading",raw:t[0],depth:"="===t[2].charAt(0)?1:2,text:t[1],tokens:this.lexer.inline(t[1])}}paragraph(e){const t=this.rules.block.paragraph.exec(e);if(t){const e="\n"===t[1].charAt(t[1].length-1)?t[1].slice(0,-1):t[1];return{type:"paragraph",raw:t[0],text:e,tokens:this.lexer.inline(e)}}}text(e){const t=this.rules.block.text.exec(e);if(t)return{type:"text",raw:t[0],text:t[0],tokens:this.lexer.inline(t[0])}}escape(e){const t=this.rules.inline.escape.exec(e);if(t)return{type:"escape",raw:t[0],text:t[1]}}tag(e){const t=this.rules.inline.tag.exec(e);if(t)return!this.lexer.state.inLink&&this.rules.other.startATag.test(t[0])?this.lexer.state.inLink=!0:this.lexer.state.inLink&&this.rules.other.endATag.test(t[0])&&(this.lexer.state.inLink=!1),!this.lexer.state.inRawBlock&&this.rules.other.startPreScriptTag.test(t[0])?this.lexer.state.inRawBlock=!0:this.lexer.state.inRawBlock&&this.rules.other.endPreScriptTag.test(t[0])&&(this.lexer.state.inRawBlock=!1),{type:"html",raw:t[0],inLink:this.lexer.state.inLink,inRawBlock:this.lexer.state.inRawBlock,block:!1,text:t[0]}}link(e){const t=this.rules.inline.link.exec(e);if(t){const e=t[2].trim();if(!this.options.pedantic&&this.rules.other.startAngleBracket.test(e)){if(!this.rules.other.endAngleBracket.test(e))return;const t=F(e.slice(0,-1),"\\");if((e.length-t.length)%2==0)return}else{const e=function(e,t){if(-1===e.indexOf(t[1]))return-1;let n=0;for(let s=0;s<e.length;s++)if("\\"===e[s])s++;else if(e[s]===t[0])n++;else if(e[s]===t[1]&&(n--,n<0))return s;return-1}(t[2],"()");if(e>-1){const n=(0===t[0].indexOf("!")?5:4)+t[1].length+e;t[2]=t[2].substring(0,e),t[0]=t[0].substring(0,n).trim(),t[3]=""}}let n=t[2],s="";if(this.options.pedantic){const e=this.rules.other.pedanticHrefTitle.exec(n);e&&(n=e[1],s=e[3])}else s=t[3]?t[3].slice(1,-1):"";return n=n.trim(),this.rules.other.startAngleBracket.test(n)&&(n=this.options.pedantic&&!this.rules.other.endAngleBracket.test(e)?n.slice(1):n.slice(1,-1)),U(t,{href:n?n.replace(this.rules.inline.anyPunctuation,"$1"):n,title:s?s.replace(this.rules.inline.anyPunctuation,"$1"):s},t[0],this.lexer,this.rules)}}reflink(e,t){let n;if((n=this.rules.inline.reflink.exec(e))||(n=this.rules.inline.nolink.exec(e))){const e=t[(n[2]||n[1]).replace(this.rules.other.multipleSpaceGlobal," ").toLowerCase()];if(!e){const e=n[0].charAt(0);return{type:"text",raw:e,text:e}}return U(n,e,n[0],this.lexer,this.rules)}}emStrong(e,t,n=""){let s=this.rules.inline.emStrongLDelim.exec(e);if(!s)return;if(s[3]&&n.match(this.rules.other.unicodeAlphaNumeric))return;if(!(s[1]||s[2]||"")||!n||this.rules.inline.punctuation.exec(n)){const n=[...s[0]].length-1;let r,i,l=n,o=0;const a="*"===s[0][0]?this.rules.inline.emStrongRDelimAst:this.rules.inline.emStrongRDelimUnd;for(a.lastIndex=0,t=t.slice(-1*e.length+n);null!=(s=a.exec(t));){if(r=s[1]||s[2]||s[3]||s[4]||s[5]||s[6],!r)continue;if(i=[...r].length,s[3]||s[4]){l+=i;continue}if((s[5]||s[6])&&n%3&&!((n+i)%3)){o+=i;continue}if(l-=i,l>0)continue;i=Math.min(i,i+l+o);const t=[...s[0]][0].length,a=e.slice(0,n+s.index+t+i);if(Math.min(n,i)%2){const e=a.slice(1,-1);return{type:"em",raw:a,text:e,tokens:this.lexer.inlineTokens(e)}}const c=a.slice(2,-2);return{type:"strong",raw:a,text:c,tokens:this.lexer.inlineTokens(c)}}}}codespan(e){const t=this.rules.inline.code.exec(e);if(t){let e=t[2].replace(this.rules.other.newLineCharGlobal," ");const n=this.rules.other.nonSpaceChar.test(e),s=this.rules.other.startingSpaceChar.test(e)&&this.rules.other.endingSpaceChar.test(e);return n&&s&&(e=e.substring(1,e.length-1)),{type:"codespan",raw:t[0],text:e}}}br(e){const t=this.rules.inline.br.exec(e);if(t)return{type:"br",raw:t[0]}}del(e){const t=this.rules.inline.del.exec(e);if(t)return{type:"del",raw:t[0],text:t[2],tokens:this.lexer.inlineTokens(t[2])}}autolink(e){const t=this.rules.inline.autolink.exec(e);if(t){let e,n;return"@"===t[2]?(e=t[1],n="mailto:"+e):(e=t[1],n=e),{type:"link",raw:t[0],text:e,href:n,tokens:[{type:"text",raw:e,text:e}]}}}url(e){let t;if(t=this.rules.inline.url.exec(e)){let e,n;if("@"===t[2])e=t[0],n="mailto:"+e;else{let s;do{s=t[0],t[0]=this.rules.inline._backpedal.exec(t[0])?.[0]??""}while(s!==t[0]);e=t[0],n="www."===t[1]?"http://"+t[0]:t[0]}return{type:"link",raw:t[0],text:e,href:n,tokens:[{type:"text",raw:e,text:e}]}}}inlineText(e){const t=this.rules.inline.text.exec(e);if(t){const e=this.lexer.state.inRawBlock;return{type:"text",raw:t[0],text:t[0],escaped:e}}}}class K{tokens;options;state;tokenizer;inlineQueue;constructor(t){this.tokens=[],this.tokens.links=Object.create(null),this.options=t||e.defaults,this.options.tokenizer=this.options.tokenizer||new J,this.tokenizer=this.options.tokenizer,this.tokenizer.options=this.options,this.tokenizer.lexer=this,this.inlineQueue=[],this.state={inLink:!1,inRawBlock:!1,top:!0};const n={other:i,block:M.normal,inline:O.normal};this.options.pedantic?(n.block=M.pedantic,n.inline=O.pedantic):this.options.gfm&&(n.block=M.gfm,this.options.breaks?n.inline=O.breaks:n.inline=O.gfm),this.tokenizer.rules=n}static get rules(){return{block:M,inline:O}}static lex(e,t){return new K(t).lex(e)}static lexInline(e,t){return new K(t).inlineTokens(e)}lex(e){e=e.replace(i.carriageReturn,"\n"),this.blockTokens(e,this.tokens);for(let e=0;e<this.inlineQueue.length;e++){const t=this.inlineQueue[e];this.inlineTokens(t.src,t.tokens)}return this.inlineQueue=[],this.tokens}blockTokens(e,t=[],n=!1){let s,r,l;for(this.options.pedantic&&(e=e.replace(i.tabCharGlobal,"    ").replace(i.spaceLine,""));e;)if(!(this.options.extensions&&this.options.extensions.block&&this.options.extensions.block.some((n=>!!(s=n.call({lexer:this},e,t))&&(e=e.substring(s.raw.length),t.push(s),!0)))))if(s=this.tokenizer.space(e))e=e.substring(s.raw.length),1===s.raw.length&&t.length>0?t[t.length-1].raw+="\n":t.push(s);else if(s=this.tokenizer.code(e))e=e.substring(s.raw.length),r=t[t.length-1],!r||"paragraph"!==r.type&&"text"!==r.type?t.push(s):(r.raw+="\n"+s.raw,r.text+="\n"+s.text,this.inlineQueue[this.inlineQueue.length-1].src=r.text);else if(s=this.tokenizer.fences(e))e=e.substring(s.raw.length),t.push(s);else if(s=this.tokenizer.heading(e))e=e.substring(s.raw.length),t.push(s);else if(s=this.tokenizer.hr(e))e=e.substring(s.raw.length),t.push(s);else if(s=this.tokenizer.blockquote(e))e=e.substring(s.raw.length),t.push(s);else if(s=this.tokenizer.list(e))e=e.substring(s.raw.length),t.push(s);else if(s=this.tokenizer.html(e))e=e.substring(s.raw.length),t.push(s);else if(s=this.tokenizer.def(e))e=e.substring(s.raw.length),r=t[t.length-1],!r||"paragraph"!==r.type&&"text"!==r.type?this.tokens.links[s.tag]||(this.tokens.links[s.tag]={href:s.href,title:s.title}):(r.raw+="\n"+s.raw,r.text+="\n"+s.raw,this.inlineQueue[this.inlineQueue.length-1].src=r.text);else if(s=this.tokenizer.table(e))e=e.substring(s.raw.length),t.push(s);else if(s=this.tokenizer.lheading(e))e=e.substring(s.raw.length),t.push(s);else{if(l=e,this.options.extensions&&this.options.extensions.startBlock){let t=1/0;const n=e.slice(1);let s;this.options.extensions.startBlock.forEach((e=>{s=e.call({lexer:this},n),"number"==typeof s&&s>=0&&(t=Math.min(t,s))})),t<1/0&&t>=0&&(l=e.substring(0,t+1))}if(this.state.top&&(s=this.tokenizer.paragraph(l)))r=t[t.length-1],n&&"paragraph"===r?.type?(r.raw+="\n"+s.raw,r.text+="\n"+s.text,this.inlineQueue.pop(),this.inlineQueue[this.inlineQueue.length-1].src=r.text):t.push(s),n=l.length!==e.length,e=e.substring(s.raw.length);else if(s=this.tokenizer.text(e))e=e.substring(s.raw.length),r=t[t.length-1],r&&"text"===r.type?(r.raw+="\n"+s.raw,r.text+="\n"+s.text,this.inlineQueue.pop(),this.inlineQueue[this.inlineQueue.length-1].src=r.text):t.push(s);else if(e){const t="Infinite loop on byte: "+e.charCodeAt(0);if(this.options.silent){console.error(t);break}throw new Error(t)}}return this.state.top=!0,t}inline(e,t=[]){return this.inlineQueue.push({src:e,tokens:t}),t}inlineTokens(e,t=[]){let n,s,r,i,l,o,a=e;if(this.tokens.links){const e=Object.keys(this.tokens.links);if(e.length>0)for(;null!=(i=this.tokenizer.rules.inline.reflinkSearch.exec(a));)e.includes(i[0].slice(i[0].lastIndexOf("[")+1,-1))&&(a=a.slice(0,i.index)+"["+"a".repeat(i[0].length-2)+"]"+a.slice(this.tokenizer.rules.inline.reflinkSearch.lastIndex))}for(;null!=(i=this.tokenizer.rules.inline.blockSkip.exec(a));)a=a.slice(0,i.index)+"["+"a".repeat(i[0].length-2)+"]"+a.slice(this.tokenizer.rules.inline.blockSkip.lastIndex);for(;null!=(i=this.tokenizer.rules.inline.anyPunctuation.exec(a));)a=a.slice(0,i.index)+"++"+a.slice(this.tokenizer.rules.inline.anyPunctuation.lastIndex);for(;e;)if(l||(o=""),l=!1,!(this.options.extensions&&this.options.extensions.inline&&this.options.extensions.inline.some((s=>!!(n=s.call({lexer:this},e,t))&&(e=e.substring(n.raw.length),t.push(n),!0)))))if(n=this.tokenizer.escape(e))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.tag(e))e=e.substring(n.raw.length),s=t[t.length-1],t.push(n);else if(n=this.tokenizer.link(e))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.reflink(e,this.tokens.links))e=e.substring(n.raw.length),s=t[t.length-1],s&&"text"===n.type&&"text"===s.type?(s.raw+=n.raw,s.text+=n.text):t.push(n);else if(n=this.tokenizer.emStrong(e,a,o))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.codespan(e))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.br(e))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.del(e))e=e.substring(n.raw.length),t.push(n);else if(n=this.tokenizer.autolink(e))e=e.substring(n.raw.length),t.push(n);else if(this.state.inLink||!(n=this.tokenizer.url(e))){if(r=e,this.options.extensions&&this.options.extensions.startInline){let t=1/0;const n=e.slice(1);let s;this.options.extensions.startInline.forEach((e=>{s=e.call({lexer:this},n),"number"==typeof s&&s>=0&&(t=Math.min(t,s))})),t<1/0&&t>=0&&(r=e.substring(0,t+1))}if(n=this.tokenizer.inlineText(r))e=e.substring(n.raw.length),"_"!==n.raw.slice(-1)&&(o=n.raw.slice(-1)),l=!0,s=t[t.length-1],s&&"text"===s.type?(s.raw+=n.raw,s.text+=n.text):t.push(n);else if(e){const t="Infinite loop on byte: "+e.charCodeAt(0);if(this.options.silent){console.error(t);break}throw new Error(t)}}else e=e.substring(n.raw.length),t.push(n);return t}}class V{options;parser;constructor(t){this.options=t||e.defaults}space(e){return""}code({text:e,lang:t,escaped:n}){const s=(t||"").match(i.notSpaceStart)?.[0],r=e.replace(i.endingNewline,"")+"\n";return s?'<pre><code class="language-'+G(s)+'">'+(n?r:G(r,!0))+"</code></pre>\n":"<pre><code>"+(n?r:G(r,!0))+"</code></pre>\n"}blockquote({tokens:e}){return`<blockquote>\n${this.parser.parse(e)}</blockquote>\n`}html({text:e}){return e}heading({tokens:e,depth:t}){return`<h${t}>${this.parser.parseInline(e)}</h${t}>\n`}hr(e){return"<hr>\n"}list(e){const t=e.ordered,n=e.start;let s="";for(let t=0;t<e.items.length;t++){const n=e.items[t];s+=this.listitem(n)}const r=t?"ol":"ul";return"<"+r+(t&&1!==n?' start="'+n+'"':"")+">\n"+s+"</"+r+">\n"}listitem(e){let t="";if(e.task){const n=this.checkbox({checked:!!e.checked});e.loose?e.tokens.length>0&&"paragraph"===e.tokens[0].type?(e.tokens[0].text=n+" "+e.tokens[0].text,e.tokens[0].tokens&&e.tokens[0].tokens.length>0&&"text"===e.tokens[0].tokens[0].type&&(e.tokens[0].tokens[0].text=n+" "+G(e.tokens[0].tokens[0].text),e.tokens[0].tokens[0].escaped=!0)):e.tokens.unshift({type:"text",raw:n+" ",text:n+" ",escaped:!0}):t+=n+" "}return t+=this.parser.parse(e.tokens,!!e.loose),`<li>${t}</li>\n`}checkbox({checked:e}){return"<input "+(e?'checked="" ':"")+'disabled="" type="checkbox">'}paragraph({tokens:e}){return`<p>${this.parser.parseInline(e)}</p>\n`}table(e){let t="",n="";for(let t=0;t<e.header.length;t++)n+=this.tablecell(e.header[t]);t+=this.tablerow({text:n});let s="";for(let t=0;t<e.rows.length;t++){const r=e.rows[t];n="";for(let e=0;e<r.length;e++)n+=this.tablecell(r[e]);s+=this.tablerow({text:n})}return s&&(s=`<tbody>${s}</tbody>`),"<table>\n<thead>\n"+t+"</thead>\n"+s+"</table>\n"}tablerow({text:e}){return`<tr>\n${e}</tr>\n`}tablecell(e){const t=this.parser.parseInline(e.tokens),n=e.header?"th":"td";return(e.align?`<${n} align="${e.align}">`:`<${n}>`)+t+`</${n}>\n`}strong({tokens:e}){return`<strong>${this.parser.parseInline(e)}</strong>`}em({tokens:e}){return`<em>${this.parser.parseInline(e)}</em>`}codespan({text:e}){return`<code>${G(e,!0)}</code>`}br(e){return"<br>"}del({tokens:e}){return`<del>${this.parser.parseInline(e)}</del>`}link({href:e,title:t,tokens:n}){const s=this.parser.parseInline(n),r=H(e);if(null===r)return s;let i='<a href="'+(e=r)+'"';return t&&(i+=' title="'+G(t)+'"'),i+=">"+s+"</a>",i}image({href:e,title:t,text:n}){const s=H(e);if(null===s)return G(n);let r=`<img src="${e=s}" alt="${n}"`;return t&&(r+=` title="${G(t)}"`),r+=">",r}text(e){return"tokens"in e&&e.tokens?this.parser.parseInline(e.tokens):"escaped"in e&&e.escaped?e.text:G(e.text)}}class W{strong({text:e}){return e}em({text:e}){return e}codespan({text:e}){return e}del({text:e}){return e}html({text:e}){return e}text({text:e}){return e}link({text:e}){return""+e}image({text:e}){return""+e}br(){return""}}class Y{options;renderer;textRenderer;constructor(t){this.options=t||e.defaults,this.options.renderer=this.options.renderer||new V,this.renderer=this.options.renderer,this.renderer.options=this.options,this.renderer.parser=this,this.textRenderer=new W}static parse(e,t){return new Y(t).parse(e)}static parseInline(e,t){return new Y(t).parseInline(e)}parse(e,t=!0){let n="";for(let s=0;s<e.length;s++){const r=e[s];if(this.options.extensions&&this.options.extensions.renderers&&this.options.extensions.renderers[r.type]){const e=r,t=this.options.extensions.renderers[e.type].call({parser:this},e);if(!1!==t||!["space","hr","heading","code","table","blockquote","list","html","paragraph","text"].includes(e.type)){n+=t||"";continue}}const i=r;switch(i.type){case"space":n+=this.renderer.space(i);continue;case"hr":n+=this.renderer.hr(i);continue;case"heading":n+=this.renderer.heading(i);continue;case"code":n+=this.renderer.code(i);continue;case"table":n+=this.renderer.table(i);continue;case"blockquote":n+=this.renderer.blockquote(i);continue;case"list":n+=this.renderer.list(i);continue;case"html":n+=this.renderer.html(i);continue;case"paragraph":n+=this.renderer.paragraph(i);continue;case"text":{let r=i,l=this.renderer.text(r);for(;s+1<e.length&&"text"===e[s+1].type;)r=e[++s],l+="\n"+this.renderer.text(r);n+=t?this.renderer.paragraph({type:"paragraph",raw:l,text:l,tokens:[{type:"text",raw:l,text:l,escaped:!0}]}):l;continue}default:{const e='Token with "'+i.type+'" type was not found.';if(this.options.silent)return console.error(e),"";throw new Error(e)}}}return n}parseInline(e,t){t=t||this.renderer;let n="";for(let s=0;s<e.length;s++){const r=e[s];if(this.options.extensions&&this.options.extensions.renderers&&this.options.extensions.renderers[r.type]){const e=this.options.extensions.renderers[r.type].call({parser:this},r);if(!1!==e||!["escape","html","link","image","strong","em","codespan","br","del","text"].includes(r.type)){n+=e||"";continue}}const i=r;switch(i.type){case"escape":case"text":n+=t.text(i);break;case"html":n+=t.html(i);break;case"link":n+=t.link(i);break;case"image":n+=t.image(i);break;case"strong":n+=t.strong(i);break;case"em":n+=t.em(i);break;case"codespan":n+=t.codespan(i);break;case"br":n+=t.br(i);break;case"del":n+=t.del(i);break;default:{const e='Token with "'+i.type+'" type was not found.';if(this.options.silent)return console.error(e),"";throw new Error(e)}}}return n}}class ee{options;block;constructor(t){this.options=t||e.defaults}static passThroughHooks=new Set(["preprocess","postprocess","processAllTokens"]);preprocess(e){return e}postprocess(e){return e}processAllTokens(e){return e}provideLexer(){return this.block?K.lex:K.lexInline}provideParser(){return this.block?Y.parse:Y.parseInline}}class te{defaults={async:!1,breaks:!1,extensions:null,gfm:!0,hooks:null,pedantic:!1,renderer:null,silent:!1,tokenizer:null,walkTokens:null};options=this.setOptions;parse=this.parseMarkdown(!0);parseInline=this.parseMarkdown(!1);Parser=Y;Renderer=V;TextRenderer=W;Lexer=K;Tokenizer=J;Hooks=ee;constructor(...e){this.use(...e)}walkTokens(e,t){let n=[];for(const s of e)switch(n=n.concat(t.call(this,s)),s.type){case"table":{const e=s;for(const s of e.header)n=n.concat(this.walkTokens(s.tokens,t));for(const s of e.rows)for(const e of s)n=n.concat(this.walkTokens(e.tokens,t));break}case"list":{const e=s;n=n.concat(this.walkTokens(e.items,t));break}default:{const e=s;this.defaults.extensions?.childTokens?.[e.type]?this.defaults.extensions.childTokens[e.type].forEach((s=>{const r=e[s].flat(1/0);n=n.concat(this.walkTokens(r,t))})):e.tokens&&(n=n.concat(this.walkTokens(e.tokens,t)))}}return n}use(...e){const t=this.defaults.extensions||{renderers:{},childTokens:{}};return e.forEach((e=>{const n={...e};if(n.async=this.defaults.async||n.async||!1,e.extensions&&(e.extensions.forEach((e=>{if(!e.name)throw new Error("extension name required");if("renderer"in e){const n=t.renderers[e.name];t.renderers[e.name]=n?function(...t){let s=e.renderer.apply(this,t);return!1===s&&(s=n.apply(this,t)),s}:e.renderer}if("tokenizer"in e){if(!e.level||"block"!==e.level&&"inline"!==e.level)throw new Error("extension level must be 'block' or 'inline'");const n=t[e.level];n?n.unshift(e.tokenizer):t[e.level]=[e.tokenizer],e.start&&("block"===e.level?t.startBlock?t.startBlock.push(e.start):t.startBlock=[e.start]:"inline"===e.level&&(t.startInline?t.startInline.push(e.start):t.startInline=[e.start]))}"childTokens"in e&&e.childTokens&&(t.childTokens[e.name]=e.childTokens)})),n.extensions=t),e.renderer){const t=this.defaults.renderer||new V(this.defaults);for(const n in e.renderer){if(!(n in t))throw new Error(`renderer '${n}' does not exist`);if(["options","parser"].includes(n))continue;const s=n,r=e.renderer[s],i=t[s];t[s]=(...e)=>{let n=r.apply(t,e);return!1===n&&(n=i.apply(t,e)),n||""}}n.renderer=t}if(e.tokenizer){const t=this.defaults.tokenizer||new J(this.defaults);for(const n in e.tokenizer){if(!(n in t))throw new Error(`tokenizer '${n}' does not exist`);if(["options","rules","lexer"].includes(n))continue;const s=n,r=e.tokenizer[s],i=t[s];t[s]=(...e)=>{let n=r.apply(t,e);return!1===n&&(n=i.apply(t,e)),n}}n.tokenizer=t}if(e.hooks){const t=this.defaults.hooks||new ee;for(const n in e.hooks){if(!(n in t))throw new Error(`hook '${n}' does not exist`);if(["options","block"].includes(n))continue;const s=n,r=e.hooks[s],i=t[s];ee.passThroughHooks.has(n)?t[s]=e=>{if(this.defaults.async)return Promise.resolve(r.call(t,e)).then((e=>i.call(t,e)));const n=r.call(t,e);return i.call(t,n)}:t[s]=(...e)=>{let n=r.apply(t,e);return!1===n&&(n=i.apply(t,e)),n}}n.hooks=t}if(e.walkTokens){const t=this.defaults.walkTokens,s=e.walkTokens;n.walkTokens=function(e){let n=[];return n.push(s.call(this,e)),t&&(n=n.concat(t.call(this,e))),n}}this.defaults={...this.defaults,...n}})),this}setOptions(e){return this.defaults={...this.defaults,...e},this}lexer(e,t){return K.lex(e,t??this.defaults)}parser(e,t){return Y.parse(e,t??this.defaults)}parseMarkdown(e){return(t,n)=>{const s={...n},r={...this.defaults,...s},i=this.onError(!!r.silent,!!r.async);if(!0===this.defaults.async&&!1===s.async)return i(new Error("marked(): The async option was set to true by an extension. Remove async: false from the parse options object to return a Promise."));if(null==t)return i(new Error("marked(): input parameter is undefined or null"));if("string"!=typeof t)return i(new Error("marked(): input parameter is of type "+Object.prototype.toString.call(t)+", string expected"));r.hooks&&(r.hooks.options=r,r.hooks.block=e);const l=r.hooks?r.hooks.provideLexer():e?K.lex:K.lexInline,o=r.hooks?r.hooks.provideParser():e?Y.parse:Y.parseInline;if(r.async)return Promise.resolve(r.hooks?r.hooks.preprocess(t):t).then((e=>l(e,r))).then((e=>r.hooks?r.hooks.processAllTokens(e):e)).then((e=>r.walkTokens?Promise.all(this.walkTokens(e,r.walkTokens)).then((()=>e)):e)).then((e=>o(e,r))).then((e=>r.hooks?r.hooks.postprocess(e):e)).catch(i);try{r.hooks&&(t=r.hooks.preprocess(t));let e=l(t,r);r.hooks&&(e=r.hooks.processAllTokens(e)),r.walkTokens&&this.walkTokens(e,r.walkTokens);let n=o(e,r);return r.hooks&&(n=r.hooks.postprocess(n)),n}catch(e){return i(e)}}}onError(e,t){return n=>{if(n.message+="\nPlease report this to https://github.com/markedjs/marked.",e){const e="<p>An error occurred:</p><pre>"+G(n.message+"",!0)+"</pre>";return t?Promise.resolve(e):e}if(t)return Promise.reject(n);throw n}}}const ne=new te;function se(e,t){return ne.parse(e,t)}se.options=se.setOptions=function(e){return ne.setOptions(e),se.defaults=ne.defaults,n(se.defaults),se},se.getDefaults=t,se.defaults=e.defaults,se.use=function(...e){return ne.use(...e),se.defaults=ne.defaults,n(se.defaults),se},se.walkTokens=function(e,t){return ne.walkTokens(e,t)},se.parseInline=ne.parseInline,se.Parser=Y,se.parser=Y.parse,se.Renderer=V,se.TextRenderer=W,se.Lexer=K,se.lexer=K.lex,se.Tokenizer=J,se.Hooks=ee,se.parse=se;const re=se.options,ie=se.setOptions,le=se.use,oe=se.walkTokens,ae=se.parseInline,ce=se,he=Y.parse,pe=K.lex;e.Hooks=ee,e.Lexer=K,e.Marked=te,e.Parser=Y,e.Renderer=V,e.TextRenderer=W,e.Tokenizer=J,e.getDefaults=t,e.lexer=pe,e.marked=se,e.options=re,e.parse=ce,e.parseInline=ae,e.parser=he,e.setOptions=ie,e.use=le,e.walkTokens=oe}));
</script>
<script>/* DOMPurify 3.2.4 (inline) */
/*! @license DOMPurify 3.2.4 | (c) Cure53 and other contributors | Released under the Apache license 2.0 and Mozilla Public License 2.0 | github.com/cure53/DOMPurify/blob/3.2.4/LICENSE */
!function(e,t){"object"==typeof exports&&"undefined"!=typeof module?module.exports=t():"function"==typeof define&&define.amd?define(t):(e="undefined"!=typeof globalThis?globalThis:e||self).DOMPurify=t()}(this,(function(){"use strict";const{entries:e,setPrototypeOf:t,isFrozen:n,getPrototypeOf:o,getOwnPropertyDescriptor:r}=Object;let{freeze:i,seal:a,create:l}=Object,{apply:c,construct:s}="undefined"!=typeof Reflect&&Reflect;i||(i=function(e){return e}),a||(a=function(e){return e}),c||(c=function(e,t,n){return e.apply(t,n)}),s||(s=function(e,t){return new e(...t)});const u=R(Array.prototype.forEach),m=R(Array.prototype.lastIndexOf),p=R(Array.prototype.pop),f=R(Array.prototype.push),d=R(Array.prototype.splice),h=R(String.prototype.toLowerCase),g=R(String.prototype.toString),T=R(String.prototype.match),y=R(String.prototype.replace),E=R(String.prototype.indexOf),A=R(String.prototype.trim),_=R(Object.prototype.hasOwnProperty),S=R(RegExp.prototype.test),b=(N=TypeError,function(){for(var e=arguments.length,t=new Array(e),n=0;n<e;n++)t[n]=arguments[n];return s(N,t)});var N;function R(e){return function(t){for(var n=arguments.length,o=new Array(n>1?n-1:0),r=1;r<n;r++)o[r-1]=arguments[r];return c(e,t,o)}}function w(e,o){let r=arguments.length>2&&void 0!==arguments[2]?arguments[2]:h;t&&t(e,null);let i=o.length;for(;i--;){let t=o[i];if("string"==typeof t){const e=r(t);e!==t&&(n(o)||(o[i]=e),t=e)}e[t]=!0}return e}function O(e){for(let t=0;t<e.length;t++){_(e,t)||(e[t]=null)}return e}function D(t){const n=l(null);for(const[o,r]of e(t)){_(t,o)&&(Array.isArray(r)?n[o]=O(r):r&&"object"==typeof r&&r.constructor===Object?n[o]=D(r):n[o]=r)}return n}function v(e,t){for(;null!==e;){const n=r(e,t);if(n){if(n.get)return R(n.get);if("function"==typeof n.value)return R(n.value)}e=o(e)}return function(){return null}}const L=i(["a","abbr","acronym","address","area","article","aside","audio","b","bdi","bdo","big","blink","blockquote","body","br","button","canvas","caption","center","cite","code","col","colgroup","content","data","datalist","dd","decorator","del","details","dfn","dialog","dir","div","dl","dt","element","em","fieldset","figcaption","figure","font","footer","form","h1","h2","h3","h4","h5","h6","head","header","hgroup","hr","html","i","img","input","ins","kbd","label","legend","li","main","map","mark","marquee","menu","menuitem","meter","nav","nobr","ol","optgroup","option","output","p","picture","pre","progress","q","rp","rt","ruby","s","samp","section","select","shadow","small","source","spacer","span","strike","strong","style","sub","summary","sup","table","tbody","td","template","textarea","tfoot","th","thead","time","tr","track","tt","u","ul","var","video","wbr"]),C=i(["svg","a","altglyph","altglyphdef","altglyphitem","animatecolor","animatemotion","animatetransform","circle","clippath","defs","desc","ellipse","filter","font","g","glyph","glyphref","hkern","image","line","lineargradient","marker","mask","metadata","mpath","path","pattern","polygon","polyline","radialgradient","rect","stop","style","switch","symbol","text","textpath","title","tref","tspan","view","vkern"]),x=i(["feBlend","feColorMatrix","feComponentTransfer","feComposite","feConvolveMatrix","feDiffuseLighting","feDisplacementMap","feDistantLight","feDropShadow","feFlood","feFuncA","feFuncB","feFuncG","feFuncR","feGaussianBlur","feImage","feMerge","feMergeNode","feMorphology","feOffset","fePointLight","feSpecularLighting","feSpotLight","feTile","feTurbulence"]),M=i(["animate","color-profile","cursor","discard","font-face","font-face-format","font-face-name","font-face-src","font-face-uri","foreignobject","hatch","hatchpath","mesh","meshgradient","meshpatch","meshrow","missing-glyph","script","set","solidcolor","unknown","use"]),k=i(["math","menclose","merror","mfenced","mfrac","mglyph","mi","mlabeledtr","mmultiscripts","mn","mo","mover","mpadded","mphantom","mroot","mrow","ms","mspace","msqrt","mstyle","msub","msup","msubsup","mtable","mtd","mtext","mtr","munder","munderover","mprescripts"]),I=i(["maction","maligngroup","malignmark","mlongdiv","mscarries","mscarry","msgroup","mstack","msline","msrow","semantics","annotation","annotation-xml","mprescripts","none"]),U=i(["#text"]),z=i(["accept","action","align","alt","autocapitalize","autocomplete","autopictureinpicture","autoplay","background","bgcolor","border","capture","cellpadding","cellspacing","checked","cite","class","clear","color","cols","colspan","controls","controlslist","coords","crossorigin","datetime","decoding","default","dir","disabled","disablepictureinpicture","disableremoteplayback","download","draggable","enctype","enterkeyhint","face","for","headers","height","hidden","high","href","hreflang","id","inputmode","integrity","ismap","kind","label","lang","list","loading","loop","low","max","maxlength","media","method","min","minlength","multiple","muted","name","nonce","noshade","novalidate","nowrap","open","optimum","pattern","placeholder","playsinline","popover","popovertarget","popovertargetaction","poster","preload","pubdate","radiogroup","readonly","rel","required","rev","reversed","role","rows","rowspan","spellcheck","scope","selected","shape","size","sizes","span","srclang","start","src","srcset","step","style","summary","tabindex","title","translate","type","usemap","valign","value","width","wrap","xmlns","slot"]),P=i(["accent-height","accumulate","additive","alignment-baseline","amplitude","ascent","attributename","attributetype","azimuth","basefrequency","baseline-shift","begin","bias","by","class","clip","clippathunits","clip-path","clip-rule","color","color-interpolation","color-interpolation-filters","color-profile","color-rendering","cx","cy","d","dx","dy","diffuseconstant","direction","display","divisor","dur","edgemode","elevation","end","exponent","fill","fill-opacity","fill-rule","filter","filterunits","flood-color","flood-opacity","font-family","font-size","font-size-adjust","font-stretch","font-style","font-variant","font-weight","fx","fy","g1","g2","glyph-name","glyphref","gradientunits","gradienttransform","height","href","id","image-rendering","in","in2","intercept","k","k1","k2","k3","k4","kerning","keypoints","keysplines","keytimes","lang","lengthadjust","letter-spacing","kernelmatrix","kernelunitlength","lighting-color","local","marker-end","marker-mid","marker-start","markerheight","markerunits","markerwidth","maskcontentunits","maskunits","max","mask","media","method","mode","min","name","numoctaves","offset","operator","opacity","order","orient","orientation","origin","overflow","paint-order","path","pathlength","patterncontentunits","patterntransform","patternunits","points","preservealpha","preserveaspectratio","primitiveunits","r","rx","ry","radius","refx","refy","repeatcount","repeatdur","restart","result","rotate","scale","seed","shape-rendering","slope","specularconstant","specularexponent","spreadmethod","startoffset","stddeviation","stitchtiles","stop-color","stop-opacity","stroke-dasharray","stroke-dashoffset","stroke-linecap","stroke-linejoin","stroke-miterlimit","stroke-opacity","stroke","stroke-width","style","surfacescale","systemlanguage","tabindex","tablevalues","targetx","targety","transform","transform-origin","text-anchor","text-decoration","text-rendering","textlength","type","u1","u2","unicode","values","viewbox","visibility","version","vert-adv-y","vert-origin-x","vert-origin-y","width","word-spacing","wrap","writing-mode","xchannelselector","ychannelselector","x","x1","x2","xmlns","y","y1","y2","z","zoomandpan"]),H=i(["accent","accentunder","align","bevelled","close","columnsalign","columnlines","columnspan","denomalign","depth","dir","display","displaystyle","encoding","fence","frame","height","href","id","largeop","length","linethickness","lspace","lquote","mathbackground","mathcolor","mathsize","mathvariant","maxsize","minsize","movablelimits","notation","numalign","open","rowalign","rowlines","rowspacing","rowspan","rspace","rquote","scriptlevel","scriptminsize","scriptsizemultiplier","selection","separator","separators","stretchy","subscriptshift","supscriptshift","symmetric","voffset","width","xmlns"]),F=i(["xlink:href","xml:id","xlink:title","xml:space","xmlns:xlink"]),B=a(/\{\{[\w\W]*|[\w\W]*\}\}/gm),W=a(/<%[\w\W]*|[\w\W]*%>/gm),G=a(/\$\{[\w\W]*/gm),Y=a(/^data-[\-\w.\u00B7-\uFFFF]+$/),j=a(/^aria-[\-\w]+$/),X=a(/^(?:(?:(?:f|ht)tps?|mailto|tel|callto|sms|cid|xmpp):|[^a-z]|[a-z+.\-]+(?:[^a-z+.\-:]|$))/i),q=a(/^(?:\w+script|data):/i),$=a(/[\u0000-\u0020\u00A0\u1680\u180E\u2000-\u2029\u205F\u3000]/g),K=a(/^html$/i),V=a(/^[a-z][.\w]*(-[.\w]+)+$/i);var Z=Object.freeze({__proto__:null,ARIA_ATTR:j,ATTR_WHITESPACE:$,CUSTOM_ELEMENT:V,DATA_ATTR:Y,DOCTYPE_NAME:K,ERB_EXPR:W,IS_ALLOWED_URI:X,IS_SCRIPT_OR_DATA:q,MUSTACHE_EXPR:B,TMPLIT_EXPR:G});const J=1,Q=3,ee=7,te=8,ne=9,oe=function(){return"undefined"==typeof window?null:window};var re=function t(){let n=arguments.length>0&&void 0!==arguments[0]?arguments[0]:oe();const o=e=>t(e);if(o.version="3.2.4",o.removed=[],!n||!n.document||n.document.nodeType!==ne||!n.Element)return o.isSupported=!1,o;let{document:r}=n;const a=r,c=a.currentScript,{DocumentFragment:s,HTMLTemplateElement:N,Node:R,Element:O,NodeFilter:B,NamedNodeMap:W=n.NamedNodeMap||n.MozNamedAttrMap,HTMLFormElement:G,DOMParser:Y,trustedTypes:j}=n,q=O.prototype,$=v(q,"cloneNode"),V=v(q,"remove"),re=v(q,"nextSibling"),ie=v(q,"childNodes"),ae=v(q,"parentNode");if("function"==typeof N){const e=r.createElement("template");e.content&&e.content.ownerDocument&&(r=e.content.ownerDocument)}let le,ce="";const{implementation:se,createNodeIterator:ue,createDocumentFragment:me,getElementsByTagName:pe}=r,{importNode:fe}=a;let de={afterSanitizeAttributes:[],afterSanitizeElements:[],afterSanitizeShadowDOM:[],beforeSanitizeAttributes:[],beforeSanitizeElements:[],beforeSanitizeShadowDOM:[],uponSanitizeAttribute:[],uponSanitizeElement:[],uponSanitizeShadowNode:[]};o.isSupported="function"==typeof e&&"function"==typeof ae&&se&&void 0!==se.createHTMLDocument;const{MUSTACHE_EXPR:he,ERB_EXPR:ge,TMPLIT_EXPR:Te,DATA_ATTR:ye,ARIA_ATTR:Ee,IS_SCRIPT_OR_DATA:Ae,ATTR_WHITESPACE:_e,CUSTOM_ELEMENT:Se}=Z;let{IS_ALLOWED_URI:be}=Z,Ne=null;const Re=w({},[...L,...C,...x,...k,...U]);let we=null;const Oe=w({},[...z,...P,...H,...F]);let De=Object.seal(l(null,{tagNameCheck:{writable:!0,configurable:!1,enumerable:!0,value:null},attributeNameCheck:{writable:!0,configurable:!1,enumerable:!0,value:null},allowCustomizedBuiltInElements:{writable:!0,configurable:!1,enumerable:!0,value:!1}})),ve=null,Le=null,Ce=!0,xe=!0,Me=!1,ke=!0,Ie=!1,Ue=!0,ze=!1,Pe=!1,He=!1,Fe=!1,Be=!1,We=!1,Ge=!0,Ye=!1,je=!0,Xe=!1,qe={},$e=null;const Ke=w({},["annotation-xml","audio","colgroup","desc","foreignobject","head","iframe","math","mi","mn","mo","ms","mtext","noembed","noframes","noscript","plaintext","script","style","svg","template","thead","title","video","xmp"]);let Ve=null;const Ze=w({},["audio","video","img","source","image","track"]);let Je=null;const Qe=w({},["alt","class","for","id","label","name","pattern","placeholder","role","summary","title","value","style","xmlns"]),et="http://www.w3.org/1998/Math/MathML",tt="http://www.w3.org/2000/svg",nt="http://www.w3.org/1999/xhtml";let ot=nt,rt=!1,it=null;const at=w({},[et,tt,nt],g);let lt=w({},["mi","mo","mn","ms","mtext"]),ct=w({},["annotation-xml"]);const st=w({},["title","style","font","a","script"]);let ut=null;const mt=["application/xhtml+xml","text/html"];let pt=null,ft=null;const dt=r.createElement("form"),ht=function(e){return e instanceof RegExp||e instanceof Function},gt=function(){let e=arguments.length>0&&void 0!==arguments[0]?arguments[0]:{};if(!ft||ft!==e){if(e&&"object"==typeof e||(e={}),e=D(e),ut=-1===mt.indexOf(e.PARSER_MEDIA_TYPE)?"text/html":e.PARSER_MEDIA_TYPE,pt="application/xhtml+xml"===ut?g:h,Ne=_(e,"ALLOWED_TAGS")?w({},e.ALLOWED_TAGS,pt):Re,we=_(e,"ALLOWED_ATTR")?w({},e.ALLOWED_ATTR,pt):Oe,it=_(e,"ALLOWED_NAMESPACES")?w({},e.ALLOWED_NAMESPACES,g):at,Je=_(e,"ADD_URI_SAFE_ATTR")?w(D(Qe),e.ADD_URI_SAFE_ATTR,pt):Qe,Ve=_(e,"ADD_DATA_URI_TAGS")?w(D(Ze),e.ADD_DATA_URI_TAGS,pt):Ze,$e=_(e,"FORBID_CONTENTS")?w({},e.FORBID_CONTENTS,pt):Ke,ve=_(e,"FORBID_TAGS")?w({},e.FORBID_TAGS,pt):{},Le=_(e,"FORBID_ATTR")?w({},e.FORBID_ATTR,pt):{},qe=!!_(e,"USE_PROFILES")&&e.USE_PROFILES,Ce=!1!==e.ALLOW_ARIA_ATTR,xe=!1!==e.ALLOW_DATA_ATTR,Me=e.ALLOW_UNKNOWN_PROTOCOLS||!1,ke=!1!==e.ALLOW_SELF_CLOSE_IN_ATTR,Ie=e.SAFE_FOR_TEMPLATES||!1,Ue=!1!==e.SAFE_FOR_XML,ze=e.WHOLE_DOCUMENT||!1,Fe=e.RETURN_DOM||!1,Be=e.RETURN_DOM_FRAGMENT||!1,We=e.RETURN_TRUSTED_TYPE||!1,He=e.FORCE_BODY||!1,Ge=!1!==e.SANITIZE_DOM,Ye=e.SANITIZE_NAMED_PROPS||!1,je=!1!==e.KEEP_CONTENT,Xe=e.IN_PLACE||!1,be=e.ALLOWED_URI_REGEXP||X,ot=e.NAMESPACE||nt,lt=e.MATHML_TEXT_INTEGRATION_POINTS||lt,ct=e.HTML_INTEGRATION_POINTS||ct,De=e.CUSTOM_ELEMENT_HANDLING||{},e.CUSTOM_ELEMENT_HANDLING&&ht(e.CUSTOM_ELEMENT_HANDLING.tagNameCheck)&&(De.tagNameCheck=e.CUSTOM_ELEMENT_HANDLING.tagNameCheck),e.CUSTOM_ELEMENT_HANDLING&&ht(e.CUSTOM_ELEMENT_HANDLING.attributeNameCheck)&&(De.attributeNameCheck=e.CUSTOM_ELEMENT_HANDLING.attributeNameCheck),e.CUSTOM_ELEMENT_HANDLING&&"boolean"==typeof e.CUSTOM_ELEMENT_HANDLING.allowCustomizedBuiltInElements&&(De.allowCustomizedBuiltInElements=e.CUSTOM_ELEMENT_HANDLING.allowCustomizedBuiltInElements),Ie&&(xe=!1),Be&&(Fe=!0),qe&&(Ne=w({},U),we=[],!0===qe.html&&(w(Ne,L),w(we,z)),!0===qe.svg&&(w(Ne,C),w(we,P),w(we,F)),!0===qe.svgFilters&&(w(Ne,x),w(we,P),w(we,F)),!0===qe.mathMl&&(w(Ne,k),w(we,H),w(we,F))),e.ADD_TAGS&&(Ne===Re&&(Ne=D(Ne)),w(Ne,e.ADD_TAGS,pt)),e.ADD_ATTR&&(we===Oe&&(we=D(we)),w(we,e.ADD_ATTR,pt)),e.ADD_URI_SAFE_ATTR&&w(Je,e.ADD_URI_SAFE_ATTR,pt),e.FORBID_CONTENTS&&($e===Ke&&($e=D($e)),w($e,e.FORBID_CONTENTS,pt)),je&&(Ne["#text"]=!0),ze&&w(Ne,["html","head","body"]),Ne.table&&(w(Ne,["tbody"]),delete ve.tbody),e.TRUSTED_TYPES_POLICY){if("function"!=typeof e.TRUSTED_TYPES_POLICY.createHTML)throw b('TRUSTED_TYPES_POLICY configuration option must provide a "createHTML" hook.');if("function"!=typeof e.TRUSTED_TYPES_POLICY.createScriptURL)throw b('TRUSTED_TYPES_POLICY configuration option must provide a "createScriptURL" hook.');le=e.TRUSTED_TYPES_POLICY,ce=le.createHTML("")}else void 0===le&&(le=function(e,t){if("object"!=typeof e||"function"!=typeof e.createPolicy)return null;let n=null;const o="data-tt-policy-suffix";t&&t.hasAttribute(o)&&(n=t.getAttribute(o));const r="dompurify"+(n?"#"+n:"");try{return e.createPolicy(r,{createHTML:e=>e,createScriptURL:e=>e})}catch(e){return console.warn("TrustedTypes policy "+r+" could not be created."),null}}(j,c)),null!==le&&"string"==typeof ce&&(ce=le.createHTML(""));i&&i(e),ft=e}},Tt=w({},[...C,...x,...M]),yt=w({},[...k,...I]),Et=function(e){f(o.removed,{element:e});try{ae(e).removeChild(e)}catch(t){V(e)}},At=function(e,t){try{f(o.removed,{attribute:t.getAttributeNode(e),from:t})}catch(e){f(o.removed,{attribute:null,from:t})}if(t.removeAttribute(e),"is"===e)if(Fe||Be)try{Et(t)}catch(e){}else try{t.setAttribute(e,"")}catch(e){}},_t=function(e){let t=null,n=null;if(He)e="<remove></remove>"+e;else{const t=T(e,/^[\r\n\t ]+/);n=t&&t[0]}"application/xhtml+xml"===ut&&ot===nt&&(e='<html xmlns="http://www.w3.org/1999/xhtml"><head></head><body>'+e+"</body></html>");const o=le?le.createHTML(e):e;if(ot===nt)try{t=(new Y).parseFromString(o,ut)}catch(e){}if(!t||!t.documentElement){t=se.createDocument(ot,"template",null);try{t.documentElement.innerHTML=rt?ce:o}catch(e){}}const i=t.body||t.documentElement;return e&&n&&i.insertBefore(r.createTextNode(n),i.childNodes[0]||null),ot===nt?pe.call(t,ze?"html":"body")[0]:ze?t.documentElement:i},St=function(e){return ue.call(e.ownerDocument||e,e,B.SHOW_ELEMENT|B.SHOW_COMMENT|B.SHOW_TEXT|B.SHOW_PROCESSING_INSTRUCTION|B.SHOW_CDATA_SECTION,null)},bt=function(e){return e instanceof G&&("string"!=typeof e.nodeName||"string"!=typeof e.textContent||"function"!=typeof e.removeChild||!(e.attributes instanceof W)||"function"!=typeof e.removeAttribute||"function"!=typeof e.setAttribute||"string"!=typeof e.namespaceURI||"function"!=typeof e.insertBefore||"function"!=typeof e.hasChildNodes)},Nt=function(e){return"function"==typeof R&&e instanceof R};function Rt(e,t,n){u(e,(e=>{e.call(o,t,n,ft)}))}const wt=function(e){let t=null;if(Rt(de.beforeSanitizeElements,e,null),bt(e))return Et(e),!0;const n=pt(e.nodeName);if(Rt(de.uponSanitizeElement,e,{tagName:n,allowedTags:Ne}),e.hasChildNodes()&&!Nt(e.firstElementChild)&&S(/<[/\w]/g,e.innerHTML)&&S(/<[/\w]/g,e.textContent))return Et(e),!0;if(e.nodeType===ee)return Et(e),!0;if(Ue&&e.nodeType===te&&S(/<[/\w]/g,e.data))return Et(e),!0;if(!Ne[n]||ve[n]){if(!ve[n]&&Dt(n)){if(De.tagNameCheck instanceof RegExp&&S(De.tagNameCheck,n))return!1;if(De.tagNameCheck instanceof Function&&De.tagNameCheck(n))return!1}if(je&&!$e[n]){const t=ae(e)||e.parentNode,n=ie(e)||e.childNodes;if(n&&t){for(let o=n.length-1;o>=0;--o){const r=$(n[o],!0);r.__removalCount=(e.__removalCount||0)+1,t.insertBefore(r,re(e))}}}return Et(e),!0}return e instanceof O&&!function(e){let t=ae(e);t&&t.tagName||(t={namespaceURI:ot,tagName:"template"});const n=h(e.tagName),o=h(t.tagName);return!!it[e.namespaceURI]&&(e.namespaceURI===tt?t.namespaceURI===nt?"svg"===n:t.namespaceURI===et?"svg"===n&&("annotation-xml"===o||lt[o]):Boolean(Tt[n]):e.namespaceURI===et?t.namespaceURI===nt?"math"===n:t.namespaceURI===tt?"math"===n&&ct[o]:Boolean(yt[n]):e.namespaceURI===nt?!(t.namespaceURI===tt&&!ct[o])&&!(t.namespaceURI===et&&!lt[o])&&!yt[n]&&(st[n]||!Tt[n]):!("application/xhtml+xml"!==ut||!it[e.namespaceURI]))}(e)?(Et(e),!0):"noscript"!==n&&"noembed"!==n&&"noframes"!==n||!S(/<\/no(script|embed|frames)/i,e.innerHTML)?(Ie&&e.nodeType===Q&&(t=e.textContent,u([he,ge,Te],(e=>{t=y(t,e," ")})),e.textContent!==t&&(f(o.removed,{element:e.cloneNode()}),e.textContent=t)),Rt(de.afterSanitizeElements,e,null),!1):(Et(e),!0)},Ot=function(e,t,n){if(Ge&&("id"===t||"name"===t)&&(n in r||n in dt))return!1;if(xe&&!Le[t]&&S(ye,t));else if(Ce&&S(Ee,t));else if(!we[t]||Le[t]){if(!(Dt(e)&&(De.tagNameCheck instanceof RegExp&&S(De.tagNameCheck,e)||De.tagNameCheck instanceof Function&&De.tagNameCheck(e))&&(De.attributeNameCheck instanceof RegExp&&S(De.attributeNameCheck,t)||De.attributeNameCheck instanceof Function&&De.attributeNameCheck(t))||"is"===t&&De.allowCustomizedBuiltInElements&&(De.tagNameCheck instanceof RegExp&&S(De.tagNameCheck,n)||De.tagNameCheck instanceof Function&&De.tagNameCheck(n))))return!1}else if(Je[t]);else if(S(be,y(n,_e,"")));else if("src"!==t&&"xlink:href"!==t&&"href"!==t||"script"===e||0!==E(n,"data:")||!Ve[e]){if(Me&&!S(Ae,y(n,_e,"")));else if(n)return!1}else;return!0},Dt=function(e){return"annotation-xml"!==e&&T(e,Se)},vt=function(e){Rt(de.beforeSanitizeAttributes,e,null);const{attributes:t}=e;if(!t||bt(e))return;const n={attrName:"",attrValue:"",keepAttr:!0,allowedAttributes:we,forceKeepAttr:void 0};let r=t.length;for(;r--;){const i=t[r],{name:a,namespaceURI:l,value:c}=i,s=pt(a);let m="value"===a?c:A(c);if(n.attrName=s,n.attrValue=m,n.keepAttr=!0,n.forceKeepAttr=void 0,Rt(de.uponSanitizeAttribute,e,n),m=n.attrValue,!Ye||"id"!==s&&"name"!==s||(At(a,e),m="user-content-"+m),Ue&&S(/((--!?|])>)|<\/(style|title)/i,m)){At(a,e);continue}if(n.forceKeepAttr)continue;if(At(a,e),!n.keepAttr)continue;if(!ke&&S(/\/>/i,m)){At(a,e);continue}Ie&&u([he,ge,Te],(e=>{m=y(m,e," ")}));const f=pt(e.nodeName);if(Ot(f,s,m)){if(le&&"object"==typeof j&&"function"==typeof j.getAttributeType)if(l);else switch(j.getAttributeType(f,s)){case"TrustedHTML":m=le.createHTML(m);break;case"TrustedScriptURL":m=le.createScriptURL(m)}try{l?e.setAttributeNS(l,a,m):e.setAttribute(a,m),bt(e)?Et(e):p(o.removed)}catch(e){}}}Rt(de.afterSanitizeAttributes,e,null)},Lt=function e(t){let n=null;const o=St(t);for(Rt(de.beforeSanitizeShadowDOM,t,null);n=o.nextNode();)Rt(de.uponSanitizeShadowNode,n,null),wt(n),vt(n),n.content instanceof s&&e(n.content);Rt(de.afterSanitizeShadowDOM,t,null)};return o.sanitize=function(e){let t=arguments.length>1&&void 0!==arguments[1]?arguments[1]:{},n=null,r=null,i=null,l=null;if(rt=!e,rt&&(e="\x3c!--\x3e"),"string"!=typeof e&&!Nt(e)){if("function"!=typeof e.toString)throw b("toString is not a function");if("string"!=typeof(e=e.toString()))throw b("dirty is not a string, aborting")}if(!o.isSupported)return e;if(Pe||gt(t),o.removed=[],"string"==typeof e&&(Xe=!1),Xe){if(e.nodeName){const t=pt(e.nodeName);if(!Ne[t]||ve[t])throw b("root node is forbidden and cannot be sanitized in-place")}}else if(e instanceof R)n=_t("\x3c!----\x3e"),r=n.ownerDocument.importNode(e,!0),r.nodeType===J&&"BODY"===r.nodeName||"HTML"===r.nodeName?n=r:n.appendChild(r);else{if(!Fe&&!Ie&&!ze&&-1===e.indexOf("<"))return le&&We?le.createHTML(e):e;if(n=_t(e),!n)return Fe?null:We?ce:""}n&&He&&Et(n.firstChild);const c=St(Xe?e:n);for(;i=c.nextNode();)wt(i),vt(i),i.content instanceof s&&Lt(i.content);if(Xe)return e;if(Fe){if(Be)for(l=me.call(n.ownerDocument);n.firstChild;)l.appendChild(n.firstChild);else l=n;return(we.shadowroot||we.shadowrootmode)&&(l=fe.call(a,l,!0)),l}let m=ze?n.outerHTML:n.innerHTML;return ze&&Ne["!doctype"]&&n.ownerDocument&&n.ownerDocument.doctype&&n.ownerDocument.doctype.name&&S(K,n.ownerDocument.doctype.name)&&(m="<!DOCTYPE "+n.ownerDocument.doctype.name+">\n"+m),Ie&&u([he,ge,Te],(e=>{m=y(m,e," ")})),le&&We?le.createHTML(m):m},o.setConfig=function(){gt(arguments.length>0&&void 0!==arguments[0]?arguments[0]:{}),Pe=!0},o.clearConfig=function(){ft=null,Pe=!1},o.isValidAttribute=function(e,t,n){ft||gt({});const o=pt(e),r=pt(t);return Ot(o,r,n)},o.addHook=function(e,t){"function"==typeof t&&f(de[e],t)},o.removeHook=function(e,t){if(void 0!==t){const n=m(de[e],t);return-1===n?void 0:d(de[e],n,1)[0]}return p(de[e])},o.removeHooks=function(e){de[e]=[]},o.removeAllHooks=function(){de={afterSanitizeAttributes:[],afterSanitizeElements:[],afterSanitizeShadowDOM:[],beforeSanitizeAttributes:[],beforeSanitizeElements:[],beforeSanitizeShadowDOM:[],uponSanitizeAttribute:[],uponSanitizeElement:[],uponSanitizeShadowNode:[]}},o}();return re}));
</script>
<script>/* marked config: GFM + soft line breaks */
if(typeof marked!=='undefined')marked.setOptions({breaks:true,gfm:true});
</script>
<style>
:root {
  --bg0:#06090d; --bg1:#0b1017; --bg2:#0f1722; --bg3:#141f2e; --bg4:#1a2740;
  --border:#1e2d3d; --border2:#263d57; --border3:#2e4d6b;
  --cyan:#00e5ff; --cyan-dim:rgba(0,229,255,0.08);
  --orange:#ff6b35; --orange-dim:rgba(255,107,53,0.08);
  --purple:#7c3fff; --purple-light:#a87fff; --purple-dim:rgba(124,63,255,0.08);
  --green:#39ff14; --green-dim:rgba(57,255,20,0.08);
  --red:#ff3e5e; --yellow:#ffd700; --gold:#ff9f43;
  --text0:#e8f0f8; --text1:#94a8bc; --text2:#4e6880; --text3:#2a3f54;
  --mono:'Space Mono',monospace; --sans:'Syne',sans-serif;
  --r:5px; --shadow:0 8px 32px rgba(0,0,0,.5); --shadow-lg:0 24px 80px rgba(0,0,0,.8);
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
html,body{height:100%;overflow:hidden;}
body{background:var(--bg0);color:var(--text0);font-family:var(--mono);font-size:13px;line-height:1.5;}
button{cursor:pointer;font-family:inherit;}
input,select,textarea{font-family:inherit;}

/* ── SCROLLBARS visibles ── */
::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-track{background:var(--bg1);}
::-webkit-scrollbar-thumb{background:var(--border3);border-radius:3px;}
::-webkit-scrollbar-thumb:hover{background:#3a5a7a;}
* { scrollbar-width: thin; scrollbar-color: var(--border3) var(--bg1); }

/* ── LAYOUT ── */
.shell{display:flex;flex-direction:column;height:100vh;}

/* ── TOPBAR ── */
.topbar{height:50px;display:flex;align-items:center;padding:0 16px 0 0;
  background:var(--bg1);border-bottom:1px solid var(--border);flex-shrink:0;z-index:100;}
.topbar-brand{width:210px;flex-shrink:0;display:flex;align-items:center;gap:10px;
  padding:0 16px;border-right:1px solid var(--border);height:100%;}
.brand-dot{width:8px;height:8px;border-radius:50%;background:var(--green);
  box-shadow:0 0 10px var(--green);animation:blink 2.5s infinite;}
@keyframes blink{0%,90%,100%{opacity:1}95%{opacity:.2}}
.brand-name{font-family:var(--sans);font-size:13px;font-weight:800;letter-spacing:.15em;color:var(--cyan);}
.brand-version{font-size:9px;color:var(--text2);letter-spacing:.05em;}
.topbar-nav{display:flex;align-items:center;height:100%;padding:0 4px;gap:1px;}
.nav-tab{display:flex;align-items:center;gap:6px;padding:0 13px;height:100%;
  background:none;border:none;color:var(--text2);font-size:11px;letter-spacing:.08em;
  text-transform:uppercase;transition:all .15s;position:relative;white-space:nowrap;}
.nav-tab:hover{color:var(--text0);background:rgba(255,255,255,.03);}
.nav-tab.active{color:var(--tab-color,var(--cyan));background:rgba(255,255,255,.04);}
.nav-tab.active::after{content:'';position:absolute;bottom:0;left:0;right:0;height:2px;
  background:var(--tab-color,var(--cyan));box-shadow:0 0 8px var(--tab-color,var(--cyan));}
.nav-badge{background:var(--tab-color,var(--cyan));color:var(--bg0);
  font-size:9px;font-weight:700;padding:1px 5px;border-radius:2px;min-width:18px;text-align:center;}
.topbar-right{margin-left:auto;display:flex;align-items:center;gap:14px;}
.topbar-stat{display:flex;flex-direction:column;align-items:flex-end;}
.topbar-stat-val{color:var(--cyan);font-weight:700;font-family:var(--sans);font-size:14px;line-height:1;}
.topbar-stat-label{color:var(--text2);font-size:9px;margin-top:2px;}
.topbar-clock{font-size:10px;color:var(--text2);padding-left:14px;border-left:1px solid var(--border);}

/* ── BODY ── */
.body-area{display:flex;flex:1;overflow:hidden;}

/* ── SIDEBAR ── */
.sidebar{width:210px;flex-shrink:0;background:var(--bg1);border-right:1px solid var(--border);
  display:flex;flex-direction:column;overflow-y:auto;overflow-x:hidden;}
.sidebar-section{padding:16px 0 6px;}
.sidebar-label{font-size:9px;color:var(--text3);letter-spacing:.18em;text-transform:uppercase;padding:0 14px 6px;}
.sidebar-link{display:flex;align-items:center;gap:8px;padding:8px 14px;
  font-size:12px;color:var(--text1);border-left:2px solid transparent;transition:all .12s;
  background:none;border-top:none;border-right:none;border-bottom:none;width:100%;text-align:left;}
.sidebar-link:hover{color:var(--text0);background:rgba(255,255,255,.03);}
.sidebar-link.active{color:var(--lc,var(--cyan));border-left-color:var(--lc,var(--cyan));background:rgba(255,255,255,.04);}
.sidebar-link-badge{margin-left:auto;background:var(--lc,var(--border2));color:var(--bg0);
  font-size:9px;font-weight:700;padding:1px 5px;border-radius:2px;}
.sidebar-div{height:1px;background:var(--border);margin:6px 14px;}
.sb-widget{margin:0 10px 12px;background:var(--bg2);border:1px solid var(--border);border-radius:var(--r);padding:10px;}
.sb-widget-title{font-size:9px;color:var(--text3);letter-spacing:.12em;text-transform:uppercase;margin-bottom:8px;}
.conn-row{display:flex;align-items:center;gap:6px;font-size:10px;color:var(--text1);padding:3px 0;}
.conn-dot{width:5px;height:5px;border-radius:50%;flex-shrink:0;}
.gs-row{display:flex;justify-content:space-between;font-size:11px;color:var(--text1);
  padding:4px 0;border-bottom:1px solid var(--border);}
.gs-row:last-child{border-bottom:none;}
.gs-val{font-weight:700;}

/* ── MAIN ── */
.main{flex:1;display:flex;flex-direction:column;overflow:hidden;}
.panel{display:none;flex:1;flex-direction:column;overflow:hidden;}
.panel.active{display:flex;}
.panel-header{display:flex;align-items:center;justify-content:space-between;
  padding:12px 18px;border-bottom:1px solid var(--border);flex-shrink:0;background:var(--bg1);}
.panel-title{font-family:var(--sans);font-size:14px;font-weight:700;display:flex;align-items:center;gap:8px;}
.panel-sub{font-size:10px;color:var(--text2);font-weight:400;font-family:var(--mono);}
.panel-actions{display:flex;gap:7px;align-items:center;}

/* ── BUTTONS ── */
.btn{padding:6px 13px;border:1px solid var(--border2);background:var(--bg3);color:var(--text1);
  font-size:11px;letter-spacing:.05em;border-radius:var(--r);transition:all .15s;
  display:inline-flex;align-items:center;gap:5px;}
.btn:hover{border-color:var(--border3);color:var(--text0);background:var(--bg4);}
.btn-primary{background:var(--cyan);color:var(--bg0);border-color:var(--cyan);font-weight:700;}
.btn-primary:hover{background:#00cfe8;}
.btn-danger{color:var(--red);border-color:rgba(255,62,94,.25);}
.btn-danger:hover{background:rgba(255,62,94,.08);border-color:var(--red);}
.btn-sm{padding:3px 9px;font-size:10px;}
.btn-icon{padding:5px 7px;}

/* ── TAGS (prioridad/impacto, usados en DEV) ── */
.tag{padding:1px 6px;border-radius:2px;font-size:9px;font-weight:700;text-transform:uppercase;}
.tag-alto{background:rgba(255,62,94,.1);color:var(--red);border:1px solid rgba(255,62,94,.2);}
.tag-medio{background:rgba(255,215,0,.08);color:var(--yellow);border:1px solid rgba(255,215,0,.2);}
.tag-bajo{background:rgba(78,104,128,.12);color:var(--text1);border:1px solid var(--border2);}
/* ── ACTION BUTTONS (usados en DEV product/task cards) ── */
.ca-btn{width:20px;height:20px;background:var(--bg3);border:1px solid var(--border2);
  border-radius:3px;color:var(--text1);font-size:9px;display:flex;align-items:center;justify-content:center;transition:all .1s;}
.ca-btn:hover{background:var(--bg4);color:var(--text0);}
.ca-btn.del:hover{color:var(--red);border-color:var(--red);}

/* ══ DEV ══ */
.dev-wrap{flex:1;display:flex;overflow:hidden;}
.dev-body{flex:1;overflow-y:auto;padding:14px 18px;display:flex;flex-direction:column;gap:10px;}
/* product card */
.dev-product{background:var(--bg1);border:1px solid var(--border);border-radius:8px;overflow:hidden;flex-shrink:0;}
.dev-prod-head{display:flex;align-items:center;gap:10px;padding:11px 14px;
  cursor:pointer;user-select:none;transition:background .12s;}
.dev-prod-head:hover{background:rgba(255,255,255,.02);}
.dev-prod-icon{font-size:18px;flex-shrink:0;}
.dev-prod-name{font-family:var(--sans);font-size:13px;font-weight:800;flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.dev-prod-status{font-size:9px;padding:2px 7px;border-radius:2px;font-weight:700;text-transform:uppercase;}
.dev-prod-status.activo{background:var(--green-dim);color:var(--green);border:1px solid rgba(57,255,20,.25);}
.dev-prod-status.idea{background:var(--purple-dim);color:var(--purple-light);border:1px solid rgba(168,127,255,.25);}
.dev-prod-status.pausado{background:rgba(78,104,128,.15);color:var(--text2);border:1px solid var(--border2);}
.dev-prod-status.archivado{background:rgba(78,104,128,.1);color:var(--text3);border:1px solid var(--border);}
.dev-prod-prog{display:flex;align-items:center;gap:8px;margin-left:auto;}
.prog-track{width:80px;height:3px;background:var(--border2);border-radius:2px;overflow:hidden;}
.prog-fill{height:100%;border-radius:2px;transition:width .4s;}
.prog-pct{font-size:10px;width:26px;text-align:right;}
.dev-prod-actions{display:none;gap:3px;flex-shrink:0;}
.dev-prod-head:hover .dev-prod-actions{display:flex;}
.chevron{transition:transform .2s;font-size:9px;color:var(--text2);display:inline-block;flex-shrink:0;}
.chevron.open{transform:rotate(90deg);}
.dev-prod-body{border-top:1px solid var(--border);}
/* module within product */
.dev-module{display:flex;flex-direction:column;}
.dev-mod-head{display:flex;align-items:center;gap:8px;padding:8px 14px 8px 36px;
  background:var(--bg2);cursor:pointer;user-select:none;transition:background .1s;
  border-bottom:1px solid var(--border);}
.dev-mod-head:hover{background:var(--bg3);}
.dev-mod-head.last-closed{border-bottom:none;}
.dev-mod-title{font-size:11px;font-weight:700;color:var(--text1);flex:1;display:flex;align-items:center;gap:6px;}
.dev-mod-meta{font-size:9px;color:var(--text2);font-weight:400;font-family:var(--mono);}
.dev-task-list{display:flex;flex-direction:column;}
.dev-task{display:flex;align-items:flex-start;gap:9px;padding:7px 14px 7px 52px;
  border-bottom:1px solid var(--border);transition:background .1s;}
.dev-task:last-child{border-bottom:none;}
.dev-task:hover{background:rgba(255,255,255,.015);}
.dtc{width:14px;height:14px;border-radius:3px;border:1px solid var(--border2);
  background:var(--bg2);flex-shrink:0;margin-top:2px;cursor:pointer;
  display:flex;align-items:center;justify-content:center;transition:all .12s;font-size:8px;font-weight:700;}
.dtc.checked{background:var(--green);border-color:var(--green);color:var(--bg0);}
.dtc.doing{background:var(--orange);border-color:var(--orange);color:var(--bg0);}
.dt-info{flex:1;min-width:0;}
.dt-name{font-size:11px;color:var(--text0);line-height:1.3;}
.dt-name.done{color:var(--text2);text-decoration:line-through;}
.dt-desc{font-size:10px;color:var(--text2);margin-top:2px;line-height:1.4;}
.dt-tags{display:flex;gap:4px;flex-wrap:wrap;margin-top:4px;}
.sp{padding:1px 5px;border-radius:2px;font-size:8px;font-weight:700;text-transform:uppercase;}
.sp-todo{background:rgba(78,104,128,.15);color:var(--text2);border:1px solid var(--border2);}
.sp-doing{background:rgba(255,107,53,.12);color:var(--orange);border:1px solid rgba(255,107,53,.25);}
.sp-done{background:rgba(57,255,20,.08);color:var(--green);border:1px solid rgba(57,255,20,.2);}
.dt-actions{display:none;gap:3px;flex-shrink:0;}
.dev-task:hover .dt-actions{display:flex;}

/* ══ CAMPAÑA ══ */
.campaign-layout{flex:1;display:grid;grid-template-columns:1fr 340px;overflow:hidden;}
.campaign-left{display:flex;flex-direction:column;overflow:hidden;border-right:1px solid var(--border);}
.camp-metrics-bar{display:flex;gap:8px;padding:12px 16px;border-bottom:1px solid var(--border);flex-shrink:0;background:var(--bg1);overflow-x:auto;}
.mc{background:var(--bg2);border:1px solid var(--border);border-radius:var(--r);padding:10px 16px;min-width:100px;flex-shrink:0;}
.mc-label{font-size:9px;color:var(--text2);letter-spacing:.1em;text-transform:uppercase;margin-bottom:3px;}
.mc-val{font-family:var(--sans);font-size:22px;font-weight:800;line-height:1;}
.mc-sub{font-size:9px;color:var(--text2);margin-top:2px;}
.camp-charts{display:grid;grid-template-columns:1fr 1fr;gap:10px;padding:12px 16px;flex-shrink:0;border-bottom:1px solid var(--border);}
.chart-box{background:var(--bg1);border:1px solid var(--border);border-radius:var(--r);padding:12px;position:relative;}
.chart-box-title{font-size:9px;color:var(--text2);letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px;}
.chart-box canvas{max-height:120px;}
.camp-cards-wrap{flex:1;overflow-y:auto;padding:12px 16px;display:flex;flex-direction:column;gap:8px;}
.status-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;}
.status-card{background:var(--bg1);border:1px solid var(--border);border-top:2px solid var(--border2);
  border-radius:var(--r);padding:16px 14px;text-align:center;}
.status-card.todo{border-top-color:var(--text2);}
.status-card.doing{border-top-color:var(--orange);}
.status-card.done{border-top-color:var(--green);}
.status-val{font-family:var(--sans);font-size:30px;font-weight:800;line-height:1;}
.status-card.todo .status-val{color:var(--text1);}
.status-card.doing .status-val{color:var(--orange);}
.status-card.done .status-val{color:var(--green);}
.status-lbl{font-size:10px;color:var(--text2);letter-spacing:.06em;text-transform:uppercase;margin-top:8px;}

/* right panel - progreso por producto */
.campaign-right{display:flex;flex-direction:column;overflow:hidden;background:var(--bg1);}
.cr-head{padding:12px 14px;border-bottom:1px solid var(--border);font-size:10px;color:var(--text2);letter-spacing:.1em;text-transform:uppercase;flex-shrink:0;}
.cr-body{flex:1;overflow-y:auto;padding:12px 14px;display:flex;flex-direction:column;gap:10px;}
.canal-summary{background:var(--bg2);border:1px solid var(--border);border-radius:var(--r);padding:10px 12px;cursor:pointer;transition:all .12s;}
.canal-summary:hover{border-color:var(--border3);background:var(--bg3);}
.canal-summary.selected{border-color:var(--purple-light);background:var(--purple-dim);}
.cs-head{display:flex;align-items:center;gap:8px;margin-bottom:6px;}
.cs-icon{font-size:16px;}
.cs-name{font-family:var(--sans);font-size:12px;font-weight:700;}
.cs-conv{margin-left:auto;font-size:11px;font-weight:700;}
.cs-bar-wrap{height:3px;background:var(--border2);border-radius:2px;overflow:hidden;}
.cs-bar-fill{height:100%;border-radius:2px;background:var(--purple-light);transition:width .4s;}
.cs-stats{display:flex;gap:8px;margin-top:5px;}
.cs-stat{font-size:10px;color:var(--text2);}
.cs-stat span{color:var(--text1);font-weight:700;}

/* ══ STRATEGY ══ */
.strategy-body{flex:1;overflow-y:auto;padding:14px 18px;display:flex;flex-direction:column;gap:7px;}
.log-entry{background:var(--bg1);border:1px solid var(--border);
  border-left:3px solid var(--lc,var(--border2));border-radius:0 var(--r) var(--r) 0;
  padding:11px 13px;transition:background .12s;position:relative;}
.log-entry:hover{background:var(--bg2);}
.log-header{display:flex;align-items:center;gap:8px;margin-bottom:5px;flex-wrap:wrap;padding-right:56px;}
.log-badge{padding:2px 8px;border-radius:2px;font-size:9px;font-weight:700;letter-spacing:.1em;
  text-transform:uppercase;color:var(--lc);border:1px solid var(--lc);background:rgba(0,0,0,.3);}
.log-title{font-family:var(--sans);font-size:12px;font-weight:700;}
.log-date{font-size:10px;color:var(--text2);margin-left:auto;}
.log-text{font-size:11px;color:var(--text1);line-height:1.6;margin-bottom:6px;white-space:pre-wrap;}
.log-links{display:flex;gap:4px;flex-wrap:wrap;}
.log-link-tag{font-size:9px;padding:2px 7px;border:1px solid var(--border2);border-radius:2px;color:var(--text2);background:var(--bg3);}
.log-actions{position:absolute;top:9px;right:9px;display:none;gap:4px;}
.log-entry:hover .log-actions{display:flex;}

/* ══ GUÍA — layout optimizado ══ */
.guide-body{flex:1;overflow-y:auto;padding:16px 20px;}
.guide-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;max-width:1400px;}
.guide-hero{grid-column:1/-1;background:var(--bg1);border:1px solid var(--border2);
  border-radius:8px;padding:20px 24px;display:flex;align-items:center;gap:28px;position:relative;overflow:hidden;}
.guide-hero::before{content:'';position:absolute;top:-60px;right:-60px;width:220px;height:220px;
  border-radius:50%;background:radial-gradient(circle,rgba(0,229,255,.05) 0%,transparent 70%);}
.guide-hero-left{flex:1;}
.guide-hero-title{font-family:var(--sans);font-size:19px;font-weight:800;color:var(--cyan);letter-spacing:.04em;margin-bottom:6px;}
.guide-hero-sub{font-size:11px;color:var(--text1);line-height:1.65;max-width:520px;}
.guide-hero-tags{display:flex;gap:6px;margin-top:12px;flex-wrap:wrap;}
.guide-hero-tag{padding:3px 10px;background:var(--cyan-dim);border:1px solid rgba(0,229,255,.2);
  border-radius:20px;font-size:9px;color:var(--cyan);letter-spacing:.08em;text-transform:uppercase;}
.guide-hero-right{flex-shrink:0;display:flex;flex-direction:column;gap:6px;min-width:180px;}
.guide-stat-pill{background:var(--bg2);border:1px solid var(--border);border-radius:6px;
  padding:8px 12px;display:flex;align-items:center;gap:10px;}
.guide-stat-val{font-family:var(--sans);font-size:20px;font-weight:800;}
.guide-stat-label{font-size:10px;color:var(--text2);}

.guide-card{background:var(--bg1);border:1px solid var(--border);border-radius:8px;padding:14px;display:flex;flex-direction:column;gap:10px;}
.guide-card-head{display:flex;align-items:center;gap:10px;}
.guide-card-icon{font-size:18px;width:36px;height:36px;display:flex;align-items:center;justify-content:center;
  background:var(--bg3);border:1px solid var(--border2);border-radius:7px;flex-shrink:0;}
.guide-card-title{font-family:var(--sans);font-size:13px;font-weight:700;}
.guide-card-sub{font-size:9px;color:var(--text2);margin-top:1px;letter-spacing:.06em;text-transform:uppercase;}
.guide-steps{display:flex;flex-direction:column;}
.guide-step{display:flex;gap:9px;padding:7px 0;border-bottom:1px solid var(--border);}
.guide-step:last-child{border-bottom:none;}
.gsn{width:18px;height:18px;border-radius:50%;background:var(--bg3);border:1px solid var(--border2);
  display:flex;align-items:center;justify-content:center;font-size:8px;font-weight:700;
  flex-shrink:0;color:var(--text2);margin-top:1px;}
.gsn.hl{background:var(--cyan-dim);border-color:rgba(0,229,255,.4);color:var(--cyan);}
.gs-title{font-size:11px;font-weight:700;color:var(--text0);margin-bottom:2px;}
.gs-desc{font-size:10px;color:var(--text1);line-height:1.45;}
.guide-kbd{display:inline-block;padding:1px 5px;background:var(--bg3);border:1px solid var(--border2);
  border-radius:3px;font-size:8px;color:var(--text1);}
.guide-tip{background:var(--bg3);border:1px solid var(--border);border-left:3px solid var(--gold);
  border-radius:0 4px 4px 0;padding:8px 10px;font-size:10px;color:var(--text1);line-height:1.5;}
.guide-tip strong{color:var(--gold);}

.guide-flow{grid-column:1/-1;background:var(--bg1);border:1px solid var(--border);border-radius:8px;padding:14px 18px;}
.guide-flow-title{font-size:10px;color:var(--text2);letter-spacing:.12em;text-transform:uppercase;margin-bottom:12px;}
.guide-flow-row{display:flex;align-items:center;gap:4px;flex-wrap:wrap;row-gap:8px;}
.gfn{background:var(--bg3);border:1px solid var(--border2);border-radius:6px;
  padding:8px 12px;font-size:10px;text-align:center;flex:1;min-width:90px;}
.gfn-icon{font-size:16px;margin-bottom:3px;}
.gfn-label{font-family:var(--sans);font-weight:700;font-size:10px;}
.gfn-sub{font-size:9px;color:var(--text2);margin-top:1px;}
.gfa{color:var(--text3);font-size:16px;flex-shrink:0;}

.guide-shortcuts{grid-column:1;background:var(--bg1);border:1px solid var(--border);border-radius:8px;padding:14px;}
.guide-shortcut-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:10px;}
.gsr{display:flex;align-items:center;justify-content:space-between;
  padding:6px 8px;background:var(--bg3);border:1px solid var(--border);border-radius:4px;}
.gsr-label{font-size:10px;color:var(--text1);}

.guide-rutina{grid-column:1;background:var(--bg1);border:1px solid var(--border);border-radius:8px;padding:14px;}
.gr-row{display:flex;gap:10px;padding:8px 0;border-bottom:1px solid var(--border);align-items:flex-start;}
.gr-row:last-child{border-bottom:none;}
.gr-icon{width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:13px;flex-shrink:0;}

/* ══ MODAL ══ */
.overlay{position:fixed;inset:0;z-index:750;background:rgba(6,9,13,.85);
  display:none;align-items:center;justify-content:center;padding:20px;backdrop-filter:blur(4px);}
.overlay.open{display:flex;animation:fadeIn .15s ease;}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
.modal{background:var(--bg2);border:1px solid var(--border3);border-radius:8px;
  width:500px;max-width:100%;max-height:88vh;overflow-y:auto;box-shadow:var(--shadow-lg);
  animation:slideUp .18s ease;}
@keyframes slideUp{from{transform:translateY(10px);opacity:0}to{transform:translateY(0);opacity:1}}
.modal-head{display:flex;align-items:center;justify-content:space-between;
  padding:14px 18px;border-bottom:1px solid var(--border);position:sticky;top:0;background:var(--bg2);z-index:1;}
.modal-title{font-family:var(--sans);font-size:14px;font-weight:700;}
.modal-close{width:26px;height:26px;border-radius:4px;background:var(--bg3);border:1px solid var(--border2);
  color:var(--text1);font-size:13px;display:flex;align-items:center;justify-content:center;}
.modal-close:hover{color:var(--text0);}
.modal-body{padding:18px;display:flex;flex-direction:column;gap:13px;}
.modal-foot{padding:12px 18px;border-top:1px solid var(--border);display:flex;justify-content:space-between;align-items:center;}
.modal-confirm{width:400px;}
.confirm-msg{font-size:12px;color:var(--text1);line-height:1.6;}
.f-group{display:flex;flex-direction:column;gap:4px;}
.f-row{display:grid;grid-template-columns:1fr 1fr;gap:10px;}
.f-label{font-size:9px;color:var(--text2);letter-spacing:.1em;text-transform:uppercase;}
.f-input,.f-select,.f-textarea{background:var(--bg3);border:1px solid var(--border2);
  border-radius:4px;color:var(--text0);font-size:12px;padding:8px 10px;width:100%;outline:none;transition:border-color .15s;}
.f-input:focus,.f-select:focus,.f-textarea:focus{border-color:var(--cyan);}
.f-select option{background:var(--bg2);}
.f-textarea{resize:vertical;min-height:75px;line-height:1.5;}
.f-hint{font-size:9px;color:var(--text2);}

/* ── TOAST ── */
.toast-wrap{position:fixed;bottom:18px;right:18px;z-index:9999;display:flex;flex-direction:column;gap:5px;}
.toast{background:var(--bg3);border:1px solid var(--border2);border-radius:var(--r);
  padding:9px 14px;font-size:11px;color:var(--text0);box-shadow:var(--shadow);
  animation:stIn .2s ease,stOut .3s ease 2.7s forwards;display:flex;align-items:center;gap:7px;min-width:200px;}
@keyframes stIn{from{transform:translateX(16px);opacity:0}to{transform:translateX(0);opacity:1}}
@keyframes stOut{to{opacity:0;transform:translateX(8px)}}
.toast.success{border-left:3px solid var(--green);}
.toast.error{border-left:3px solid var(--red);}
.toast.info{border-left:3px solid var(--cyan);}
.empty{text-align:center;padding:40px;color:var(--text2);}
.empty-icon{font-size:32px;margin-bottom:10px;opacity:.4;}
.active-filter{border-color:var(--green)!important;color:var(--green)!important;background:var(--green-dim)!important;}

/* ══ CHATBOT ══ */
.chat-fab{position:fixed;bottom:22px;right:22px;z-index:800;width:48px;height:48px;
  border-radius:50%;background:var(--purple);border:2px solid rgba(124,63,255,.4);
  color:#fff;font-size:20px;cursor:pointer;box-shadow:0 4px 24px rgba(124,63,255,.5);
  transition:all .2s;display:flex;align-items:center;justify-content:center;}
.chat-fab:hover{transform:scale(1.08);box-shadow:0 6px 32px rgba(124,63,255,.7);}
.chat-fab.open{background:var(--bg3);border-color:var(--border2);font-size:14px;}
.chat-panel{position:fixed;bottom:82px;right:22px;z-index:800;width:370px;
  background:var(--bg2);border:1px solid var(--border3);border-radius:12px;
  box-shadow:var(--shadow-lg);display:none;flex-direction:column;overflow:hidden;
  max-height:520px;}
.chat-panel.open{display:flex;animation:slideUp .18s ease;}
.chat-head{padding:11px 15px;border-bottom:1px solid var(--border);background:var(--bg3);
  display:flex;align-items:center;gap:10px;flex-shrink:0;}
.chat-head-icon{font-size:18px;width:32px;height:32px;background:var(--purple-dim);
  border:1px solid rgba(124,63,255,.3);border-radius:8px;display:flex;align-items:center;
  justify-content:center;flex-shrink:0;}
.chat-head-info{flex:1;min-width:0;}
.chat-head-title{font-family:var(--sans);font-weight:800;font-size:13px;color:var(--purple-light);}
.chat-head-sub{font-size:9px;color:var(--text2);letter-spacing:.07em;text-transform:uppercase;}
.chat-online{width:7px;height:7px;border-radius:50%;background:var(--green);
  box-shadow:0 0 8px var(--green);flex-shrink:0;}
.chat-clear{background:none;border:none;color:var(--text2);cursor:pointer;font-size:14px;
  padding:4px;border-radius:6px;transition:all .15s;line-height:1;}
.chat-clear:hover{color:var(--red,#ef4444);background:rgba(239,68,68,.1);}
.chat-messages{flex:1;overflow-y:auto;padding:12px;display:flex;flex-direction:column;
  gap:8px;min-height:80px;}
.chat-msg{display:flex;flex-direction:column;}
.chat-msg.user{align-items:flex-end;}
.chat-msg.bot{align-items:flex-start;}
.chat-bubble{padding:8px 12px;border-radius:10px;font-size:12px;line-height:1.55;
  max-width:92%;word-break:break-word;}
.chat-msg.user .chat-bubble{background:var(--purple);color:#fff;border-bottom-right-radius:3px;}
.chat-msg.bot .chat-bubble{background:var(--bg4);color:var(--text0);border-bottom-left-radius:3px;
  border:1px solid var(--border2);}
.chat-typing-wrap{display:flex;align-items:center;gap:6px;padding:8px 12px;}
.chat-dot{width:6px;height:6px;border-radius:50%;background:var(--purple-light);
  animation:typingPulse 1.2s infinite;}
.chat-dot:nth-child(2){animation-delay:.2s;}
.chat-dot:nth-child(3){animation-delay:.4s;}
@keyframes typingPulse{0%,80%,100%{opacity:.25;transform:scale(.7)}40%{opacity:1;transform:scale(1)}}
.chat-input-wrap{padding:9px 12px;border-top:1px solid var(--border);display:flex;
  gap:7px;flex-shrink:0;background:var(--bg2);}
.chat-input{flex:1;background:var(--bg3);border:1px solid var(--border2);border-radius:6px;
  color:var(--text0);font-size:12px;padding:7px 10px;outline:none;resize:none;
  font-family:var(--mono);line-height:1.4;max-height:80px;}
.chat-input:focus{border-color:var(--purple);}
.chat-send{width:34px;height:34px;background:var(--purple);border:none;border-radius:6px;
  color:#fff;font-size:14px;cursor:pointer;flex-shrink:0;display:flex;align-items:center;
  justify-content:center;transition:all .15s;align-self:flex-end;}
.chat-send:hover:not(:disabled){background:var(--purple-light);}
.chat-send:disabled{opacity:.35;cursor:not-allowed;}
.chat-tool-badge{display:inline-flex;align-items:center;gap:4px;font-size:9px;
  color:var(--purple-light);background:var(--purple-dim);border:1px solid rgba(124,63,255,.3);
  border-radius:3px;padding:1px 6px;margin-top:3px;}
/* ══ CHAT MARKDOWN ══ */
.chat-bubble p{margin:0 0 6px}
.chat-bubble p:last-child{margin-bottom:0}
.chat-bubble ul,.chat-bubble ol{margin:4px 0;padding-left:18px}
.chat-bubble li{margin-bottom:2px}
.chat-bubble code{background:var(--bg3);padding:1px 5px;border-radius:3px;font-family:var(--mono);font-size:11px;color:var(--purple-light)}
.chat-bubble pre{background:rgba(0,0,0,.25);padding:8px 10px;border-radius:6px;overflow-x:auto;margin:6px 0;border:1px solid var(--border2)}
.chat-bubble pre code{background:none;padding:0;font-size:11px;color:inherit}
.chat-bubble blockquote{border-left:2px solid var(--purple);padding:2px 10px;margin:6px 0;color:var(--text2)}
.chat-bubble h1,.chat-bubble h2,.chat-bubble h3,.chat-bubble h4,.chat-bubble h5,.chat-bubble h6{margin:8px 0 4px;font-weight:700;line-height:1.3;font-family:var(--sans)}
.chat-bubble h1{font-size:14px}.chat-bubble h2{font-size:13px}.chat-bubble h3{font-size:12px}
.chat-bubble h4,.chat-bubble h5,.chat-bubble h6{font-size:11px}
.chat-bubble a{color:var(--purple-light);text-decoration:underline}
.chat-bubble hr{border:none;border-top:1px solid var(--border2);margin:8px 0}
.chat-bubble table{border-collapse:collapse;width:100%;margin:6px 0;font-size:11px}
.chat-bubble th,.chat-bubble td{border:1px solid var(--border2);padding:4px 6px;text-align:left}
.chat-bubble th{background:var(--bg3);font-weight:700}

/* ===== RESPONSIVE — REGISTRO =====
   ledger: P1=done P2=done P3=done P4=done P5=done P6=done P7=done P8=done P9=done
   Cada fase voltea su tag a =done y añade su marcador abajo.
   Cascada mobile-last: desktop -> mas estrecho. Todo gateado tras @media. */

/* [P7] base: hamburguesa y scrim ocultos en desktop */
.hamburger{display:none;}
.sidebar-scrim{display:none;}
/* [P8] base: bottom nav oculto en desktop */
.mobile-bottom-nav{display:none;}
/* ══ MOBILE BOTTOM NAV — base styles ══ */
.mobile-bottom-nav{position:fixed;bottom:0;left:0;right:0;height:56px;
  background:var(--bg1);border-top:1px solid var(--border);z-index:700;
  justify-content:space-around;align-items:center;
  padding-bottom:env(safe-area-inset-bottom,0px);}
.mb-nav-tab{display:flex;flex-direction:column;align-items:center;gap:2px;
  background:none;border:none;color:var(--text2);padding:6px 8px;
  position:relative;min-width:0;cursor:pointer;font-family:inherit;
  transition:color .15s;-webkit-tap-highlight-color:transparent;}
.mb-nav-tab:active{opacity:.7;}
.mb-nav-tab.active{color:var(--tab-color,var(--cyan));}
.mb-nav-tab[data-panel="dev"]{--tab-color:var(--orange);}
.mb-nav-tab[data-panel="campaign"]{--tab-color:var(--purple-light);}
.mb-nav-tab[data-panel="strategy"]{--tab-color:var(--green);}
.mb-nav-tab[data-panel="guide"]{--tab-color:var(--gold);}
.mb-chat-btn{--tab-color:var(--purple);}
.mb-icon{font-size:18px;line-height:1;}
.mb-label{font-size:9px;letter-spacing:.05em;text-transform:uppercase;}
.mb-badge{position:absolute;top:2px;right:2px;background:var(--tab-color,var(--orange));
  color:var(--bg0);font-size:8px;font-weight:700;padding:1px 4px;
  border-radius:2px;min-width:15px;text-align:center;line-height:1.3;}


/* ── TABLET <=1024 ── */
@media (max-width:1024px){
  /* [P1] sidebar off-canvas <=768 (preludio: encoge a 1024) */
  .sidebar{width:170px;}
  .topbar-brand{width:170px;}
  /* [P3] grids -> 1col (campaign 2col->1, guide 3col->2)
     minmax(0,1fr): evita que canvas/contenido fuercen la columna mas ancha
     que el viewport (causa de desborde a la derecha recortado por overflow-x) */
  .campaign-layout{grid-template-columns:minmax(0,1fr);}
  .guide-grid{grid-template-columns:repeat(2,minmax(0,1fr));}
  /* mitigacion overflow anidado: el panel scrollea como un todo */
  #panel-campaign.active{overflow-y:auto;}
  .campaign-layout{overflow:visible;min-width:0;}
  .campaign-left,.campaign-right{overflow:visible;border-right:none;min-width:0;}
  .camp-cards-wrap,.cr-body{overflow:visible;flex:none;min-width:0;}
  /* cadena flex sin min-width:0 no encoge bajo su contenido -> desborde */
  .main,.panel{min-width:0;}
  /* [P4] topbar 1-row, nav hidden <=768 (preludio: compacta a 1024) */
  .topbar-clock{display:none;}
  .nav-tab{padding:0 9px;font-size:10px;}
}

/* ── MOBILE <=768 ── */
@media (max-width:768px){
  /* [P1] sidebar off-canvas <=768 */
  .sidebar{position:fixed;top:50px;left:0;bottom:0;width:240px;z-index:600;
    transform:translateX(-100%);transition:transform .22s ease;box-shadow:var(--shadow-lg);}
  .sidebar.open{transform:translateX(0);}
  .topbar-brand{width:auto;border-right:none;padding:0 12px;}
  .sidebar-scrim.open{display:block;position:fixed;inset:50px 0 0 0;
    background:rgba(0,0,0,.5);z-index:550;}
  /* [P2] chat/modal/toast fluid */
  .chat-panel{width:340px;}
  /* [P3] grids -> 1col */
  .camp-charts{grid-template-columns:1fr;}
  .guide-grid{grid-template-columns:1fr;}
  .guide-hero{flex-direction:column;align-items:flex-start;gap:16px;}
  .guide-hero-right{min-width:0;width:100%;flex-direction:row;flex-wrap:wrap;}
  /* [P7] hamburguesa visible */
  .hamburger{display:flex;align-items:center;justify-content:center;width:38px;height:38px;
    background:none;border:none;color:var(--text0);font-size:18px;cursor:pointer;flex-shrink:0;margin-left:auto;}
  /* [P4] topbar 1-row, nav hidden <=768 */
  .topbar-nav{display:none;}
  .topbar-stat:nth-child(2),.topbar-stat:nth-child(3){display:none;}
  .topbar-right{gap:8px;}
  /* [P6] typo/spacing + red de overflow */
  html,body{overflow-x:hidden;}
  .panel-header{padding:10px 12px;flex-wrap:wrap;gap:6px;}
  .dev-task{padding-left:16px;}
  .guide-body,.strategy-body{padding:12px;}
  .camp-metrics-bar{gap:6px;}
  /* [P8] bottom nav bar */
  .mobile-bottom-nav{display:flex;}
  .main{padding-bottom:56px;padding-bottom:calc(56px + env(safe-area-inset-bottom,0px));}
  /* adjust FAB so it doesn't hide behind bottom nav */
  .chat-fab{bottom:calc(22px + 56px);}
  .chat-panel{bottom:calc(82px + 56px);}
  /* [P6] hide progress bars on mobile — keeps only % and count */
  .prog-track{display:none;}
  /* gap real bajo el bottom nav: altura nav (56px) + respiro + safe-area
     para que la ultima tarea/log no quede pegada al nav */
  .dev-body,.strategy-body,.guide-body,.camp-cards-wrap,.cr-body{
    padding-bottom:calc(56px + 28px + env(safe-area-inset-bottom,0px));}
  /* panel-actions wrap para filtros de estrategia */
  .panel-actions{flex-wrap:wrap;gap:4px;}
  .campaign-right{border-top:1px solid var(--border);}
  /* bottom padding forzado para DEV, Estrategia y Guia */
  .dev-body{padding:14px 18px calc(56px + 28px + env(safe-area-inset-bottom,0px)) 18px!important;}
  .strategy-body{padding:12px 18px calc(56px + 28px + env(safe-area-inset-bottom,0px)) 18px!important;}
  .guide-body{padding:12px 16px calc(56px + 28px + env(safe-area-inset-bottom,0px)) 16px!important;}
  /* campaña: padding-right extra para que tarjetas y Productos no se peguen al lateral */
  .camp-cards-wrap{padding:12px 16px calc(56px + 28px + env(safe-area-inset-bottom,0px)) 16px!important;}
  .cr-body{padding:12px 16px calc(56px + 28px + env(safe-area-inset-bottom,0px)) 16px!important;}
}

/* ── SMALL MOBILE <=480 ── */
@media (max-width:480px){
  /* [P2] chat/modal/toast fluid */
  .chat-panel{width:calc(100vw - 24px);left:12px;right:12px;max-height:70vh;bottom:76px;}
  .chat-fab{bottom:16px;right:16px;}
  .toast-wrap{left:12px;right:12px;bottom:12px;}
  .toast{min-width:0;width:100%;}
  .modal{width:100%;max-height:92vh;}
  .modal-body{padding:14px;}
  .f-row{grid-template-columns:1fr;}
  /* [P3] grids -> 1col */
  .status-grid{grid-template-columns:repeat(3,minmax(0,1fr));gap:6px;}
  .status-card{padding:10px 6px;}
  .status-val{font-size:18px;}
  .status-lbl{font-size:8px;}
  /* [P4] topbar 1-row, nav hidden <=768 (small: oculta stats) */
  .topbar-right{display:none;}
  /* [P5] touch actions visible (small) */
  .btn{padding:8px 12px;}
  /* [P6] typo/spacing: reduce numericos grandes */
  .mc-val{font-size:18px;}
  .guide-hero-title{font-size:16px;}
  /* [P9] bottom sheet modals */
  .overlay{align-items:flex-end;padding:0;}
  .modal{border-radius:12px 12px 0 0;margin:0;max-height:90vh;}
  /* compactar filtros estrategia */
  #log-filter .btn-sm{padding:2px 6px;font-size:9px;}
}

/* ── TOUCH (sin hover / puntero grueso) ── */
@media (hover:none),(pointer:coarse){
  /* [P5] touch actions visible: acciones hover-only siempre mostradas */
  .dt-actions{display:flex!important;}
  .log-actions{display:flex!important;}
  .nav-tab,.sidebar-link,.btn,.modal-close,.chat-send{min-height:38px;}
  .dtc{width:18px;height:18px;}
  /* [P10] touch targets mas grandes para acciones */
  .ca-btn{width:32px;height:32px;font-size:12px;}
  /* logs colapsados en touch: tap sobre el cuerpo expande */
  .log-text{display:-webkit-box;-webkit-line-clamp:5;-webkit-box-orient:vertical;overflow:hidden;cursor:pointer;}
  .log-text.expanded{display:block;-webkit-line-clamp:none;overflow:visible;}
}
/* ===== /RESPONSIVE ===== */
.chat-fab.modal-open{bottom:auto;top:16px;right:16px;}
</style>
</head>
<body>
<div class="shell">

<!-- TOPBAR -->
<div class="topbar">
  <div class="topbar-brand">
    <div class="brand-dot"></div>
    <div><div class="brand-name">OPENAETH</div><div class="brand-version">COMMAND CORE v2.1</div></div>
  </div>
  <div class="topbar-nav">
    <button class="nav-tab active" style="--tab-color:var(--orange)" data-panel="dev" onclick="switchPanel('dev',this)">
      ⚙️ DEV <span class="nav-badge" id="nb-dev" style="background:var(--orange)">0</span><span id="nb-dev-tasks" style="font-size:9px;color:var(--text2);margin-left:2px"></span>
    </button>
    <button class="nav-tab" style="--tab-color:var(--purple-light)" data-panel="campaign" onclick="switchPanel('campaign',this)">
      🚀 CAMPAÑA
    </button>
    <button class="nav-tab" style="--tab-color:var(--green)" data-panel="strategy" onclick="switchPanel('strategy',this)">
      🧠 ESTRATEGIA <span class="nav-badge" id="nb-str" style="background:var(--green)">0</span>
    </button>
    <button class="nav-tab" style="--tab-color:var(--gold)" data-panel="guide" onclick="switchPanel('guide',this)">
      📖 GUÍA
    </button>
  </div>
  <div class="topbar-right">
    <div class="topbar-stat"><div class="topbar-stat-val" id="ts-todo">—</div><div class="topbar-stat-label">tasks todo</div></div>
    <div class="topbar-stat"><div class="topbar-stat-val" style="color:var(--orange)" id="ts-doing">—</div><div class="topbar-stat-label">en doing</div></div>
    <div class="topbar-stat"><div class="topbar-stat-val" style="color:var(--purple-light)" id="ts-logs">—</div><div class="topbar-stat-label">logs</div></div>
    <div class="topbar-clock" id="clock">—</div>
  </div>
  <button class="hamburger" id="hamburger" aria-label="Menú" aria-expanded="false" onclick="toggleSidebar()">☰</button>
</div>

<div class="body-area">

<!-- SIDEBAR -->
<div class="sidebar">
  <div class="sidebar-section">
    <div class="sidebar-label">Secciones</div>
    <button class="sidebar-link active" style="--lc:var(--orange)" id="sl-dev" onclick="switchPanel('dev')">
      ⚙️ DEV <span class="sidebar-link-badge" style="background:var(--orange)" id="slb-dev">0</span>
    </button>
    <button class="sidebar-link" style="--lc:var(--purple-light)" id="sl-campaign" onclick="switchPanel('campaign')">🚀 Campaña</button>
    <button class="sidebar-link" style="--lc:var(--green)" id="sl-strategy" onclick="switchPanel('strategy')">
      🧠 Estrategia <span class="sidebar-link-badge" style="background:var(--green)" id="slb-str">0</span>
    </button>
    <button class="sidebar-link" style="--lc:var(--gold)" id="sl-guide" onclick="switchPanel('guide')">📖 Guía</button>
  </div>
  <div class="sidebar-div"></div>
  <div class="sidebar-section">
    <div class="sidebar-label">Conexiones</div>
    <div class="sb-widget">
      <div class="conn-row"><div class="conn-dot" style="background:var(--orange)"></div>Task → Producto</div>
      <div class="conn-row"><div class="conn-dot" style="background:var(--green)"></div>Cumplimiento → Readiness</div>
      <div class="conn-row"><div class="conn-dot" style="background:var(--purple-light)"></div>Producto → Campaña</div>
      <div class="conn-row"><div class="conn-dot" style="background:var(--cyan)"></div>Insight → Decisión</div>
    </div>
  </div>
  <div class="sidebar-section">
    <div class="sidebar-label">Estado global</div>
    <div class="sb-widget">
      <div class="gs-row"><span>Tasks pendientes</span><span class="gs-val" style="color:var(--orange)" id="gs-t">—</span></div>
      <div class="gs-row"><span>Productos activos</span><span class="gs-val" style="color:var(--orange)" id="gs-prod">—</span></div>
      <div class="gs-row"><span>Dev completado</span><span class="gs-val" style="color:var(--green)" id="gs-p">—</span></div>
      <div class="gs-row"><span>Logs estratégicos</span><span class="gs-val" style="color:var(--purple-light)" id="gs-l">—</span></div>
    </div>
  </div>
</div>
<div class="sidebar-scrim" id="sidebar-scrim" onclick="closeSidebar()"></div>

<!-- MAIN -->
<div class="main">

  <!-- DEV -->
  <div class="panel active" id="panel-dev">
    <div class="panel-header">
      <div class="panel-title" style="color:var(--orange)">⚙️ DEV <span class="panel-sub">— Portafolio de productos</span></div>
      <div class="panel-actions">
        <button class="btn btn-sm" onclick="expandAllProducts()">Expandir todo</button>
        <button class="btn btn-sm" onclick="collapseAllProducts()">Colapsar todo</button>
        <button class="btn btn-sm" onclick="openProductModal()">+ Producto</button>
        <button class="btn btn-sm btn-primary" onclick="openTaskModal()">+ Task</button>
      </div>
    </div>
    <div class="dev-wrap">
      <div class="dev-body" id="dev-body"></div>
    </div>
  </div>

  <!-- CAMPAÑA — Dashboard de la Desarrolladora -->
  <div class="panel" id="panel-campaign">
    <div class="panel-header">
      <div class="panel-title" style="color:var(--purple-light);width:100%">🚀 CAMPAÑA <span class="panel-sub">— Dashboard de la Desarrolladora · datos en vivo del módulo DEV</span><button class="btn" style="margin-left:auto;flex-shrink:0" onclick="loadDevMetrics().then(renderCampaign)">↻ Actualizar</button></div>
    </div>
    <div class="campaign-layout">
      <div class="campaign-left">
        <div class="camp-metrics-bar" id="camp-metrics-bar"></div>
        <div class="camp-charts" id="camp-charts">
          <div class="chart-box"><div class="chart-box-title">Tareas completadas · últimos 7 días</div><canvas id="chart-weekly"></canvas></div>
          <div class="chart-box"><div class="chart-box-title">Distribución de tareas por estado</div><canvas id="chart-status"></canvas></div>
        </div>
        <div class="camp-cards-wrap" id="camp-cards-wrap"></div>
      </div>
      <div class="campaign-right">
        <div class="cr-head">Productos · progreso</div>
        <div class="cr-body" id="camp-right-body"></div>
      </div>
    </div>
  </div>

  <!-- ESTRATEGIA -->
  <div class="panel" id="panel-strategy">
    <div class="panel-header">
      <div class="panel-title" style="color:var(--green)">🧠 ESTRATEGIA <span class="panel-sub">— Cerebro externo</span></div>
      <div class="panel-actions">
        <div id="log-filter" style="display:flex;gap:3px">
          <button class="btn btn-sm active-filter" onclick="setLogFilter(this,'')">Todos</button>
          <button class="btn btn-sm" onclick="setLogFilter(this,'Decision')">💡 Decisión</button>
          <button class="btn btn-sm" onclick="setLogFilter(this,'Insight')">🔍 Insight</button>
          <button class="btn btn-sm" onclick="setLogFilter(this,'Oportunidad')">🚀 Oportunidad</button>
          <button class="btn btn-sm" onclick="setLogFilter(this,'Riesgo')">⚠️ Riesgo</button>
          <button class="btn btn-sm" onclick="setLogFilter(this,'Aprendizaje')">🧠 Aprendizaje</button>
          <button class="btn btn-sm" onclick="setLogFilter(this,'Objetivo')">🎯 Objetivo</button>
          <button class="btn btn-sm" onclick="setLogFilter(this,'Hipótesis')">🔬 Hipótesis</button>
          <button class="btn btn-sm" onclick="setLogFilter(this,'Hito')">🏆 Hito</button>
        </div>
        <button class="btn btn-primary" onclick="openLogModal()">+ Log</button>
      </div>
    </div>
    <div class="strategy-body" id="strategy-body"></div>
  </div>

  <!-- GUÍA -->
  <div class="panel" id="panel-guide">
    <div class="panel-header">
      <div class="panel-title" style="color:var(--gold)">📖 GUÍA <span class="panel-sub">— Cómo operar el sistema</span></div>
    </div>
    <div class="guide-body">
      <div class="guide-grid">

        <!-- Hero compacto con stats en vivo -->
        <div class="guide-hero">
          <div class="guide-hero-left">
            <div class="guide-hero-title">OpenAETH Command Core</div>
            <div class="guide-hero-sub">Sistema operativo para tu startup, centrado en ejecución. Gestión de tareas con IA integrada — el cumplimiento del módulo DEV alimenta las métricas de campaña, y la estrategia captura el aprendizaje.</div>
            <div class="guide-hero-tags">
              <span class="guide-hero-tag">⚙️ DEV</span>
              <span class="guide-hero-tag">🚀 CAMPAÑA</span>
              <span class="guide-hero-tag">🧠 ESTRATEGIA</span>
              <span class="guide-hero-tag">🤖 IA</span>
            </div>
          </div>
          <div class="guide-hero-right">
            <div class="guide-stat-pill"><div class="guide-stat-val" style="color:var(--orange)" id="g-st">—</div><div class="guide-stat-label">tasks activas</div></div>
            <div class="guide-stat-pill"><div class="guide-stat-val" style="color:var(--orange)" id="g-prod">—</div><div class="guide-stat-label">productos activos</div></div>
            <div class="guide-stat-pill"><div class="guide-stat-val" style="color:var(--green)" id="g-sp">—</div><div class="guide-stat-label">dev completado</div></div>
            <div class="guide-stat-pill"><div class="guide-stat-val" style="color:var(--purple-light)" id="g-sl">—</div><div class="guide-stat-label">logs estratégicos</div></div>
          </div>
        </div>

        <!-- Flujo compacto -->
        <div class="guide-flow">
          <div class="guide-flow-title">🔗 Flujo del sistema</div>
          <div class="guide-flow-row">
            <div class="gfn"><div class="gfn-icon">🤖</div><div class="gfn-label">IA</div><div class="gfn-sub">cargá productos/tasks</div></div>
            <div class="gfa">→</div>
            <div class="gfn"><div class="gfn-icon">⚙️</div><div class="gfn-label">Task DEV</div><div class="gfn-sub">backlog por módulo</div></div>
            <div class="gfa">→</div>
            <div class="gfn"><div class="gfn-icon">✅</div><div class="gfn-label">Cumplimiento</div><div class="gfn-sub">done / total</div></div>
            <div class="gfa">→</div>
            <div class="gfn"><div class="gfn-icon">🚀</div><div class="gfn-label">Campaña</div><div class="gfn-sub">readiness→conv</div></div>
            <div class="gfa">→</div>
            <div class="gfn"><div class="gfn-icon">🔍</div><div class="gfn-label">Insight</div><div class="gfn-sub">Estrategia</div></div>
          </div>
        </div>

        <!-- AETHY card -->
        <div class="guide-card">
          <div class="guide-card-head">
            <div class="guide-card-icon">🤖</div>
            <div><div class="guide-card-title" style="color:var(--purple-light)">AETHY</div><div class="guide-card-sub">Asistente IA · Qwen3-32B · gestión por chat</div></div>
          </div>
          <div class="guide-steps">
            <div class="guide-step"><div class="gsn hl">1</div><div><div class="gs-title">Crear</div><div class="gs-desc"><em>"Cargá el producto X con las tareas A, B, C"</em> — producto + tareas en una sola operación.</div></div></div>
            <div class="guide-step"><div class="gsn">2</div><div><div class="gs-title">Modificar y mover</div><div class="gs-desc"><em>"marcá Y como done"</em>, <em>"poné X en pausado"</em>, <em>"mové la task Z a otro producto"</em>. Por nombre, sin IDs.</div></div></div>
            <div class="guide-step"><div class="gsn">3</div><div><div class="gs-title">Consultar avance</div><div class="gs-desc"><em>"¿cómo viene el avance?"</em>, <em>"qué tasks tiene X"</em>. Reporta % completado y estados reales.</div></div></div>
            <div class="guide-step"><div class="gsn">4</div><div><div class="gs-title">Eliminar y registrar</div><div class="gs-desc"><em>"borrá la task W"</em>, <em>"registrá esta decisión en Estrategia"</em>. Lo destructivo lo confirma antes.</div></div></div>
          </div>
          <div class="guide-tip"><strong>Tip:</strong> AETHY busca todo por nombre. Los paneles DEV y Campaña se refrescan solos tras cada acción.</div>
        </div>

        <!-- ARTEFACTOS card -->
        <div class="guide-card">
          <div class="guide-card-head">
            <div class="guide-card-icon">🗣️</div>
            <div><div class="guide-card-title" style="color:var(--purple-light)">HABLAR CON AETHY</div><div class="guide-card-sub">Artefactos · formas naturales de operar</div></div>
          </div>
          <div class="guide-steps">
            <div class="guide-step"><div class="gsn hl">📥</div><div><div class="gs-title">Volcado</div><div class="gs-desc"><em>"Tengo en la cabeza: arreglar el login, el dashboard carga lento, falta el email de bienvenida"</em> → lo parte en tareas, infiriendo módulo y prioridad.</div></div></div>
            <div class="guide-step"><div class="gsn">💡</div><div><div class="gs-title">Decisión</div><div class="gs-desc"><em>"Decidí usar Mongo sobre Postgres porque el esquema es flexible"</em> → registra un log de Decisión con el texto completo y literal.</div></div></div>
            <div class="guide-step"><div class="gsn">📓</div><div><div class="gs-title">Bitácora del día</div><div class="gs-desc"><em>"Hoy avancé en X, me trabé con Y, aprendí Z"</em> → te propone marcar tareas y crear logs; confirma antes de escribir.</div></div></div>
          </div>
          <div class="guide-tip"><strong>Tip:</strong> No hay que recordar comandos: hablá natural. AETHY reconoce estos patrones por intención y siempre responde con la misma estructura.</div>
        </div>

        <!-- DEV card -->
        <div class="guide-card">
          <div class="guide-card-head">
            <div class="guide-card-icon">⚙️</div>
            <div><div class="guide-card-title" style="color:var(--orange)">DEV</div><div class="guide-card-sub">Módulos · Tasks · Progreso</div></div>
          </div>
          <div class="guide-steps">
            <div class="guide-step"><div class="gsn hl">1</div><div><div class="gs-title">Portafolio de productos</div><div class="gs-desc">Cada producto tiene sus módulos. Convbot, Bitacora, PromptForge, TerraGazette y más.</div></div></div>
            <div class="guide-step"><div class="gsn">2</div><div><div class="gs-title">Crear / editar productos</div><div class="gs-desc">+ Producto en el header. Nombre, ícono, estado (activo/idea/pausado/archivado) y color.</div></div></div>
            <div class="guide-step"><div class="gsn">3</div><div><div class="gs-title">Tasks por módulo</div><div class="gs-desc">+ Task en el header del módulo. Seleccioná producto y módulo al crear. Progreso en tiempo real.</div></div></div>
            <div class="guide-step"><div class="gsn">4</div><div><div class="gs-title">Marcar y editar</div><div class="gs-desc">Click en el checkbox → done. Hover task → <span class="guide-kbd">✎</span> para editar estado, prioridad e impacto.</div></div></div>
          </div>
          <div class="guide-tip"><strong>Tip:</strong> Usá el estado "idea" para productos en exploración — aparecen con badge diferente sin contaminar el foco.</div>
        </div>

        <!-- CAMPAÑA card -->
        <div class="guide-card">
          <div class="guide-card-head">
            <div class="guide-card-icon">🚀</div>
            <div><div class="guide-card-title" style="color:var(--purple-light)">CAMPAÑA</div><div class="guide-card-sub">Dashboard de la Desarrolladora</div></div>
          </div>
          <div class="guide-steps">
            <div class="guide-step"><div class="gsn hl">1</div><div><div class="gs-title">Datos en vivo de DEV</div><div class="gs-desc">No se carga nada a mano: todo se deriva de productos y tareas del módulo DEV.</div></div></div>
            <div class="guide-step"><div class="gsn">2</div><div><div class="gs-title">Métricas y estados</div><div class="gs-desc">Proyectos, % completado, y tareas por estado: Por hacer / Haciendo / Hecho.</div></div></div>
            <div class="guide-step"><div class="gsn">3</div><div><div class="gs-title">Gráfico semanal</div><div class="gs-desc">Tareas completadas en los últimos 7 días + distribución por estado.</div></div></div>
            <div class="guide-step"><div class="gsn">4</div><div><div class="gs-title">Progreso por producto</div><div class="gs-desc">Panel lateral con el avance de cada producto. Se actualiza al entrar o con ↻.</div></div></div>
          </div>
          <div class="guide-tip"><strong>Tip:</strong> Cargá y movés tareas desde DEV o por el asistente IA; Campaña refleja el pulso de la desarrolladora.</div>
        </div>

        <!-- ESTRATEGIA card -->
        <div class="guide-card">
          <div class="guide-card-head">
            <div class="guide-card-icon">🧠</div>
            <div><div class="guide-card-title" style="color:var(--green)">ESTRATEGIA</div><div class="guide-card-sub">Decisiones · Insights · Riesgos</div></div>
          </div>
          <div class="guide-steps">
            <div class="guide-step"><div class="gsn hl">1</div><div><div class="gs-title">Registrar decisiones</div><div class="gs-desc">Cada decisión no trivial → registrar. ¿Por qué? ¿Qué descartaste?</div></div></div>
            <div class="guide-step"><div class="gsn">2</div><div><div class="gs-title">Capturar insights</div><div class="gs-desc">Algo sorprendente, un cliente revelador, un canal inesperado → Insight.</div></div></div>
            <div class="guide-step"><div class="gsn">3</div><div><div class="gs-title">Mapear riesgos</div><div class="gs-desc">Un riesgo escrito es gestionable. Dependencias, supuestos frágiles, amenazas.</div></div></div>
            <div class="guide-step"><div class="gsn">4</div><div><div class="gs-title">Conectar módulos</div><div class="gs-desc">Campo Conexiones: vinculá el log a productos, tasks o canales.</div></div></div>
          </div>
          <div class="guide-tip"><strong>Tip:</strong> Revisá semanalmente. La diferencia entre hoy y hace un mes está ahí escrita.</div>
        </div>

        <!-- Shortcuts -->
        <div class="guide-shortcuts">
          <div class="guide-card-head">
            <div class="guide-card-icon">⌨️</div>
            <div><div class="guide-card-title">Atajos de teclado</div></div>
          </div>
          <div class="guide-shortcut-grid">
            <div class="gsr"><span class="gsr-label">DEV</span><span class="guide-kbd">1</span></div>
            <div class="gsr"><span class="gsr-label">Campaña</span><span class="guide-kbd">2</span></div>
            <div class="gsr"><span class="gsr-label">Estrategia</span><span class="guide-kbd">3</span></div>
            <div class="gsr"><span class="gsr-label">Guía</span><span class="guide-kbd">4</span></div>
            <div class="gsr"><span class="gsr-label">Cerrar modal</span><span class="guide-kbd">Esc</span></div>
            <div class="gsr"><span class="gsr-label">Asistente IA</span><span class="guide-kbd">🤖</span></div>
          </div>
        </div>

        <!-- Rutina diaria -->
        <div class="guide-rutina">
          <div class="guide-card-head" style="margin-bottom:10px">
            <div class="guide-card-icon">📅</div>
            <div><div class="guide-card-title">Rutina diaria · 15 min</div></div>
          </div>
          <div class="gr-row">
            <div class="gr-icon" style="background:var(--purple-dim);border:1px solid rgba(124,63,255,.2)">☀️</div>
            <div><div class="gs-title" style="color:var(--purple-light)">Mañana — Planificar con IA (5 min)</div><div class="gs-desc">Pedile al asistente que cargue el backlog del día por producto y módulo.</div></div>
          </div>
          <div class="gr-row">
            <div class="gr-icon" style="background:var(--orange-dim);border:1px solid rgba(255,107,53,.2)">⚡</div>
            <div><div class="gs-title" style="color:var(--orange)">Mediodía — DEV (5 min)</div><div class="gs-desc">Marcar completadas, mover todo→doing. ¿Hay algo bloqueado?</div></div>
          </div>
          <div class="gr-row">
            <div class="gr-icon" style="background:var(--green-dim);border:1px solid rgba(57,255,20,.2)">🌙</div>
            <div><div class="gs-title" style="color:var(--green)">Noche — Estrategia (5 min)</div><div class="gs-desc">¿Aprendiste algo? ¿Tomaste una decisión? Registrala antes de cerrar.</div></div>
          </div>
        </div>

      </div><!-- /guide-grid -->
    </div><!-- /guide-body -->
  </div><!-- /panel-guide -->

</div><!-- /main -->
</div><!-- /body-area -->
</div><!-- /shell -->

<!-- MOBILE BOTTOM NAV (<=768px) -->
<nav class="mobile-bottom-nav" id="mobile-bottom-nav">
  <button class="mb-nav-tab active" data-panel="dev" onclick="switchPanel('dev',this)">
    <span class="mb-icon">⚙️</span>
    <span class="mb-label">DEV</span>
    <span class="mb-badge" id="mb-badge-dev">0</span>
  </button>
  <button class="mb-nav-tab" data-panel="campaign" onclick="switchPanel('campaign',this)">
    <span class="mb-icon">🚀</span>
    <span class="mb-label">CAMPAÑA</span>
  </button>
  <button class="mb-nav-tab" data-panel="strategy" onclick="switchPanel('strategy',this)">
    <span class="mb-icon">🧠</span>
    <span class="mb-label">ESTRATEGIA</span>
    <span class="mb-badge" id="mb-badge-str">0</span>
  </button>
  <button class="mb-nav-tab" data-panel="guide" onclick="switchPanel('guide',this)">
    <span class="mb-icon">📖</span>
    <span class="mb-label">GUÍA</span>
  </button>
  <button class="mb-nav-tab mb-chat-btn" onclick="toggleChat()">
    <span class="mb-icon">🤖</span>
    <span class="mb-label">IA</span>
  </button>
</nav>

<div class="toast-wrap" id="toast-wrap"></div>

<!-- MODAL TASK -->
<div class="overlay" id="modal-task" onclick="overlayClose(event,'modal-task')">
  <div class="modal">
    <div class="modal-head"><div class="modal-title" id="task-modal-title">Nueva Task</div><button class="modal-close" onclick="closeModal('modal-task')">✕</button></div>
    <div class="modal-body">
      <input type="hidden" id="task-id"/>
      <div class="f-row">
        <div class="f-group"><label class="f-label">Producto *</label><select class="f-select" id="t-product" onchange="fillModuleDatalist(this.value)"></select></div>
        <div class="f-group"><label class="f-label">Módulo *</label><input class="f-input" id="t-module" list="t-module-list" autocomplete="off" placeholder="Elegí o escribí uno nuevo..."/><datalist id="t-module-list"></datalist></div>
      </div>
      <div class="f-group"><label class="f-label">Nombre *</label><input class="f-input" id="t-name" placeholder="Descripción concisa..."/></div>
      <div class="f-group"><label class="f-label">Descripción</label><textarea class="f-textarea" id="t-desc" style="min-height:55px" placeholder="Detalles, criterios de aceptación..."></textarea></div>
      <div class="f-row">
        <div class="f-group"><label class="f-label">Estado</label><select class="f-select" id="t-status"><option value="todo">⬜ Todo</option><option value="doing">🔶 Doing</option><option value="done">✅ Done</option></select></div>
        <div class="f-group"><label class="f-label">Prioridad</label><select class="f-select" id="t-priority"><option value="alto">🔴 Alto</option><option value="medio">🟡 Medio</option><option value="bajo">🟢 Bajo</option></select></div>
      </div>
      <div class="f-row">
        <div class="f-group"><label class="f-label">Impacto</label><select class="f-select" id="t-impact"><option value="alto">💥 Alto</option><option value="medio">⚡ Medio</option><option value="bajo">💤 Bajo</option></select></div>
        <div class="f-group"></div>
      </div>
    </div>
    <div class="modal-foot">
      <button class="btn btn-danger btn-sm" id="task-del-btn" onclick="deleteTask()" style="display:none">🗑 Eliminar</button>
      <div style="display:flex;gap:7px"><button class="btn" onclick="closeModal('modal-task')">Cancelar</button><button class="btn btn-primary" onclick="saveTask()">Guardar</button></div>
    </div>
  </div>
</div>

<!-- MODAL PRODUCTO -->
<div class="overlay" id="modal-product" onclick="overlayClose(event,'modal-product')">
  <div class="modal">
    <div class="modal-head"><div class="modal-title" id="product-modal-title">Nuevo Producto</div><button class="modal-close" onclick="closeModal('modal-product')">✕</button></div>
    <div class="modal-body">
      <input type="hidden" id="product-id"/>
      <div class="f-row">
        <div class="f-group"><label class="f-label">Nombre *</label><input class="f-input" id="p-name" placeholder="Nombre del producto"/></div>
        <div class="f-group"><label class="f-label">Ícono</label><input class="f-input" id="p-icon" placeholder="🚀" maxlength="4"/></div>
      </div>
      <div class="f-group"><label class="f-label">Descripción</label><input class="f-input" id="p-desc" placeholder="Qué hace, para quién, propuesta de valor..."/></div>
      <div class="f-row">
        <div class="f-group"><label class="f-label">Estado</label>
          <select class="f-select" id="p-status">
            <option value="activo">🟢 Activo</option>
            <option value="idea">💡 Idea</option>
            <option value="pausado">⏸ Pausado</option>
            <option value="archivado">📦 Archivado</option>
          </select>
        </div>
        <div class="f-group"><label class="f-label">Color de acento</label>
          <select class="f-select" id="p-color">
            <option value="#00e5ff">⬡ Cyan</option>
            <option value="#39ff14">⬡ Verde</option>
            <option value="#ff9f43">⬡ Naranja</option>
            <option value="#a87fff">⬡ Púrpura</option>
            <option value="#ff6b35">⬡ Rojo-naranja</option>
            <option value="#ffd700">⬡ Amarillo</option>
            <option value="#ff3e5e">⬡ Rojo</option>
          </select>
        </div>
      </div>
    </div>
    <div class="modal-foot">
      <button class="btn btn-danger btn-sm" id="product-del-btn" onclick="deleteProduct()" style="display:none">🗑 Eliminar</button>
      <div style="display:flex;gap:7px"><button class="btn" onclick="closeModal('modal-product')">Cancelar</button><button class="btn btn-primary" onclick="saveProduct()">Guardar</button></div>
    </div>
  </div>
</div>

<!-- MODAL LOG -->
<div class="overlay" id="modal-log" onclick="overlayClose(event,'modal-log')">
  <div class="modal">
    <div class="modal-head"><div class="modal-title" id="log-modal-title">Nuevo Log Estratégico</div><button class="modal-close" onclick="closeModal('modal-log')">✕</button></div>
    <div class="modal-body">
      <input type="hidden" id="l-id"/>
      <div class="f-row">
        <div class="f-group"><label class="f-label">Tipo</label><select class="f-select" id="l-type"><option value="Decision">💡 Decisión</option><option value="Insight">🔍 Insight</option><option value="Oportunidad">🚀 Oportunidad</option><option value="Riesgo">⚠️ Riesgo</option><option value="Aprendizaje">🧠 Aprendizaje</option><option value="Objetivo">🎯 Objetivo</option><option value="Hipótesis">🔬 Hipótesis</option><option value="Hito">🏆 Hito</option></select></div>
        <div class="f-group"><label class="f-label">Fecha</label><input class="f-input" id="l-date" type="date"/></div>
      </div>
      <div class="f-group"><label class="f-label">Título</label><input class="f-input" id="l-title" placeholder="Título breve"/></div>
      <div class="f-group"><label class="f-label">Contenido *</label><textarea class="f-textarea" id="l-text" rows="4" placeholder="Describí con suficiente contexto para entenderlo en el futuro..."></textarea></div>
      <div class="f-group">
        <label class="f-label">Conexiones <span class="f-hint">(separadas por coma)</span></label>
        <input class="f-input" id="l-links" placeholder="DEV→Backend, Campaña→Landing, Producto→X"/>
      </div>
    </div>
    <div class="modal-foot">
      <span></span>
      <div style="display:flex;gap:7px"><button class="btn" onclick="closeModal('modal-log')">Cancelar</button><button class="btn btn-primary" id="log-save-btn" onclick="saveLog()">Registrar</button></div>
    </div>
  </div>
</div>

<!-- MODAL CONFIRM -->
<div class="overlay" id="modal-confirm" onclick="overlayClose(event,'modal-confirm')">
  <div class="modal modal-confirm">
    <div class="modal-head"><div class="modal-title" id="confirm-title">Confirmar</div><button class="modal-close" onclick="resolveConfirm(false)">✕</button></div>
    <div class="modal-body"><div class="confirm-msg" id="confirm-msg"></div></div>
    <div class="modal-foot">
      <span></span>
      <div style="display:flex;gap:7px"><button class="btn" onclick="resolveConfirm(false)">Cancelar</button><button class="btn btn-danger" id="confirm-ok-btn" onclick="resolveConfirm(true)">Eliminar</button></div>
    </div>
  </div>
</div>

<script>
const API='';
let STATE={tasks:[],devMetrics:{},logs:[],stats:{},products:[]};
let logFilter='';
let modExp={Auth:true,Backend:true,UI:true,'Multi-IA':true};
let charts={weekly:null,status:null};

const LOG_C={Decision:'#00e5ff',Insight:'#ffd700',Riesgo:'#ff3e5e',Oportunidad:'#39ff14',Aprendizaje:'#b388ff',Objetivo:'#ff9100','Hipótesis':'#18ffff',Hito:'#ffab00'};
const LOG_I={Decision:'💡',Insight:'🔍',Riesgo:'⚠️',Oportunidad:'🚀',Aprendizaje:'🧠',Objetivo:'🎯','Hipótesis':'🔬',Hito:'🏆'};
const MODS=['Auth','Backend','UI','Multi-IA'];

// clock
function tick(){const n=new Date();document.getElementById('clock').textContent=n.toLocaleDateString('es-AR',{weekday:'short'}).toUpperCase()+' '+n.toTimeString().slice(0,8);}
setInterval(tick,1000);tick();

// api
async function api(path,method='GET',body=null){
  const opts={method,headers:{'Content-Type':'application/json'}};
  if(body) opts.body=JSON.stringify(body);
  let r;
  try{r=await fetch(API+path,opts);}
  catch(e){throw new Error('Error de red: '+e.message);}
  if(!r.ok){let m='Error '+r.status;try{m=await r.text();}catch(e){}throw new Error(m);}
  return r.json();
}

async function boot(){
  try{
    await Promise.all([loadProducts(),loadTasks(),loadDevMetrics(),loadLogs(),loadStats()]);
    const old=document.getElementById('boot-err');if(old)old.remove();
    render();
  }catch(e){
    console.error('Boot error:',e);
    const existing=document.getElementById('boot-err');if(existing)existing.remove();
    document.body.insertAdjacentHTML('afterbegin',
      `<div id="boot-err" style="position:fixed;top:0;left:0;right:0;z-index:9999;background:#ff3e5e;color:#fff;
        font-size:11px;padding:9px 20px;text-align:center;font-family:monospace;">
        &#9888; Error al cargar datos: ${e.message}
        <button onclick="document.getElementById('boot-err').remove();boot()" style="margin-left:12px;padding:2px 10px;
          background:rgba(0,0,0,.3);border:1px solid rgba(255,255,255,.4);color:#fff;border-radius:3px;cursor:pointer;font-family:monospace">
          Reintentar
        </button></div>`);
    render();
  }
}

async function loadProducts(){STATE.products=await api('/api/products');}
async function loadTasks(){STATE.tasks=await api('/api/tasks');}
async function loadDevMetrics(){STATE.devMetrics=await api('/api/dev-metrics');}
async function loadLogs(){STATE.logs=await api('/api/logs');}
async function loadStats(){STATE.stats=await api('/api/stats');}
function render(){renderDev();renderCampaign();renderStrategy();renderStats();}

// panel switch
function switchPanel(name,btnEl){
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  document.getElementById('panel-'+name).classList.add('active');
  // Sync active state across all nav surfaces by panel name (no matter which control fired it)
  document.querySelectorAll('.nav-tab').forEach(b=>b.classList.remove('active'));
  document.querySelector('.nav-tab[data-panel="'+name+'"]')?.classList.add('active');
  document.querySelectorAll('.mb-nav-tab').forEach(b=>b.classList.remove('active'));
  document.querySelector('.mb-nav-tab[data-panel="'+name+'"]')?.classList.add('active');
  document.querySelectorAll('.sidebar-link').forEach(b=>b.classList.remove('active'));
  document.getElementById('sl-'+name)?.classList.add('active');
  if(name==='campaign') loadDevMetrics().then(()=>renderCampaign()).catch(()=>renderCharts());
  closeSidebar();
}

/* [P7] drawer JS: hamburger+scrim+switchPanel close */
function setSidebarOpen(open){
  const sb=document.querySelector('.sidebar');
  if(!sb)return;
  sb.classList.toggle('open',open);
  document.getElementById('sidebar-scrim')?.classList.toggle('open',open);
  document.getElementById('hamburger')?.setAttribute('aria-expanded',open?'true':'false');
  try{localStorage.setItem('aethySidebarOpen',open?'1':'0');}catch(e){}
}
function toggleSidebar(){
  setSidebarOpen(!document.querySelector('.sidebar')?.classList.contains('open'));
}
function closeSidebar(){
  if(!document.querySelector('.sidebar')?.classList.contains('open'))return;
  setSidebarOpen(false);
}
// Restore drawer state (mobile only; desktop sidebar is always visible regardless of .open)
function restoreSidebarState(){
  try{
    if(localStorage.getItem('aethySidebarOpen')==='1' && window.matchMedia('(max-width:768px)').matches){
      setSidebarOpen(true);
    }
  }catch(e){}
}

// stats
function renderStats(){
  const s=STATE.stats;
  const pct=s.tasks_total?Math.round(s.tasks_done/s.tasks_total*100):0;
  const setText=(id,v)=>{const el=document.getElementById(id);if(el)el.textContent=v??'—';};
  setText('ts-todo',s.tasks_todo);setText('ts-doing',s.tasks_doing);setText('ts-logs',s.logs_total);
  setText('gs-t',s.tasks_todo);setText('gs-prod',(s.products_active||s.products_total||0)+' activos');setText('gs-p',pct+'%');setText('gs-l',s.logs_total);
  setText('g-st',(s.tasks_doing||0)+(s.tasks_todo||0));setText('g-prod',s.products_active||s.products_total||0);setText('g-sp',pct+'%');setText('g-sl',s.logs_total);
  setText('nb-dev',s.products_total||0);setText('slb-dev',s.products_total||0);
  setText('mb-badge-dev',s.products_total||0);
  const devTaskEl=document.getElementById('nb-dev-tasks');
  if(devTaskEl) devTaskEl.textContent=(s.tasks_doing||0)+' doing';
  setText('nb-str',s.logs_total||0);setText('slb-str',s.logs_total||0);
  setText('mb-badge-str',s.logs_total||0);
}

// ══ DEV ══
let prodExp={};  // {product_id: {open:bool, modules:{modName:bool}}}

function expandAllProducts(){
  STATE.products.forEach(p=>{
    prodExp[p.id]={open:true,modules:{}};
    const mods=[...new Set(STATE.tasks.filter(t=>t.product_id===p.id).map(t=>t.module))];
    mods.forEach(m=>{prodExp[p.id].modules[m]=true;});
  });
  renderDev();
}
function collapseAllProducts(){
  STATE.products.forEach(p=>{prodExp[p.id]={open:false,modules:{}};});
  renderDev();
}

function renderDev(){
  const body=document.getElementById('dev-body');
  if(!STATE.products.length){body.innerHTML='<div class="empty"><div class="empty-icon">📦</div>Sin productos. Creá uno con + Producto.</div>';return;}
  body.innerHTML='';
  STATE.products.forEach(prod=>{
    if(!prodExp[prod.id]) prodExp[prod.id]={open:true,modules:{}};
    const state=prodExp[prod.id];
    const ptasks=STATE.tasks.filter(t=>t.product_id===prod.id);
    const pdone=ptasks.filter(t=>t.done).length;
    const ppct=ptasks.length?Math.round(pdone/ptasks.length*100):0;
    const clr=prod.color||'#00e5ff';

    // group by module
    const modules=[...new Set(ptasks.map(t=>t.module))];
    
    const el=document.createElement('div');
    el.className='dev-product';
    el.style.borderLeftColor=clr;
    el.style.borderLeftWidth='3px';

    let modHtml='';
    if(state.open){
      modules.forEach((mod,mi)=>{
        if(state.modules[mod]===undefined) state.modules[mod]=true;
        const mtasks=ptasks.filter(t=>t.module===mod);
        const mdone=mtasks.filter(t=>t.done).length;
        const mpct=mtasks.length?Math.round(mdone/mtasks.length*100):0;
        const mopen=state.modules[mod]!==false;
        const isLast=mi===modules.length-1;
        modHtml+=`<div class="dev-module">
          <div class="dev-mod-head ${!mopen&&isLast?'last-closed':''}" onclick="toggleMod('${prod.id}','${mod}')">
            <span class="chevron ${mopen?'open':''}">▶</span>
            <div class="dev-mod-title">${mod}<span class="dev-mod-meta">${mdone}/${mtasks.length}</span></div>
            <div style="display:flex;align-items:center;gap:7px;margin-left:auto">
              <div class="prog-track"><div class="prog-fill" style="width:${mpct}%;background:${clr}"></div></div>
              <span class="prog-pct" style="color:${clr}">${mpct}%</span>
              <button class="btn btn-sm btn-icon" onclick="event.stopPropagation();openTaskModal(null,'${prod.id}','${mod}')">+</button>
            </div>
          </div>
          ${mopen?`<div class="dev-task-list">${mtasks.map(t=>dtHTML(t,clr)).join('')}</div>`:''}
        </div>`;
      });
    }

    el.innerHTML=`<div class="dev-prod-head" onclick="toggleProd('${prod.id}')">
        <span class="chevron ${state.open?'open':''}">▶</span>
        <span class="dev-prod-icon">${prod.icon}</span>
        <div class="dev-prod-name" style="color:${clr}">${prod.name}</div>
        <span class="dev-prod-status ${prod.status}">${prod.status}</span>
        <div class="dev-prod-prog">
          <div class="prog-track"><div class="prog-fill" style="width:${ppct}%;background:${clr}88"></div></div>
          <span class="prog-pct" style="color:${clr}">${ppct}%</span>
          <span style="font-size:10px;color:var(--text2)">${pdone}/${ptasks.length}</span>
        </div>
        <div class="dev-prod-actions">
          <button class="ca-btn" onclick="event.stopPropagation();openProductModal('${prod.id}')" title="Editar producto">✎</button>
          <button class="ca-btn" onclick="event.stopPropagation();openTaskModal(null,'${prod.id}',null)" title="Nueva task">+</button>
        </div>
      </div>
      ${state.open?`<div class="dev-prod-body">${modHtml}${modules.length===0?'<div style="padding:14px 52px;font-size:11px;color:var(--text2)">Sin tasks. Agregá una con + Task.</div>':''}</div>`:''}`;
    body.appendChild(el);
  });
}

function dtHTML(t,clr='#00e5ff'){
  const cc=t.done?'checked':(t.status==='doing'?'doing':'');
  const ct=t.done?'✓':(t.status==='doing'?'●':'');
  return `<div class="dev-task">
    <div class="dtc ${cc}" onclick="toggleTask('${t.id}')">${ct}</div>
    <div class="dt-info">
      <div class="dt-name ${t.done?'done':''}">${t.name}</div>
      ${t.description?`<div class="dt-desc">${t.description}</div>`:''}
      <div class="dt-tags"><span class="sp sp-${t.status}">${t.status}</span><span class="tag tag-${t.priority}">${t.priority}</span><span style="font-size:9px;color:var(--text2)">↑ ${t.impact}</span></div>
    </div>
    <div class="dt-actions"><button class="ca-btn" onclick="openTaskModal('${t.id}')" title="Editar">✎</button></div>
  </div>`;
}

function toggleProd(pid){
  if(!prodExp[pid]) prodExp[pid]={open:false,modules:{}};
  prodExp[pid].open=!prodExp[pid].open;
  renderDev();
}
function toggleMod(pid,mod){
  if(!prodExp[pid]) prodExp[pid]={open:true,modules:{}};
  prodExp[pid].modules[mod]=prodExp[pid].modules[mod]===false?true:false;
  renderDev();
}

async function toggleTask(id){
  try{await api('/api/tasks/'+id+'/toggle','PATCH');await loadTasks();await loadStats();renderDev();renderStats();}
  catch(e){toast(e.message,'error');}
}

function fillProductSelect(selectedId=null){
  const sel=document.getElementById('t-product');
  sel.innerHTML=STATE.products.map(p=>`<option value="${p.id}" ${p.id==selectedId?'selected':''}>${p.icon} ${p.name}</option>`).join('');
}
// Populate the module datalist with existing modules of the given product (suggestions only; new ones still allowed)
function fillModuleDatalist(prodId){
  const dl=document.getElementById('t-module-list');
  if(!dl)return;
  const mods=[...new Set(STATE.tasks.filter(t=>t.product_id===prodId).map(t=>t.module).filter(Boolean))].sort();
  dl.innerHTML=mods.map(m=>`<option value="${m.replace(/"/g,'&quot;')}"></option>`).join('');
}

function openTaskModal(id=null,defaultProdId=null,defaultMod=null){
  document.getElementById('task-id').value='';
  ['t-name','t-desc'].forEach(i=>{const el=document.getElementById(i);if(el)el.value='';});
  const initProd=defaultProdId||STATE.products[0]?.id;
  fillProductSelect(initProd);
  fillModuleDatalist(initProd);
  document.getElementById('t-module').value=defaultMod||'Backend';
  document.getElementById('t-status').value='todo';
  document.getElementById('t-priority').value='medio';
  document.getElementById('t-impact').value='medio';
  const del=document.getElementById('task-del-btn');
  if(id){
    const t=STATE.tasks.find(x=>x.id===id);if(!t)return;
    document.getElementById('task-modal-title').textContent='Editar Task';
    document.getElementById('task-id').value=id;
    fillProductSelect(t.product_id);
    fillModuleDatalist(t.product_id);
    document.getElementById('t-module').value=t.module||'Backend';
    document.getElementById('t-name').value=t.name||'';
    document.getElementById('t-desc').value=t.description||'';
    document.getElementById('t-status').value=t.status||'todo';
    document.getElementById('t-priority').value=t.priority||'medio';
    document.getElementById('t-impact').value=t.impact||'medio';
    del.style.display='block';
  }else{document.getElementById('task-modal-title').textContent='Nueva Task';del.style.display='none';}
  openModal('modal-task');
}
async function saveTask(){
  const name=document.getElementById('t-name').value.trim();if(!name){shake('t-name');return;}
  const id=document.getElementById('task-id').value;
  const pid=document.getElementById('t-product').value;
  const data={product_id:pid,module:v('t-module')||'Backend',name,description:v('t-desc'),
    status:v('t-status'),priority:v('t-priority'),impact:v('t-impact'),done:v('t-status')==='done'?1:0};
  try{
    if(id){await api('/api/tasks/'+id,'PUT',data);toast('Task actualizada','success');}
    else{await api('/api/tasks','POST',data);toast('Task creada','success');}
    closeModal('modal-task');await loadTasks();await loadStats();renderDev();renderStats();
  }catch(e){toast(e.message,'error');}
}
async function deleteTask(){
  const id=document.getElementById('task-id').value;if(!id)return;
  if(!await confirmDialog({title:'Eliminar task',message:'¿Eliminar esta task? Esta acción no se puede deshacer.',okText:'Eliminar'}))return;
  try{await api('/api/tasks/'+id,'DELETE');closeModal('modal-task');await loadTasks();await loadStats();renderDev();renderStats();toast('Task eliminada','error');}
  catch(e){toast(e.message,'error');}
}

// ── PRODUCTS CRUD ──
function openProductModal(id=null){
  document.getElementById('product-id').value='';
  ['p-name','p-desc'].forEach(i=>document.getElementById(i).value='');
  document.getElementById('p-icon').value='📦';
  document.getElementById('p-status').value='activo';
  document.getElementById('p-color').value='#00e5ff';
  const del=document.getElementById('product-del-btn');
  if(id){
    const p=STATE.products.find(x=>x.id===id);if(!p)return;
    document.getElementById('product-modal-title').textContent='Editar Producto';
    document.getElementById('product-id').value=id;
    document.getElementById('p-name').value=p.name||'';
    document.getElementById('p-icon').value=p.icon||'📦';
    document.getElementById('p-desc').value=p.description||'';
    document.getElementById('p-status').value=p.status||'activo';
    document.getElementById('p-color').value=p.color||'#00e5ff';
    del.style.display='block';
  }else{document.getElementById('product-modal-title').textContent='Nuevo Producto';del.style.display='none';}
  openModal('modal-product');
}
async function saveProduct(){
  const name=document.getElementById('p-name').value.trim();if(!name){shake('p-name');return;}
  const id=document.getElementById('product-id').value;
  const data={name,icon:v('p-icon')||'📦',description:v('p-desc'),status:v('p-status'),color:v('p-color')};
  try{
    if(id){await api('/api/products/'+id,'PUT',data);toast('Producto actualizado','success');}
    else{await api('/api/products','POST',data);toast('Producto creado','success');}
    closeModal('modal-product');await loadProducts();await loadStats();renderDev();renderStats();
  }catch(e){toast(e.message,'error');}
}
async function deleteProduct(){
  const id=document.getElementById('product-id').value;if(!id)return;
  if(!await confirmDialog({title:'Eliminar producto',message:'¿Eliminar el producto y todas sus tasks? Esta acción no se puede deshacer.',okText:'Eliminar'}))return;
  try{await api('/api/products/'+id,'DELETE');closeModal('modal-product');await loadProducts();await loadTasks();await loadStats();renderDev();renderStats();toast('Producto eliminado','error');}
  catch(e){toast(e.message,'error');}
}

// ══ CAMPAÑA — Dashboard de la Desarrolladora ══
function renderCampaign(){
  const m=STATE.devMetrics||{};
  const weekly=m.weekly_completed||[];
  const perProd=m.per_product||[];

  // métricas principales (todo derivado de DEV)
  document.getElementById('camp-metrics-bar').innerHTML=`
    <div class="mc"><div class="mc-label">Proyectos</div><div class="mc-val" style="color:var(--cyan)">${m.products_total||0}</div><div class="mc-sub">${m.products_active||0} activos</div></div>
    <div class="mc"><div class="mc-label">Tareas</div><div class="mc-val" style="color:var(--text0)">${m.tasks_total||0}</div><div class="mc-sub">en backlog</div></div>
    <div class="mc"><div class="mc-label">Completado</div><div class="mc-val" style="color:var(--green)">${m.pct_done||0}%</div><div class="mc-sub">${m.tasks_done||0}/${m.tasks_total||0}</div></div>
    <div class="mc"><div class="mc-label">Esta semana</div><div class="mc-val" style="color:var(--purple-light)">${m.week_total||0}</div><div class="mc-sub">tareas hechas</div></div>`;

  // cards por estado
  const wrap=document.getElementById('camp-cards-wrap');
  wrap.innerHTML=`
    <div class="status-grid">
      <div class="status-card todo"><div class="status-val">${m.tasks_todo||0}</div><div class="status-lbl">⬜ Por hacer</div></div>
      <div class="status-card doing"><div class="status-val">${m.tasks_doing||0}</div><div class="status-lbl">🔶 Haciendo</div></div>
      <div class="status-card done"><div class="status-val">${m.tasks_done||0}</div><div class="status-lbl">✅ Hecho</div></div>
    </div>`;

  // panel derecho - progreso por producto
  const rb=document.getElementById('camp-right-body');
  if(!perProd.length){
    rb.innerHTML='<div class="empty" style="padding:24px"><div class="empty-icon">📦</div>Sin productos todavía.</div>';
  }else{
    rb.innerHTML='';
    perProd.forEach(p=>{
      const div=document.createElement('div');div.className='canal-summary';
      div.innerHTML=`<div class="cs-head"><span class="cs-icon">${p.icon}</span><span class="cs-name">${p.name}</span><span class="cs-conv" style="color:${p.color}">${p.pct}%</span></div>
        <div class="cs-bar-wrap"><div class="cs-bar-fill" style="width:${p.pct}%;background:${p.color}"></div></div>
        <div class="cs-stats"><span class="cs-stat">✅ <span>${p.done}</span></span><span class="cs-stat">📋 <span>${p.total}</span></span><span class="cs-stat">${p.status}</span></div>`;
      rb.appendChild(div);
    });
  }

  renderCharts();
}

function renderCharts(){
  if(typeof Chart==='undefined')return;
  const m=STATE.devMetrics||{};
  const weekly=m.weekly_completed||[];

  const chartCfg={
    responsive:true,maintainAspectRatio:false,
    plugins:{legend:{display:false}},
    scales:{
      x:{ticks:{color:'#4e6880',font:{size:9}},grid:{color:'rgba(30,45,61,.5)'}},
      y:{beginAtZero:true,ticks:{color:'#4e6880',font:{size:9},precision:0},grid:{color:'rgba(30,45,61,.5)'}}
    }
  };

  // Gráfico semanal (tareas completadas por día)
  const c1=document.getElementById('chart-weekly');
  if(c1){
    if(charts.weekly){charts.weekly.destroy();}
    charts.weekly=new Chart(c1,{type:'bar',data:{
      labels:weekly.map(d=>d.label),
      datasets:[{data:weekly.map(d=>d.count),backgroundColor:'rgba(124,63,255,.55)',borderColor:'#7c3fff',borderWidth:1,borderRadius:3}]
    },options:chartCfg});
  }

  // Gráfico distribución por estado (doughnut)
  const c2=document.getElementById('chart-status');
  if(c2){
    if(charts.status){charts.status.destroy();}
    charts.status=new Chart(c2,{type:'doughnut',data:{
      labels:['Por hacer','Haciendo','Hecho'],
      datasets:[{data:[m.tasks_todo||0,m.tasks_doing||0,m.tasks_done||0],
        backgroundColor:['rgba(78,104,128,.6)','rgba(255,107,53,.7)','rgba(57,255,20,.55)'],
        borderColor:['#4e6880','#ff6b35','#39ff14'],borderWidth:1}]
    },options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'bottom',labels:{color:'#94a8bc',font:{size:10},boxWidth:12,padding:8}}}}});
  }
}

// ══ ESTRATEGIA ══
function setLogFilter(btn,type){
  logFilter=type;
  document.querySelectorAll('#log-filter .btn').forEach(b=>b.classList.remove('active-filter'));
  btn.classList.add('active-filter');renderStrategy();
}
function renderStrategy(){
  const body=document.getElementById('strategy-body');
  const logs=logFilter?STATE.logs.filter(l=>l.type===logFilter):STATE.logs;
  if(!logs.length){body.innerHTML='<div class="empty"><div class="empty-icon">🧠</div>Sin logs.</div>';return;}
  body.innerHTML=logs.map(l=>{
    const clr=LOG_C[l.type]||'#4e6880';
    const links=Array.isArray(l.links)?l.links:[];
    return `<div class="log-entry" style="--lc:${clr}">
      <div class="log-actions">
        <button class="btn btn-sm btn-icon log-edit" onclick="event.stopPropagation();editLog('${l.id}')" title="Editar">✎</button>
        <button class="btn btn-sm btn-danger btn-icon log-del" onclick="event.stopPropagation();deleteLog('${l.id}')" title="Eliminar">✕</button>
      </div>
      <div class="log-header"><span class="log-badge">${LOG_I[l.type]?LOG_I[l.type]+' ':''}${l.type}</span>${l.title?`<span class="log-title">${l.title}</span>`:''}<span class="log-date">${l.date||''}</span></div>
      <div class="log-text" onclick="toggleLogText(this)">${l.text}</div>
      ${links.length?`<div class="log-links">${links.map(lk=>`<span class="log-link-tag">🔗 ${lk}</span>`).join('')}</div>`:''}</div>`;
  }).join('');
}
function toggleLogText(el){el.classList.toggle('expanded');}
function openLogModal(id=null){
  ['l-title','l-text','l-links'].forEach(i=>document.getElementById(i).value='');
  document.getElementById('l-id').value='';
  document.getElementById('l-type').value='Decision';
  document.getElementById('l-date').value=new Date().toISOString().slice(0,10);
  document.getElementById('log-modal-title').textContent='Nuevo Log Estratégico';
  document.getElementById('log-save-btn').textContent='Registrar';
  if(id){
    const l=STATE.logs.find(x=>x.id===id);if(!l)return;
    document.getElementById('l-id').value=id;
    document.getElementById('l-type').value=l.type||'Decision';
    document.getElementById('l-date').value=l.date||new Date().toISOString().slice(0,10);
    document.getElementById('l-title').value=l.title||'';
    document.getElementById('l-text').value=l.text||'';
    document.getElementById('l-links').value=(Array.isArray(l.links)?l.links:[]).join(', ');
    document.getElementById('log-modal-title').textContent='Editar Log';
    document.getElementById('log-save-btn').textContent='Guardar';
  }
  openModal('modal-log');
}
function editLog(id){openLogModal(id);}
async function saveLog(){
  const text=document.getElementById('l-text').value.trim();if(!text){shake('l-text');return;}
  const links=v('l-links').split(',').map(s=>s.trim()).filter(Boolean);
  const id=v('l-id');
  const data={type:v('l-type'),title:v('l-title'),text,links,date:v('l-date')};
  try{
    if(id){await api(`/api/logs/${id}`,'PUT',data);toast('Log actualizado','success');}
    else{await api('/api/logs','POST',data);toast('Log registrado','success');}
    closeModal('modal-log');await loadLogs();await loadStats();renderStrategy();renderStats();
  }catch(e){toast(e.message,'error');}
}
async function deleteLog(id){
  if(!await confirmDialog({title:'Eliminar log',message:'¿Eliminar este log estratégico? Esta acción no se puede deshacer.',okText:'Eliminar'}))return;
  try{await api(`/api/logs/${id}`,'DELETE');await loadLogs();await loadStats();renderStrategy();renderStats();toast('Log eliminado','error');}
  catch(e){toast(e.message,'error');}
}

// modals
function openModal(id){
  document.getElementById(id).classList.add('open');
  document.getElementById('chat-fab')?.classList.add('modal-open');
}
function closeModal(id){
  document.getElementById(id).classList.remove('open');
  document.getElementById('chat-fab')?.classList.remove('modal-open');
}
function overlayClose(e,id){if(e.target.id!==id)return;if(id==='modal-confirm'&&_confirmResolve){resolveConfirm(false);return;}closeModal(id);}
// modal de confirmación reutilizable (reemplaza el confirm() nativo)
let _confirmResolve=null;
function confirmDialog({title='Confirmar',message='',okText='Eliminar'}={}){
  document.getElementById('confirm-title').textContent=title;
  document.getElementById('confirm-msg').textContent=message;
  document.getElementById('confirm-ok-btn').textContent=okText;
  openModal('modal-confirm');
  return new Promise(res=>{_confirmResolve=res;});
}
function resolveConfirm(val){
  closeModal('modal-confirm');
  if(_confirmResolve){_confirmResolve(val);_confirmResolve=null;}
}
function v(id){return document.getElementById(id).value;}
function shake(id){
  const el=document.getElementById(id);if(!el)return;
  el.style.borderColor='var(--red)';
  el.animate([{transform:'translateX(-4px)'},{transform:'translateX(4px)'},{transform:'translateX(0)'}],{duration:180});
  setTimeout(()=>el.style.borderColor='',1500);
}
function toast(msg,type='info'){
  const wrap=document.getElementById('toast-wrap');
  const el=document.createElement('div');el.className=`toast ${type}`;
  const col=type==='success'?'var(--green)':type==='error'?'var(--red)':'var(--cyan)';
  el.innerHTML=`<span style="color:${col}">${type==='success'?'✓':type==='error'?'✕':'●'}</span> ${msg}`;
  wrap.appendChild(el);setTimeout(()=>el.remove(),3100);
}
document.addEventListener('keydown',e=>{
  if(e.key==='Escape'){if(_confirmResolve)resolveConfirm(false);document.querySelectorAll('.overlay.open').forEach(o=>o.classList.remove('open'));closeSidebar();return;}
  if(e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA'||e.target.tagName==='SELECT')return;
  const map={'1':'dev','2':'campaign','3':'strategy','4':'guide'};
  if(map[e.key])switchPanel(map[e.key]);
});
// ══ CHATBOT ══
let chatHistory=[];
let chatOpen=false;

function toggleChat(){
  chatOpen=!chatOpen;
  document.getElementById('chat-panel').classList.toggle('open',chatOpen);
  const fab=document.getElementById('chat-fab');
  fab.classList.toggle('open',chatOpen);
  fab.textContent=chatOpen?'✕':'🤖';
  if(chatOpen&&chatHistory.length===0){
    appendMsg('bot','Soy AETHY. Puedo crear y modificar productos y tareas, moverlas, eliminarlas y darte el avance de la desarrolladora. Probá:\n\n- *"Cargá el producto X con las tareas A, B, C"*\n- *"¿cómo viene el avance?"*');
  }
  document.getElementById('chat-messages').scrollTop=9999;
}

function escapeHtml(s){
  return s.replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
// Render markdown (bot) or plain text (user) to sanitized HTML
function renderContent(text,isBot){
  if(!isBot)return escapeHtml(text);
  try{return DOMPurify.sanitize(marked.parse(text));}
  catch(e){return escapeHtml(text);}
}
function appendMsg(role,text){
  const isBot=role!=='user';
  const rendered=renderContent(text,isBot);
  // chatHistory keeps the ORIGINAL text (markdown/plain) for the LLM context
  chatHistory.push({role:isBot?'assistant':'user',content:text});
  const wrap=document.getElementById('chat-messages');
  const div=document.createElement('div');
  div.className='chat-msg '+(isBot?'bot':'user');
  div.innerHTML=`<div class="chat-bubble">${rendered}</div>`;
  wrap.appendChild(div);
  wrap.scrollTop=wrap.scrollHeight;
  const saved=JSON.parse(localStorage.getItem('aethyChat')||'[]');
  saved.push({role:isBot?'assistant':'user',text:text,rendered:rendered});
  if(saved.length>50)saved.splice(0,saved.length-50);
  localStorage.setItem('aethyChat',JSON.stringify(saved));
}

function showTyping(){
  const wrap=document.getElementById('chat-messages');
  const div=document.createElement('div');
  div.id='chat-typing-indicator';div.className='chat-msg bot';
  div.innerHTML='<div class="chat-bubble chat-typing-wrap"><div class="chat-dot"></div><div class="chat-dot"></div><div class="chat-dot"></div></div>';
  wrap.appendChild(div);wrap.scrollTop=wrap.scrollHeight;
}

function hideTyping(){
  const el=document.getElementById('chat-typing-indicator');if(el)el.remove();
}

async function sendChat(){
  const input=document.getElementById('chat-input');
  const text=input.value.trim();if(!text)return;
  input.value='';input.style.height='';
  appendMsg('user',text);
  showTyping();
  document.getElementById('chat-send').disabled=true;
  try{
    // Send only last 8 turns to avoid token overflow (TPM de Groq cuenta el historial)
    const msgs=chatHistory.slice(-8);
    const res=await fetch('/api/chat',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({messages:msgs})
    });
    if(!res.ok)throw new Error('HTTP '+res.status);
    const data=await res.json();
    hideTyping();
    appendMsg('bot',data.response||'Acción completada.');
    if(data.refresh){
      await Promise.all([loadProducts(),loadTasks(),loadStats(),loadDevMetrics(),loadLogs()]);
      renderDev();renderStats();renderCampaign();renderStrategy();
      toast('Sistema actualizado','success');
    }
  }catch(e){
    hideTyping();appendMsg('bot','⚠ Error: '+e.message);
  }finally{
    document.getElementById('chat-send').disabled=false;
  }
}

async function clearChat(){
  if(!await confirmDialog({title:'Limpiar historial',message:'¿Limpiar todo el historial de AETHY? Esta acción no se puede deshacer.',okText:'Limpiar'}))return;
  chatHistory=[];
  document.getElementById('chat-messages').innerHTML='';
  localStorage.removeItem('aethyChat');
  toast('Historial limpiado','info');
}

document.addEventListener('DOMContentLoaded',()=>{
  restoreSidebarState();
  const saved=JSON.parse(localStorage.getItem('aethyChat')||'[]');
  saved.forEach(entry=>{
    const role=entry.role;
    const isBot=role!=='user';
    // New format: {role,text,rendered}. Old format fallback: {role,html}
    const text=entry.text!==undefined?entry.text:(entry.html||'').replace(/<[^>]+>/g,'');
    const rendered=entry.rendered!==undefined?entry.rendered:renderContent(text,isBot);
    chatHistory.push({role,content:text});
    const wrap=document.getElementById('chat-messages');
    if(!wrap)return;
    const div=document.createElement('div');
    div.className='chat-msg '+(role==='user'?'user':'bot');
    div.innerHTML=`<div class="chat-bubble">${rendered}</div>`;
    wrap.appendChild(div);
  });
  if(saved.length)document.getElementById('chat-messages').scrollTop=9999;
  const ci=document.getElementById('chat-input');
  if(ci){
    ci.addEventListener('keydown',e=>{
      if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendChat();}
    });
    ci.addEventListener('input',function(){
      this.style.height='auto';
      this.style.height=Math.min(this.scrollHeight,80)+'px';
    });
  }
});

boot();
</script>

<!-- ══ CHATBOT WIDGET ══ -->
<button class="chat-fab" id="chat-fab" onclick="toggleChat()" title="AETHY · Asistente IA">🤖</button>
<div class="chat-panel" id="chat-panel">
  <div class="chat-head">
    <div class="chat-head-icon">🤖</div>
    <div class="chat-head-info">
      <div class="chat-head-title">AETHY</div>
      <div class="chat-head-sub">Asistente IA · Qwen3-32B</div>
    </div>
    <div class="chat-online"></div>
    <button class="chat-clear" onclick="clearChat()" title="Limpiar historial">🗑</button>
  </div>
  <div class="chat-messages" id="chat-messages"></div>
  <div class="chat-input-wrap">
    <textarea class="chat-input" id="chat-input"
      placeholder="Cargá productos, tasks… (Enter = enviar, Shift+Enter = nueva línea)"
      rows="1"></textarea>
    <button class="chat-send" id="chat-send" onclick="sendChat()">➤</button>
  </div>
</div>
</body>
</html>
"""

@app.route('/')
def index():
    return HTML, 200, {'Content-Type': 'text/html; charset=utf-8'}

if __name__ == '__main__':
    print("\n  OpenAETH Command Core — MongoDB + Groq AI")
    print(f"  DB: {MONGODB_DB}  |  Groq: {'✓' if groq_client else '✗'}  |  Modelo: {GROQ_MODEL}")
    print("  → http://localhost:5000\n")
    app.run(debug=False, host='0.0.0.0', port=5000)
