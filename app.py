from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from bson import ObjectId
from datetime import datetime, timedelta
import os, json, re, certifi

try:
    from groq import Groq as GroqClient
except ImportError:
    GroqClient = None

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

client = MongoClient(
    MONGODB_URI,
    server_api=ServerApi("1"),
    tls=True,
    tlsCAFile=certifi.where(),
    serverSelectionTimeoutMS=5000,
)
db = client[MONGODB_DB]

groq_client = GroqClient(api_key=GROQ_API_KEY) if (GroqClient and GROQ_API_KEY) else None

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
        "description": "Lista tareas. Si se pasa product_name, filtra por ese producto (búsqueda por nombre).",
        "parameters": {
            "type": "object",
            "properties": {
                "product_name": {"type": "string", "description": "Nombre del producto para filtrar (opcional)"}
            }
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
            "Registra una entrada en Estrategia: una decisión, insight, riesgo u oportunidad "
            "que surja de la conversación. Útil para capturar aprendizajes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "type":  {"type": "string", "enum": ["Decision","Insight","Riesgo","Oportunidad"]},
                "title": {"type": "string", "description": "Título breve."},
                "text":  {"type": "string", "description": "Contenido del log con contexto suficiente."},
                "links": {"type": "array", "items": {"type": "string"},
                          "description": "Conexiones opcionales, ej: 'DEV→Backend'."},
            },
            "required": ["type", "text"]
        }
    }},
]


def _insert_tasks(product_id_str: str, default_module: str, tasks: list) -> list:
    """Insert a list of task dicts under the given product_id. Returns created task summaries."""
    now = datetime.utcnow()
    created = []
    for t in tasks:
        module = t.get("module") or default_module or "General"
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

            tasks_raw = args.get("tasks", [])
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
            query = {}
            if q:
                product = db.products.find_one({"name": {"$regex": q, "$options": "i"}})
                if product:
                    query["product_id"] = str(product["_id"])
            tasks = list(db.tasks.find(query).limit(60))
            return [{
                "id": str(t["_id"]), "name": t["name"],
                "status": t.get("status"), "module": t.get("module"),
                "priority": t.get("priority"),
            } for t in tasks]

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
            "• Consultar: list_products, list_tasks, get_dev_metrics (avance, % completado, "
            "tareas por estado, esta semana, progreso por producto).\n"
            "• Estrategia: create_log (registrar decisión/insight/riesgo/oportunidad).\n\n"
            "REGLAS:\n"
            "1. Para crear producto + tareas usá SIEMPRE create_product_with_tasks (nunca por separado).\n"
            "2. Buscás productos y tareas por nombre; no inventes IDs.\n"
            "3. Para preguntas de avance/estado/reportes usá get_dev_metrics, no adivines números.\n"
            "4. Las acciones destructivas (delete_*) sólo si el usuario lo pide explícitamente; "
            "si hay ambigüedad, preguntá antes.\n"
            "5. Confirmá siempre con un resumen breve de lo que hiciste."
        )
    }

    # Rebuild conversation history — include tool messages from this request only
    # (client sends only user/assistant content turns)
    all_messages = [system_msg] + [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]

    did_tool_calls = False

    for _ in range(8):
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=all_messages,
                tools=GROQ_TOOLS,
                tool_choice="auto",
                max_tokens=1024,
                reasoning_format="hidden",
            )
        except Exception as e:
            return jsonify({'response': f'Error Groq: {str(e)}', 'refresh': did_tool_calls})

        choice = response.choices[0]
        message = choice.message
        finish_reason = choice.finish_reason

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
            for tc in message.tool_calls:
                try:
                    tool_args = json.loads(tc.function.arguments)
                except Exception:
                    tool_args = {}
                result = execute_tool(tc.function.name, tool_args)
                all_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False, default=str)
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

HTML = """


<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>OpenAETH — Command Core</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
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
.dev-prod-name{font-family:var(--sans);font-size:13px;font-weight:800;flex:1;}
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
.status-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;}
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
.log-header{display:flex;align-items:center;gap:8px;margin-bottom:5px;flex-wrap:wrap;}
.log-badge{padding:2px 8px;border-radius:2px;font-size:9px;font-weight:700;letter-spacing:.1em;
  text-transform:uppercase;color:var(--lc);border:1px solid var(--lc);background:rgba(0,0,0,.3);}
.log-title{font-family:var(--sans);font-size:12px;font-weight:700;}
.log-date{font-size:10px;color:var(--text2);margin-left:auto;}
.log-text{font-size:11px;color:var(--text1);line-height:1.6;margin-bottom:6px;}
.log-links{display:flex;gap:4px;flex-wrap:wrap;}
.log-link-tag{font-size:9px;padding:2px 7px;border:1px solid var(--border2);border-radius:2px;color:var(--text2);background:var(--bg3);}
.log-del{position:absolute;top:9px;right:9px;display:none;}
.log-entry:hover .log-del{display:flex;}

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
.overlay{position:fixed;inset:0;z-index:500;background:rgba(6,9,13,.85);
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
</div>

<div class="body-area">

<!-- SIDEBAR -->
<div class="sidebar">
  <div class="sidebar-section">
    <div class="sidebar-label">Módulos</div>
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

<!-- MAIN -->
<div class="main">

  <!-- DEV -->
  <div class="panel active" id="panel-dev">
    <div class="panel-header">
      <div class="panel-title" style="color:var(--orange)">⚙️ DEV <span class="panel-sub">— Portafolio de productos</span></div>
      <div class="panel-actions">
        <button class="btn" onclick="expandAllProducts()">Expandir todo</button>
        <button class="btn" onclick="openProductModal()">+ Producto</button>
        <button class="btn btn-primary" onclick="openTaskModal()">+ Task</button>
      </div>
    </div>
    <div class="dev-wrap">
      <div class="dev-body" id="dev-body"></div>
    </div>
  </div>

  <!-- CAMPAÑA — Dashboard de la Desarrolladora -->
  <div class="panel" id="panel-campaign">
    <div class="panel-header">
      <div class="panel-title" style="color:var(--purple-light)">🚀 CAMPAÑA <span class="panel-sub">— Dashboard de la Desarrolladora · datos en vivo del módulo DEV</span></div>
      <div class="panel-actions"><button class="btn" onclick="loadDevMetrics().then(renderCampaign)">↻ Actualizar</button></div>
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
          <button class="btn btn-sm" onclick="setLogFilter(this,'Decision')">Decisión</button>
          <button class="btn btn-sm" onclick="setLogFilter(this,'Insight')">Insight</button>
          <button class="btn btn-sm" onclick="setLogFilter(this,'Riesgo')">Riesgo</button>
          <button class="btn btn-sm" onclick="setLogFilter(this,'Oportunidad')">Oportunidad</button>
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

<div class="toast-wrap" id="toast-wrap"></div>

<!-- MODAL TASK -->
<div class="overlay" id="modal-task" onclick="overlayClose(event,'modal-task')">
  <div class="modal">
    <div class="modal-head"><div class="modal-title" id="task-modal-title">Nueva Task</div><button class="modal-close" onclick="closeModal('modal-task')">✕</button></div>
    <div class="modal-body">
      <input type="hidden" id="task-id"/>
      <div class="f-row">
        <div class="f-group"><label class="f-label">Producto *</label><select class="f-select" id="t-product"></select></div>
        <div class="f-group"><label class="f-label">Módulo *</label><input class="f-input" id="t-module" placeholder="Backend, UI, Auth..."/></div>
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
    <div class="modal-head"><div class="modal-title">Nuevo Log Estratégico</div><button class="modal-close" onclick="closeModal('modal-log')">✕</button></div>
    <div class="modal-body">
      <div class="f-row">
        <div class="f-group"><label class="f-label">Tipo</label><select class="f-select" id="l-type"><option value="Decision">💡 Decisión</option><option value="Insight">🔍 Insight</option><option value="Riesgo">⚠️ Riesgo</option><option value="Oportunidad">🚀 Oportunidad</option></select></div>
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
      <div style="display:flex;gap:7px"><button class="btn" onclick="closeModal('modal-log')">Cancelar</button><button class="btn btn-primary" onclick="saveLog()">Registrar</button></div>
    </div>
  </div>
</div>

<script>
const API='';
let STATE={tasks:[],devMetrics:{},logs:[],stats:{},products:[]};
let logFilter='';
let modExp={Auth:true,Backend:true,UI:true,'Multi-IA':true};
let charts={weekly:null,status:null};

const LOG_C={Decision:'#00e5ff',Insight:'#ffd700',Riesgo:'#ff3e5e',Oportunidad:'#39ff14'};
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
  document.querySelectorAll('.nav-tab').forEach(b=>b.classList.remove('active'));
  (btnEl||document.querySelector('.nav-tab[data-panel="'+name+'"]'))?.classList.add('active');
  document.querySelectorAll('.sidebar-link').forEach(b=>b.classList.remove('active'));
  document.getElementById('sl-'+name)?.classList.add('active');
  if(name==='campaign') loadDevMetrics().then(()=>renderCampaign()).catch(()=>renderCharts());
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
  const devTaskEl=document.getElementById('nb-dev-tasks');
  if(devTaskEl) devTaskEl.textContent=(s.tasks_doing||0)+' doing';
  setText('nb-str',s.logs_total||0);setText('slb-str',s.logs_total||0);
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

function openTaskModal(id=null,defaultProdId=null,defaultMod=null){
  document.getElementById('task-id').value='';
  ['t-name','t-desc'].forEach(i=>{const el=document.getElementById(i);if(el)el.value='';});
  fillProductSelect(defaultProdId||STATE.products[0]?.id);
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
  const id=document.getElementById('task-id').value;if(!id||!confirm('¿Eliminar?'))return;
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
  const id=document.getElementById('product-id').value;if(!id||!confirm('¿Eliminar producto y todas sus tasks?'))return;
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
      <button class="btn btn-sm btn-danger btn-icon log-del" onclick="deleteLog('${l.id}')">✕</button>
      <div class="log-header"><span class="log-badge">${l.type}</span>${l.title?`<span class="log-title">${l.title}</span>`:''}<span class="log-date">${l.date||''}</span></div>
      <div class="log-text">${l.text}</div>
      ${links.length?`<div class="log-links">${links.map(lk=>`<span class="log-link-tag">🔗 ${lk}</span>`).join('')}</div>`:''}</div>`;
  }).join('');
}
function openLogModal(){
  ['l-title','l-text','l-links'].forEach(i=>document.getElementById(i).value='');
  document.getElementById('l-type').value='Decision';
  document.getElementById('l-date').value=new Date().toISOString().slice(0,10);
  openModal('modal-log');
}
async function saveLog(){
  const text=document.getElementById('l-text').value.trim();if(!text){shake('l-text');return;}
  const links=v('l-links').split(',').map(s=>s.trim()).filter(Boolean);
  try{await api('/api/logs','POST',{type:v('l-type'),title:v('l-title'),text,links,date:v('l-date')});closeModal('modal-log');await loadLogs();await loadStats();renderStrategy();renderStats();toast('Log registrado','success');}
  catch(e){toast(e.message,'error');}
}
async function deleteLog(id){
  if(!confirm('¿Eliminar?'))return;
  try{await api(`/api/logs/${id}`,'DELETE');await loadLogs();await loadStats();renderStrategy();renderStats();toast('Log eliminado','error');}
  catch(e){toast(e.message,'error');}
}

// modals
function openModal(id){document.getElementById(id).classList.add('open');}
function closeModal(id){document.getElementById(id).classList.remove('open');}
function overlayClose(e,id){if(e.target.id===id)closeModal(id);}
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
  if(e.key==='Escape'){document.querySelectorAll('.overlay.open').forEach(o=>o.classList.remove('open'));return;}
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
    appendMsg('bot','Soy AETHY. Puedo crear y modificar productos y tareas, moverlas, eliminarlas y darte el avance de la desarrolladora. Probá:<br><em>"Cargá el producto X con las tareas A, B, C"</em> · <em>"¿cómo viene el avance?"</em>');
  }
  document.getElementById('chat-messages').scrollTop=9999;
}

function appendMsg(role,html){
  chatHistory.push({role:role==='user'?'user':'assistant',content:html.replace(/<[^>]+>/g,'')});
  const wrap=document.getElementById('chat-messages');
  const div=document.createElement('div');
  div.className='chat-msg '+(role==='user'?'user':'bot');
  div.innerHTML=`<div class="chat-bubble">${html}</div>`;
  wrap.appendChild(div);
  wrap.scrollTop=wrap.scrollHeight;
  const saved=JSON.parse(localStorage.getItem('aethyChat')||'[]');
  saved.push({role:role==='user'?'user':'assistant',html});
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
    // Send only last 12 turns to avoid token overflow
    const msgs=chatHistory.slice(-12);
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

function clearChat(){
  if(!confirm('¿Limpiar todo el historial de AETHY?'))return;
  chatHistory=[];
  document.getElementById('chat-messages').innerHTML='';
  localStorage.removeItem('aethyChat');
  toast('Historial limpiado','info');
}

document.addEventListener('DOMContentLoaded',()=>{
  const saved=JSON.parse(localStorage.getItem('aethyChat')||'[]');
  saved.forEach(({role,html})=>{
    chatHistory.push({role,content:html.replace(/<[^>]+>/g,'')});
    const wrap=document.getElementById('chat-messages');
    if(!wrap)return;
    const div=document.createElement('div');
    div.className='chat-msg '+(role==='user'?'user':'bot');
    div.innerHTML=`<div class="chat-bubble">${html}</div>`;
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
