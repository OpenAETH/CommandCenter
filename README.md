# OpenAETH — Command Core

Panel de operaciones para startups, centrado en **gestión de desarrollo con IA integrada**. Administra el backlog de productos y tareas (módulo DEV), métricas de campaña vinculadas al cumplimiento de DEV y logs estratégicos — todo desde una única interfaz web sin build step, con un asistente IA que opera el sistema por chat.

---

## Stack técnico

| Capa | Tecnología |
|---|---|
| Web framework | Flask 3.1 |
| WSGI server | Gunicorn 23 (2 workers, timeout 120 s) |
| Driver DB | PyMongo 4.10 |
| Base de datos | MongoDB (Atlas, vía `MONGODB_URI`) |
| IA / Chatbot | Groq · modelo `qwen/qwen3-32b` (function calling + reasoning) |
| Frontend | HTML/CSS/JS embebido en `app.py` (sin build step) |
| Charts | Chart.js 4.4 (CDN) |
| Deploy | Render (Web Service, Python runtime) |
| Python | 3.11 |

---

## Arquitectura

```
┌─────────────────────────────────────────────┐
│                  Render                      │
│  Gunicorn → Flask app (app.py)               │
│    ├── REST API  /api/*                      │
│    ├── Chatbot   /api/chat  (Groq tools)     │
│    └── Frontend  GET /  (HTML inline)        │
└──────────┬───────────────────────┬───────────┘
           │ TLS (certifi)         │ HTTPS
   ┌───────▼────────┐      ┌────────▼────────┐
   │ MongoDB Atlas  │      │   Groq API      │
   │ db: commandc.  │      │ qwen/qwen3-32b  │
   └────────────────┘      └─────────────────┘
```

El frontend es un SPA de página única servido directamente por Flask como string HTML. No existe proceso de build ni archivos estáticos separados.

---

## Asistente IA (chatbot)

El endpoint `/api/chat` expone un agente Groq con function calling. Las herramientas son **self-contained** (buscan productos por nombre internamente, sin pasar IDs entre rondas), lo que evita races al crear producto + tareas en paralelo.

Herramientas disponibles:

| Tool | Qué hace |
|---|---|
| `create_product_with_tasks` | Crea un producto y todas sus tareas en una operación atómica |
| `add_tasks_to_product` | Agrega tareas a un producto existente (búsqueda por nombre) |
| `list_products` | Lista productos con id, nombre y estado |
| `list_tasks` | Lista tareas, opcionalmente filtradas por producto |
| `update_tasks` | Actualiza tareas por nombre + producto (sin IDs) |

El modelo `qwen/qwen3-32b` es un modelo *reasoning*: emite bloques `<think>…</think>`. La app los oculta vía `reasoning_format="hidden"` y, como red de seguridad, los limpia con `strip_think()` antes de devolver la respuesta.

---

## API REST

Todos los endpoints responden y consumen `application/json`. No hay autenticación (la app asume red privada o acceso controlado vía Render).

### Products

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/api/products` | Lista, ordenada por `sort_order` |
| POST | `/api/products` | Crea producto; `sort_order` se asigna automáticamente |
| PUT | `/api/products/<id>` | Actualización completa |
| DELETE | `/api/products/<id>` | Elimina producto y sus tasks en cascada |

### Tasks

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/api/tasks` | Lista todas, ordenadas por `product_id, module, id` |
| POST | `/api/tasks` | Crea tarea; `done` se deriva de `status` |
| PUT | `/api/tasks/<id>` | Actualización completa |
| PATCH | `/api/tasks/<id>/toggle` | Alterna `done` entre 0/1 y actualiza `status` |
| DELETE | `/api/tasks/<id>` | Elimina tarea |

### Campaigns

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/api/campaigns` | Lista canales de campaña |
| POST | `/api/campaigns` | Crea canal |
| PUT | `/api/campaigns/<id>` | Actualización completa de métricas |
| DELETE | `/api/campaigns/<id>` | Elimina canal |

### Strategy Logs

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/api/logs` | Lista logs ordenados por `created_at DESC` |
| POST | `/api/logs` | Crea log (`type`, `title`, `text`, `links`, `date`) |
| DELETE | `/api/logs/<id>` | Elimina log |

### Chat & utilidades

| Método | Ruta | Descripción |
|--------|------|-------------|
| POST | `/api/chat` | Agente Groq con tools sobre productos/tareas |
| GET | `/api/stats` | Contadores agregados (tasks, productos, logs) |
| GET | `/api/health` | Verifica conexión a Mongo y disponibilidad de Groq |
| GET | `/` | Sirve el frontend HTML completo |

---

## Colecciones MongoDB

```
products                 tasks
├─ _id                   ├─ _id
├─ name                  ├─ product_id  (str → products._id)
├─ icon                  ├─ module
├─ description           ├─ name
├─ status                ├─ description
├─ color                 ├─ status   (todo | doing | done)
├─ sort_order            ├─ priority (alto | medio | bajo)
└─ created_at/updated_at ├─ impact   (alto | medio | bajo)
                         ├─ done     (0 | 1)
campaigns                └─ created_at/updated_at
├─ _id
├─ name                  strategy_logs
├─ icon                  ├─ _id
├─ visitas               ├─ type  (Decision | Insight | Riesgo | Oportunidad)
├─ conversion            ├─ title
├─ leads                 ├─ text
├─ backers               ├─ links  (array)
├─ notes                 ├─ date
└─ created_at/updated_at └─ created_at
```

> La colección `contacts` (del antiguo módulo CRM, migrado a otra aplicación) puede existir en la base pero **ya no es usada** por esta app.

---

## Variables de entorno

| Variable | Descripción | Requerida | Default |
|---|---|---|---|
| `MONGODB_URI` | Connection string de MongoDB Atlas | Sí | — |
| `MONGODB_DB` | Nombre de la base | No | `commandcenter` |
| `GROQ_API_KEY` | API key de Groq (habilita el chatbot) | No | — |
| `GROQ_MODEL` | Modelo Groq a usar | No | `qwen/qwen3-32b` |
| `PORT` | Puerto de escucha (inyectado por Render) | Sí (auto) | — |

---

## Deploy en Render

Configuración declarada en `render.yaml`:

- **Runtime:** Python 3.11
- **Build:** `pip install -r requirements.txt`
- **Start:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
- `MONGODB_URI`, `MONGODB_DB` y `GROQ_API_KEY` se setean manualmente en el dashboard (`sync: false`). `GROQ_MODEL` viene con default en el YAML.

---

## Setup local

```bash
# 1. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar variables de entorno
export MONGODB_URI="mongodb+srv://USER:PASSWORD@CLUSTER/?retryWrites=true&w=majority"
export MONGODB_DB="commandcenter"
export GROQ_API_KEY="gsk_..."          # opcional, habilita el chatbot
export GROQ_MODEL="qwen/qwen3-32b"     # opcional

# 4. Iniciar servidor de desarrollo
python app.py
# → http://localhost:5000
```

---

## Paneles del frontend

El SPA incluye cuatro secciones navegables por teclado (`1`–`4`) o clic, más el asistente IA flotante (🤖):

- **DEV** (`1`) — Backlog de tareas agrupadas por producto y módulo, con toggle de completado y progreso en tiempo real. Panel por defecto.
- **Campaña** (`2`) — Métricas por canal (visitas, conversión, leads, backers) con gráficos Chart.js.
- **Estrategia** (`3`) — Log de decisiones, insights, riesgos y oportunidades con filtrado por tipo.
- **Guía** (`4`) — Panel de referencia interno.

---

## Archivos del proyecto

```
.
├── app.py            # Aplicación completa (API + chatbot + frontend embebido)
├── requirements.txt  # Dependencias Python
├── render.yaml       # Configuración de deploy en Render
└── .gitignore
```
