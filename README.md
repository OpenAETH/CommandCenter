# OpenAETH — Command Core

![Python 3.11](https://img.shields.io/badge/python-3.11-blue?logo=python)
![Flask 3.1](https://img.shields.io/badge/Flask-3.1-black?logo=flask)
![MongoDB Atlas](https://img.shields.io/badge/MongoDB-Atlas-green?logo=mongodb)
![Groq](https://img.shields.io/badge/Groq-Qwen3--32B-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

Panel de operaciones para startups con **AETHY**, un asistente IA que opera el sistema por chat en español. Gestioná productos y tareas (DEV), metricas de campana vinculadas al progreso de desarrollo y una bitacora estrategica — todo desde una unica interfaz web sin build steps, con un solo archivo Python.

---

## Features

| Area | Que hace |
|------|----------|
| **AETHY** | Asistente IA conversacional que crea, lee, modifica y elimina productos/tareas via chat en espanol |
| **DEV** | Backlog de productos y tareas con estados todo/doing/done, agrupado por modulo |
| **Campana** | Metricas en vivo: tareas completadas por semana, progreso por producto, % de avance |
| **Estrategia** | Bitacora tipo "segundo cerebro": decisiones, insights, riesgos, oportunidades, aprendizajes, objetivos, hipotesis e hitos |
| **Guia** | Panel de referencia interna con tips de uso diario |
| **Sin build steps** | Un solo `pip install`, una variable de entorno, y ya corre |

---

## Stack tecnico

| Capa | Tecnologia |
|------|-----------|
| Web framework | Flask 3.1 |
| WSGI server | Gunicorn 23 (2 workers, timeout 120 s) |
| Driver DB | PyMongo 4.10 |
| Base de datos | MongoDB (Atlas, via `MONGODB_URI`) |
| IA / Chatbot | Groq · modelo `qwen/qwen3-32b` (function calling + reasoning) |
| Frontend | HTML/CSS/JS embebido en `app.py` (sin build step) |
| Charts | Chart.js 4.4 (CDN) |
| Markdown | marked.js 15.0 + DOMPurify 3.2.4 (CDN) |
| Deploy | Render (Web Service, Python runtime) |
| Python | 3.11 |

---

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                        Render (Web Service)                     │
│  Gunicorn -> Flask app (app.py)                                  │
│    +-- REST API  /api/*            (CRUD products/tasks/logs)    │
│    +-- Chatbot  /api/chat          (Groq function calling)      │
│    +-- Frontend GET /              (SPA inline HTML string)      │
└────────┬──────────────────────────────────┬──────────────────────┘
         │ TLS (certifi)                     │ HTTPS
  ┌──────▼──────────┐                ┌──────▼──────────┐
  │  MongoDB Atlas   │                │   Groq API      │
  │  db: commandc.   │                │ qwen/qwen3-32b  │
  │  products        │                └─────────────────┘
  │  tasks           │
  │  strategy_logs   │
  └──────────────────┘

┌────────── Frontend (SPA) ──────────────────────────────────┐
│  DEV (1)  |  Campana (2)  |  Estrategia (3)  |  Guia (4) │
│  Productos/tareas  |  Chart.js  |  Logs filtrables  |  Ref │
│  + AETHY chatbot flotante (esquina inf. der.)              │
└────────────────────────────────────────────────────────────┘
```

El frontend es un SPA de pagina unica servido directamente por Flask como string HTML. No existe proceso de build ni archivos estaticos separados. Navegacion por teclado (`1`-`4`) y bottom nav en mobile.

---

## Principios de diseno

- **Self-contained tools**: el chatbot busca productos y tareas por **nombre** (regex case-insensitive), no por ID. Elimina condiciones de carrera cuando el LLM ejecuta varias tools en paralelo.
- **Monolito deliberado**: un solo archivo (`app.py`) contiene backend, API, chatbot y frontend. Sin npm, sin build steps, sin carpetas static/. Deploy inmediato.
- **Metricas derivadas**: el panel Campana no almacena datos propios. Todas las metricas (progreso, completados por semana, % por producto) se calculan en vivo desde `products` y `tasks`.
- **Reasoning oculto**: el modelo Qwen3 emite bloques `<think>...</think>` internos. Se ocultan via `reasoning_format="hidden"` y se limpian con `strip_think()` como respaldo.
- **Budget para reasoning + tools**: Qwen3 es un modelo reasoning; el razonamiento consume el mismo presupuesto de salida (`max_tokens`) que el JSON de los tool calls. Por eso `GROQ_MAX_TOKENS` arranca en 4096 (un valor bajo trunca operaciones multi-tarea). El loop detecta `finish_reason == "length"` y avisa en vez de devolver una respuesta parcial; ante un JSON de argumentos truncado devuelve el error al modelo para que reintente fragmentando, en lugar de ejecutar con datos incompletos.
- **Modulos reutilizables**: al agregar tareas, el modulo pedido se resuelve contra los ya existentes del producto (normalizando mayusculas/acentos), evitando duplicados como `Backend` vs `backend`. Solo crea un modulo nuevo si ninguno encaja.

---

## Asistente IA — AETHY

El endpoint `POST /api/chat` expone **AETHY**, un agente Groq con function calling que opera directamente sobre la base de datos. Responde en espanol, mantiene contexto de 12 turnos y ejecuta hasta 8 iteraciones tool-call por mensaje. Cada tool call se persiste en `tool_logs` (auditoria liviana: timestamp, args, resultado, ok, finish_reason).

### Las 12 herramientas

| Tool | Que hace | Comportamiento |
|------|----------|---------------|
| `create_product_with_tasks` | Crea un producto + N tareas en una sola operacion atomica | Busca por nombre internamente; valida antes de escribir |
| `add_tasks_to_product` | Agrega tareas a un producto existente | Busca producto por nombre parcial |
| `list_products` | Lista productos con id, nombre y estado | Sin parametros |
| `list_tasks` | Lista tareas, opcionalmente filtradas por producto | Filtro por nombre parcial |
| `list_modules` | Lista los modulos existentes de un producto | Para reutilizar modulos y no duplicarlos |
| `update_tasks` | Actualiza status/priority/impact/module de una o varias tareas | Busca por nombre + producto |
| `update_product` | Modifica nombre/icono/descripcion/status/color de un producto | Busca por nombre parcial |
| `move_tasks` | Mueve tareas a otro producto y/o cambia su modulo | Busca por nombre parcial |
| `delete_tasks` | Elimina tareas especificas | **Destructiva** — solo si el usuario lo pide explicitamente |
| `delete_product` | Elimina un producto y TODAS sus tareas en cascada | **Destructiva** — requiere confirmacion explicita |
| `get_dev_metrics` | Devuelve metricas agregadas: productos, tareas por estado, % completado, completadas esta semana, progreso por producto | Sin parametros |
| `create_log` | Registra entrada en bitacora estrategica (Decision/Insight/Riesgo/Oportunidad/Aprendizaje/Objetivo/Hipotesis/Hito) | Guarda texto COMPLETO y literal |

> El modelo `qwen/qwen3-32b` es un modelo *reasoning*: emite bloques `<think>...</think>` internos. La app los oculta automaticamente.

---

## API REST

Todos los endpoints responden y consumen `application/json`. No hay autenticacion (la app asume red privada o acceso controlado via Render).

### Products

| Metodo | Ruta | Descripcion |
|--------|------|-------------|
| GET | `/api/products` | Lista, ordenada por `sort_order` |
| POST | `/api/products` | Crea producto; `sort_order` se asigna automaticamente |
| PUT | `/api/products/<id>` | Actualizacion completa |
| DELETE | `/api/products/<id>` | Elimina producto y sus tasks en cascada |

### Tasks

| Metodo | Ruta | Descripcion |
|--------|------|-------------|
| GET | `/api/tasks` | Lista todas, ordenadas por `product_id, module, id` |
| POST | `/api/tasks` | Crea tarea; `done` y `completed_at` se derivan de `status` |
| PUT | `/api/tasks/<id>` | Actualizacion completa |
| PATCH | `/api/tasks/<id>/toggle` | Alterna done/doing y sincroniza `completed_at` |
| DELETE | `/api/tasks/<id>` | Elimina tarea |

### Dev Metrics

| Metodo | Ruta | Descripcion |
|--------|------|-------------|
| GET | `/api/dev-metrics` | Metricas agregadas: productos activos, tareas por estado, % completado, completadas esta semana, progreso por producto |

> Este endpoint es el corazon del panel Campana. Calcula todo en vivo desde las colecciones `products` y `tasks`.

### Strategy Logs

| Metodo | Ruta | Descripcion |
|--------|------|-------------|
| GET | `/api/logs` | Lista logs ordenados por `created_at DESC` |
| POST | `/api/logs` | Crea log (`type`, `title`, `text`, `links`, `date`) |
| DELETE | `/api/logs/<id>` | Elimina log |

### Chat y utilidades

| Metodo | Ruta | Descripcion |
|--------|------|-------------|
| POST | `/api/chat` | Agente Groq AETHY con 12 tools sobre productos/tareas |
| GET | `/api/stats` | Contadores agregados (tasks, productos, logs) |
| GET | `/api/health` | Verifica conexion a Mongo y disponibilidad de Groq |
| GET | `/` | Sirve el frontend HTML completo (SPA) |

---

## Colecciones MongoDB

```
products                            tasks
+-- _id                             +-- _id
+-- name                            +-- product_id     (str -> products._id)
+-- icon        (emoji)             +-- module         (ej: Backend, UI, Auth)
+-- description                     +-- name
+-- status      (activo|idea|pausado|archivado)  +-- description
+-- color       (hex)               +-- status        (todo | doing | done)
+-- sort_order                      +-- priority      (alto | medio | bajo)
+-- created_at                      +-- impact        (alto | medio | bajo)
+-- updated_at                      +-- done          (0 | 1)
                                    +-- completed_at
                                    +-- created_at
strategy_logs                       +-- updated_at
+-- _id
+-- type        (Decision|Insight|Riesgo|Oportunidad|
|                Aprendizaje|Objetivo|Hipotesis|Hito)
+-- title
+-- text
+-- links       (array de strings)
+-- date
+-- created_at

tool_logs                           (auditoria de tool calls de AETHY)
+-- _id
+-- ts          (timestamp UTC)
+-- tool        (nombre de la herramienta)
+-- args        (argumentos recibidos)
+-- result      (valor devuelto)
+-- ok          (bool — false si result trae error)
+-- finish_reason
```

> Las colecciones `campaigns` (del modulo anterior de campanas manuales) y `contacts` (del antiguo modulo CRM, migrado a otra app) pueden existir en la base pero **ya no son usadas** por esta aplicacion.

---

## Variables de entorno

| Variable | Descripcion | Requerida | Default |
|----------|------------|-----------|---------|
| `MONGODB_URI` | Connection string de MongoDB Atlas | Si | -- |
| `MONGODB_DB` | Nombre de la base | No | `commandcenter` |
| `GROQ_API_KEY` | API key de Groq (habilita el chatbot) | No | -- |
| `GROQ_MODEL` | Modelo Groq a usar | No | `qwen/qwen3-32b` |
| `GROQ_MAX_TOKENS` | Presupuesto de tokens de salida (reasoning + tool calls) | No | `4096` |
| `PORT` | Puerto de escucha (inyectado por Render) | Si (auto) | -- |

---

## Setup local

```bash
# 1. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows (cmd)
# venv\Scripts\Activate.ps1  # Windows (PowerShell)

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar variables de entorno
# Linux/macOS (bash):
export MONGODB_URI="mongodb+srv://USER:PASSWORD@CLUSTER/?retryWrites=true&w=majority"
export MONGODB_DB="commandcenter"
export GROQ_API_KEY="gsk_..."       # opcional, habilita el chatbot
export GROQ_MODEL="qwen/qwen3-32b"  # opcional

# Windows (cmd):
# set MONGODB_URI=mongodb+srv://USER:PASSWORD@CLUSTER/?retryWrites=true&w=majority
# set MONGODB_DB=commandcenter
# set GROQ_API_KEY=gsk_...

# Windows (PowerShell):
# $env:MONGODB_URI="mongodb+srv://USER:PASSWORD@CLUSTER/?retryWrites=true&w=majority"
# $env:MONGODB_DB="commandcenter"
# $env:GROQ_API_KEY="gsk_..."

# 4. Iniciar servidor de desarrollo
python app.py
# -> http://localhost:5000
```

---

## Deploy en Render

Configuracion declarada en `render.yaml`:

- **Runtime:** Python 3.11
- **Build:** `pip install -r requirements.txt`
- **Start:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
- `MONGODB_URI`, `MONGODB_DB` y `GROQ_API_KEY` se setean manualmente en el dashboard (`sync: false`). `GROQ_MODEL` viene con default en el YAML.

---

## Paneles del frontend

El SPA incluye cuatro secciones navegables por teclado (`1`-`4`) o clic, mas el asistente IA flotante (bAe), con navegacion inferior en mobile:

| Panel | Tecla | Descripcion |
|-------|-------|-------------|
| **DEV** | `1` | Backlog de tareas agrupadas por producto y modulo, con toggle de completado, folding de modulos, CRUD desde la UI y progreso en tiempo real. Panel por defecto. |
| **Campana** | `2` | Metricas derivadas de DEV: cards de estado (todo/doing/done), grafico Chart.js semanal (barras), doughnut de completitud y progreso por producto. |
| **Estrategia** | `3` | Bitacora de decisiones, insights, riesgos, oportunidades, aprendizajes, objetivos, hipotesis e hitos, con filtrado por tipo y CRUD completo. |
| **Guia** | `4` | Panel de referencia interna con cards de help desk, tips de uso diario y documentacion de AETHY. |

Atajos de teclado:
- `1`-`4`: Cambiar de panel
- **FAB** `bAe`: Abrir/cerrar chatbot AETHY
- **Esc**: Cerrar modales y chatbot

---

## Ecosistema AETHERYON

Command Core es parte del ecosistema **AETHERYON**, una suite de herramientas para startups centradas en IA:

| Proyecto | Descripcion |
|----------|-------------|
| **Command Core** | Este proyecto — panel de operaciones con IA integrada |
| **PromptForge** | Motor visual de cadenas cognitivas (pipeline designer) — ver `Cognis-PromptForge.md` |

---

## Troubleshooting

| Problema | Causa probable | Solucion |
|----------|---------------|----------|
| El chatbot responde "no disponible" | `GROQ_API_KEY` no configurada | Setear la variable de entorno con una API key valida de Groq |
| Error de conexion a MongoDB | `MONGODB_URI` incorrecta o IP no autorizada | Verificar el connection string y agregar la IP actual en Atlas Network Access |
| La app no arranca | Puerto ocupado | Usar `set PORT=5001` o `$env:PORT=5001` antes de iniciar |
| Los graficos no se ven | Sin conexion a internet (Chart.js via CDN) | Verificar conectividad; los graficos requieren CDN |
| El panel Campana muestra datos vacios | No hay productos/tareas creados | Ir al panel DEV o pedirle a AETHY que cree productos |

---

## Archivos del proyecto

```
.
+-- app.py                     # Aplicacion completa (API + chatbot + frontend embebido, 2728 lines)
+-- requirements.txt           # Dependencias Python (7 paquetes)
+-- render.yaml                # Configuracion de deploy en Render
+-- Cognis-PromptForge.md      # Documentacion de diseno: PromptForge (herramienta hermana)
+-- .gitignore
+-- .claude/
    +-- settings.local.json    # Configuracion local de Claude Code
```
