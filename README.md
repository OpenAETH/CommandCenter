# OpenAETH — Command Core

Panel de operaciones para startups. Gestiona CRM de prensa y partners, backlog de desarrollo, métricas de campañas y logs estratégicos desde una única interfaz web sin dependencias de frontend.

---

## Stack técnico

| Capa | Tecnología |
|---|---|
| Web framework | Flask 3.1 |
| WSGI server | Gunicorn 23 (2 workers, timeout 120 s) |
| ORM / query | SQLAlchemy 2.0 (Core, sin ORM declarativo) |
| Driver DB | psycopg2-binary 2.9 |
| Base de datos | PostgreSQL vía Supabase |
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
│    └── Frontend  GET /  (HTML inline)        │
└────────────────────┬────────────────────────┘
                     │ SSL/TLS + IPv4 forzado
         ┌───────────▼────────────┐
         │   Supabase (PostgreSQL) │
         │   Pooler: puerto 5432   │
         └────────────────────────┘
```

El frontend es un SPA de página única servido directamente por Flask como string HTML. No existe un proceso de build ni archivos estáticos separados.

---

## Solución al problema IPv6 / Render → Supabase

Render resuelve hostnames de Supabase como IPv6, pero el pooler de Supabase solo acepta IPv4. La conexión se establece mediante un `creator` personalizado pasado a SQLAlchemy:

```python
def _make_ipv4_connection():
    ipv4 = socket.getaddrinfo(host, port, socket.AF_INET)[0][4][0]
    return psycopg2.connect(
        host=p['host'],    # hostname real → SNI correcto en TLS
        hostaddr=ipv4,     # IP IPv4 → evita el DNS lookup que devuelve IPv6
        sslmode='require',
        ...
    )

engine = create_engine(
    "postgresql+psycopg2://",
    creator=_make_ipv4_connection,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=300,
)
```

`host` transporta el SNI para la autenticación TLS; `hostaddr` fuerza la IP resuelta. Sin ambos parámetros simultáneos la conexión falla.

---

## API REST

Todos los endpoints responden y consumen `application/json`. No hay autenticación (la app asume red privada o acceso controlado vía Render).

### Contacts

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/api/contacts` | Lista todos, ordenados por `updated_at DESC` |
| POST | `/api/contacts` | Crea contacto |
| PUT | `/api/contacts/<id>` | Actualización completa |
| DELETE | `/api/contacts/<id>` | Elimina contacto e interacciones en cascada |
| PATCH | `/api/contacts/<id>/status` | Actualiza solo el campo `status` |

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
| GET | `/api/logs` | Lista logs ordenados por `created_at DESC`; `links` deserializado como array |
| POST | `/api/logs` | Crea log; `links` se serializa como JSON string en DB |
| DELETE | `/api/logs/<id>` | Elimina log |

### Utilidades

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/api/stats` | Devuelve contadores agregados de todas las tablas |
| GET | `/api/health` | Verifica conexión a DB e informa la IP IPv4 resuelta |
| GET | `/` | Sirve el frontend HTML completo |

---

## Schema de base de datos

```
contacts              products
├─ id (PK)            ├─ id (PK)
├─ name               ├─ name
├─ medio              ├─ icon
├─ empresa            ├─ description
├─ tipo               ├─ status
├─ status             ├─ color
├─ email              ├─ sort_order
├─ telefono           └─ created_at / updated_at
├─ last_contact
├─ next_followup      tasks
├─ notes              ├─ id (PK)
└─ created_at /       ├─ product_id (FK → products)
   updated_at         ├─ module
                      ├─ name
contact_interactions  ├─ description
├─ id (PK)            ├─ status  (todo | doing | done)
├─ contact_id (FK)    ├─ priority
├─ type               ├─ impact
├─ note               ├─ done  (0 | 1)
└─ date               └─ created_at / updated_at

campaigns             strategy_logs
├─ id (PK)            ├─ id (PK)
├─ name               ├─ type  (Decision | Insight | Riesgo | Oportunidad)
├─ icon               ├─ title
├─ visitas            ├─ text
├─ conversion         ├─ links  (JSON string → array)
├─ leads              ├─ date
├─ backers            └─ created_at
└─ notes
```

Índices definidos sobre `contacts.updated_at`, `tasks.product_id`, `tasks.status`, `strategy_logs.created_at` y `contact_interactions.contact_id`.

---

## Variables de entorno

| Variable | Descripción | Requerida |
|---|---|---|
| `DATABASE_URL` | Connection string PostgreSQL completo | Sí |
| `PORT` | Puerto de escucha (inyectado por Render automáticamente) | Sí (auto) |

Formato esperado de `DATABASE_URL`:
```
postgresql://USER:PASSWORD@HOST:PORT/DBNAME
```

---

## Deploy en Render

La configuración está declarada en `render.yaml`:

- **Runtime:** Python 3.11
- **Build:** `pip install -r requirements.txt`
- **Start:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
- La variable `DATABASE_URL` debe setearse manualmente en el dashboard de Render (marcada como `sync: false`).

---

## Setup local

```bash
# 1. Clonar y crear entorno virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar variable de entorno
export DATABASE_URL="postgresql://USER:PASSWORD@HOST:PORT/DBNAME"

# 4. Ejecutar el schema en Supabase (SQL Editor)
# → copiar y ejecutar supabase_schema.sql

# 5. Iniciar servidor de desarrollo
python app.py
# → http://localhost:5000
```

---

## Paneles del frontend

El SPA incluye cinco secciones navegables por teclado (`1`–`5`) o clic:

- **CRM** (`1` o `/` para buscar) — Gestión de contactos de prensa, partners y clientes con estados y seguimiento
- **DEV** (`2`) — Backlog de tareas agrupadas por producto y módulo, con toggle de completado
- **Campañas** (`3`) — Métricas por canal (visitas, conversión, leads, backers) con gráficos Chart.js
- **Estrategia** (`4`) — Log de decisiones, insights, riesgos y oportunidades con filtrado por tipo
- **Guía** (`5`) — Panel de referencia interno

---

## Archivos del proyecto

```
.
├── app.py               # Aplicación completa (API + frontend embebido)
├── requirements.txt     # Dependencias Python
├── render.yaml          # Configuración de deploy en Render
├── supabase_schema.sql  # DDL + datos de seed para Supabase
└── .gitignore
```
