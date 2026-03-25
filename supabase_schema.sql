-- ============================================================
--  OpenAETH — Supabase / PostgreSQL Schema + Seed Data
--  Ejecutar en: Supabase → SQL Editor → New Query
-- ============================================================

-- ── TABLAS ───────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS contacts (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    medio       TEXT,
    empresa     TEXT,
    tipo        TEXT DEFAULT 'prensa',
    status      TEXT DEFAULT 'nuevo',
    email       TEXT,
    telefono    TEXT,
    last_contact TEXT,
    next_followup TEXT,
    notes       TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS products (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    icon        TEXT DEFAULT '📦',
    description TEXT,
    status      TEXT DEFAULT 'activo',
    color       TEXT DEFAULT '#00e5ff',
    sort_order  INTEGER DEFAULT 0,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS tasks (
    id          SERIAL PRIMARY KEY,
    product_id  INTEGER REFERENCES products(id) ON DELETE SET NULL,
    module      TEXT NOT NULL,
    name        TEXT NOT NULL,
    description TEXT,
    status      TEXT DEFAULT 'todo',
    priority    TEXT DEFAULT 'medio',
    impact      TEXT DEFAULT 'medio',
    done        INTEGER DEFAULT 0,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS campaigns (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    icon        TEXT DEFAULT '📊',
    visitas     INTEGER DEFAULT 0,
    conversion  REAL DEFAULT 0,
    leads       INTEGER DEFAULT 0,
    backers     INTEGER DEFAULT 0,
    notes       TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS strategy_logs (
    id          SERIAL PRIMARY KEY,
    type        TEXT NOT NULL,
    title       TEXT,
    text        TEXT NOT NULL,
    links       TEXT DEFAULT '[]',
    date        DATE DEFAULT CURRENT_DATE,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS contact_interactions (
    id          SERIAL PRIMARY KEY,
    contact_id  INTEGER REFERENCES contacts(id) ON DELETE CASCADE,
    type        TEXT,
    note        TEXT,
    date        DATE DEFAULT CURRENT_DATE
);

-- ── ÍNDICES ──────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_contacts_updated   ON contacts(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_tasks_product      ON tasks(product_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status       ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_strategy_logs_date ON strategy_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_interactions_contact ON contact_interactions(contact_id);

-- ── SEED: CONTACTS ───────────────────────────────────────────

INSERT INTO contacts (name, medio, empresa, tipo, status, last_contact, next_followup, notes) VALUES
    ('Valentina Cruz',  'TechCrunch ES', 'TechCrunch', 'prensa',  'contactado',     '2025-01-18', '2025-01-25', 'Interesada en IA conversacional para startups LATAM'),
    ('Marcos Oliveira', 'YC Alumni',     'YC W23',     'partner', 'en conversacion','2025-01-17', '2025-01-22', 'Posible co-inversor. Tiene red en Brasil y México.'),
    ('SaaS Demo Corp',  'LinkedIn',      'SaaS Demo',  'cliente', 'interesado',     '2025-01-15', '2025-01-21', 'Quieren integrar Multi-IA en soporte al cliente'),
    ('Diego Herrera',   'Twitter/X',     'Freelance',  'prensa',  'nuevo',          '',           '2025-01-22', 'Tech journalist, 40k seguidores. Cubre IA y startups.'),
    ('Ana Rivas',       'ProductHunt',   'PH',         'partner', 'cerrado',        '2025-01-10', '',           'Deal cerrado. Lanzamiento en PH coordinado.')
ON CONFLICT DO NOTHING;

-- ── SEED: TASKS ──────────────────────────────────────────────

INSERT INTO tasks (module, name, description, status, priority, impact, done) VALUES
    ('Auth',     'OAuth2 con Google',     'Flujo completo de autenticación Google',      'done',  'alto',  'alto',  1),
    ('Auth',     'JWT refresh tokens',    'Implementar renovación automática de tokens', 'done',  'alto',  'alto',  1),
    ('Auth',     '2FA via TOTP',          'Autenticación de dos factores',               'todo',  'medio', 'medio', 0),
    ('Backend',  'API endpoints REST',    'CRUD completo para todas las entidades',      'doing', 'alto',  'alto',  0),
    ('Backend',  'Rate limiting',         'Throttling por usuario y plan',               'todo',  'medio', 'medio', 0),
    ('Backend',  'Webhooks sistema',      'Notificaciones a sistemas externos',          'todo',  'medio', 'alto',  0),
    ('Backend',  'Caché Redis',           'Caching de respuestas frecuentes',            'todo',  'bajo',  'medio', 0),
    ('UI',       'Onboarding flow',       'Wizard de configuración inicial',             'doing', 'alto',  'alto',  0),
    ('UI',       'Dashboard métricas',    'Visualización de uso y costos',               'todo',  'medio', 'medio', 0),
    ('UI',       'Chat widget embed',     'Widget embebible para clientes',              'todo',  'alto',  'alto',  0),
    ('Multi-IA', 'Conector GPT-4',        'Integración OpenAI completa',                 'done',  'alto',  'alto',  1),
    ('Multi-IA', 'Conector Claude',       'Integración Anthropic API',                  'doing', 'alto',  'alto',  0),
    ('Multi-IA', 'Router de modelos',     'Selección automática por costo/calidad',      'todo',  'alto',  'alto',  0),
    ('Multi-IA', 'Fallback automático',   'Redirigir si un modelo falla',               'todo',  'alto',  'alto',  0)
ON CONFLICT DO NOTHING;

-- ── SEED: CAMPAIGNS ──────────────────────────────────────────

INSERT INTO campaigns (name, icon, visitas, conversion, leads, backers, notes) VALUES
    ('Landing',    '🏠', 1240, 3.2, 40, 12, 'Iterar CTA principal. El headline actual convierte poco. Probar "sin código" como hook.'),
    ('Recompensas','🎁',  340, 8.8, 30,  8, 'Tier $99 tiene mejor conversión. Agregar testimonial de beta user.'),
    ('Video Demo', '🎬',  560, 2.1, 12,  3, 'Demo 90s funciona mejor que 3min. Agregar subtítulos en español.'),
    ('Prensa',     '📰',  180, 5.5, 10,  2, 'Primer artículo en AI Weekly. Preparar kit de prensa con screenshots.')
ON CONFLICT DO NOTHING;

-- ── SEED: STRATEGY LOGS ──────────────────────────────────────

INSERT INTO strategy_logs (type, title, text, links, date) VALUES
    ('Decision',    'Pricing SMB',        'Pivotamos el pricing a $99/mes para SMBs. Enterprise queda para v2.',              '["CRM→Marcos","DEV→Backend"]', '2025-01-18'),
    ('Insight',     'Canal Twitter',      'Los leads de Twitter convierten 3x más rápido que LinkedIn. Redirigir esfuerzo.',  '["Campaña→Landing"]',          '2025-01-17'),
    ('Riesgo',      'Dependencia OpenAI', 'Multi-IA dependency en OpenAI. Si sube precios >40%, el margen colapsa.',         '["DEV→Multi-IA"]',             '2025-01-16'),
    ('Oportunidad', 'Vertical HR',        '3 empresas de HR preguntaron por integración con ATS. Sin competidor directo.',    '["CRM→SaaS Demo"]',            '2025-01-15')
ON CONFLICT DO NOTHING;
