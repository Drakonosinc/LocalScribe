# LocalScribe

**LocalScribe** es una aplicación de escritorio para Windows (Python 3.12 + PySide6) que transcribe audio en vivo de forma **100 % local**, sin enviar grabaciones a la nube ni requerir claves de API.

Permite capturar el audio del sistema mediante **WASAPI loopback**, el **micrófono**, o **ambas fuentes en simultáneo**, mostrando un borrador provisional en tiempo real y consolidando cada intervención confirmada en un historial permanente, buscable y exportable.

![Captura de pantalla de LocalScribe](docs/screenshot.png)

> _Espacio reservado para la captura de pantalla de la aplicación (`docs/screenshot.png`)._

---

## Características principales

- **Procesamiento 100 % local:** Inferencia en CPU (INT8) con [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper), con perfiles intercambiables (`tiny`, `base`, `small`).
- **Selector de fuente de audio:** Alterná desde la barra principal entre:
  - **Audio del sistema (WASAPI):** reuniones (Zoom, Meet, Teams), clases, navegador o videos.
  - **Micrófono:** captura directa del micrófono predeterminado de Windows.
  - **Ambos (Sistema + Micrófono):** mezcla simultánea de ambas fuentes en un flujo unificado a 16 kHz.
- **Historial estable y borrador protegido:**
  - El texto confirmado se conserva en un panel independiente con marcas de tiempo y **scroll inteligente** (sigue el final automáticamente, pero nunca te mueve si subiste a leer o seleccionás texto).
  - El panel **En vivo** pospone actualizaciones visuales mientras seleccionás texto para copiar.
  - Atajos y botones rápidos: *Copiar selección*, *Copiar último bloque*, *Copiar borrador actual*, *Copiar intervención reciente*, búsqueda (`Ctrl+F`) e *Ir al presente* (`Ctrl+L`).
- **Vocabulario personalizado (`initial_prompt`):** Campo opcional para guiar al modelo con términos técnicos, siglas o nombres propios (por ejemplo: `Docker, Kubernetes, FastAPI, PyTorch...`).
- **Exportación múltiple:** Guardá el historial confirmado en **`.txt`** (texto plano con marcas de tiempo), **`.md`** (Markdown estructurado) o **`.srt`** (subtítulos estándar SubRip).
- **Guardado incremental de sesión:** Cada bloque confirmado se vuelca y sincroniza automáticamente en `sessions/sesion_YYYYMMDD_HHMMSS.txt` a medida que se consolida, evitando pérdidas ante un cierre inesperado.
- **Persistencia de preferencias:** Conserva automáticamente en `config.json` el perfil de modelo, el umbral de silencio, la fuente de audio, el estado de **Siempre visible** y el vocabulario personalizado.
- **Diagnóstico integrado:** Panel ocultable con métricas en vivo (tiempo de inferencia, duración de audio, RTF, retraso y pendiente) y registro rotativo en `logs/diagnostico.log` sin almacenar contenido de voz ni texto transcripto.

---

## Requisitos

- **Sistema operativo:** Windows 10 u 11 (64 bits).
- **Python:** Python **3.12** (64 bits) con el lanzador `py` habilitado en el sistema.
- **Conexión a internet (solo la primera vez):** necesaria para instalar dependencias y descargar los pesos del modelo elegido de `faster-whisper`. Una vez en caché, funciona completamente sin conexión.

---

## Instalación

### Opción recomendada (asistente automático)

1. Cloná o descargá este repositorio en tu equipo.
2. Ejecutá **`INSTALAR.bat`** (una sola vez). El script verifica Python 3.12, crea el entorno virtual `.venv` e instala las dependencias de `requirements.txt`.

### Instalación manual (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## Ejecución

Hacé doble clic en **`INICIAR.bat`** o ejecutá desde PowerShell:

```powershell
.\.venv\Scripts\python.exe main.py
```

1. Elegí la **fuente de audio** (*Audio del sistema*, *Micrófono* o *Ambos*).
2. Seleccioná el **perfil de modelo** (*Rápido `tiny`*, *Equilibrado `base`* o *Preciso `small`*).
3. Ajustá opcionalmente el **Umbral** de detección de voz y el **Vocabulario opcional**.
4. Presioná **▶ Iniciar** para comenzar la transcripción.

---

## Arquitectura del proyecto

```text
LocalScribe/
├── INICIAR.bat              # Lanzador directo en Windows
├── INSTALAR.bat             # Instalador del entorno virtual (.venv)
├── main.py                  # Punto de entrada de la aplicación
├── requirements.txt         # Dependencias de ejecución
├── ruff.toml                # Configuración del linter/formateador
├── localscribe/
│   ├── __init__.py          # Metadatos y versión (v0.3.0)
│   ├── audio.py             # Captura WASAPI loopback, micrófono y mezclador dual
│   ├── config.py            # Carga y guardado automático de config.json
│   ├── pipeline.py          # Segmentación por energía/pausas y motor ASR
│   ├── session.py           # Orquestación de hilos y respaldo incremental en sessions/
│   ├── transcript.py        # Estado reciente y exportación a .txt, .md y .srt
│   └── ui.py                # Interfaz gráfica en PySide6
└── tests/
    ├── test_pipeline.py     # Pruebas de segmentación y cola de inferencia
    ├── test_v02.py          # Pruebas de concurrencia, copia y widgets Qt
    └── test_v03.py          # Pruebas de config.json, sesiones, audio y exportación
```

---

## Desarrollo y pruebas

Las pruebas automatizadas se ejecutan sin abrir dispositivos de audio reales ni descargar modelos (usando la plataforma `offscreen` de Qt):

```powershell
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
.\.venv\Scripts\python.exe -m pytest -v
```

---

## Privacidad y uso responsable

Todo el audio se procesa localmente en tu computadora y nunca se envía a servidores externos. Los archivos de respaldo incremental (`sessions/`), registros de diagnóstico (`logs/`) y configuración local (`config.json`) permanecen exclusivamente en tu equipo. Al transcribir reuniones o clases, respetá siempre el consentimiento de los participantes y las políticas aplicables.
