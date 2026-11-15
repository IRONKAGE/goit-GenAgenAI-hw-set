# ==============================================================================
# 🚀 Generative & Agentic AI Enterprise Makefile (Zero-Config, Nov 15, 2026)
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Експорт змінних середовища (якщо файл .env існує)
# ------------------------------------------------------------------------------
ifneq (,$(wildcard ./.env))
	include .env
	export $(shell awk -F= '/^[a-zA-Z_]/ {print $$1}' .env)
endif

# ------------------------------------------------------------------------------
# 2. Системні змінні
# ------------------------------------------------------------------------------
PY_VER := 3.12
VENV := .venv
PYTHON_CMD := $(shell command -v python3 2>/dev/null || command -v python 2>/dev/null || echo "python3")
UV := $(PYTHON_CMD) -m uv
MIN_RAM_GB := 8

# ------------------------------------------------------------------------------
# 3. Кросплатформна підтримка ОС та збір інформації про залізо (OS, Arch, RAM)
# ------------------------------------------------------------------------------
ifeq ($(OS),Windows_NT)
	OPEN_CMD := start ""
	SYS_OS_NAME := Windows
	SYS_CPU_ARCH := $(PROCESSOR_ARCHITECTURE)
	SYS_RAM_GB := $(shell powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)")
	VENV_PY := $(VENV)/Scripts/python
else
	UNAME_S := $(shell uname -s)
	SYS_CPU_ARCH := $(shell uname -m)
	VENV_PY := $(VENV)/bin/python
	ifeq ($(UNAME_S),Linux)
		OPEN_CMD := xdg-open
		SYS_OS_NAME := $(shell grep '^PRETTY_NAME=' /etc/os-release | cut -d '"' -f 2 || echo "Linux")
		SYS_RAM_GB := $(shell awk '/MemTotal/ {printf "%.0f", $$2/1024/1024}' /proc/meminfo)
	endif
	ifeq ($(UNAME_S),Darwin)
		OPEN_CMD := open
		SYS_OS_NAME := $(shell sw_vers -productName) $(shell sw_vers -productVersion)
		SYS_RAM_GB := $(shell expr $$(sysctl -n hw.memsize) / 1073741824)
	endif
endif

# ------------------------------------------------------------------------------
# 4. Кольори для термінала
# ------------------------------------------------------------------------------
CYAN := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
PURPLE := \033[35m
RESET := \033[0m

.PHONY: help check-resources setup test export-all mlflow run-hw1 run-hw2 run-hw3 run-hw4 run-hw5 run-fp clean deep-clean

# Дефолтна ціль
help:
	@echo "$(CYAN)============================================================================================$(RESET)"
	@echo "$(GREEN)               🤖 Generative & Agentic AI Portfolio (goit-GenAgenAI-hw-set)$(RESET)"
	@echo "$(PURPLE)               🔮 ОС: $(SYS_OS_NAME) | Архітектура ЦП: $(SYS_CPU_ARCH) | ОЗП: $(SYS_RAM_GB) Гб$(RESET)"
	@echo "$(CYAN)============================================================================================$(RESET)"
	@echo "$(YELLOW)🛠  Команди інфраструктури та перевірки:$(RESET)"
	@echo "  $(GREEN)make setup$(RESET)        - 📦 Налаштувати середовище та залежності через uv sync"
	@echo "  $(GREEN)make test$(RESET)         - 🧪 Прогнати автотести Pytest по всіх 6 проектах (ДЗ 1–5 + ФП)"
	@echo "  $(GREEN)make export-all$(RESET)   - 🎓 Згенерувати всі артефакти для здачі в LMS (папка submissions/)"
	@echo "  $(GREEN)make mlflow$(RESET)       - 📈 Запустити MLflow Tracking Server на порту 5000"
	@echo "  $(GREEN)make clean$(RESET)        - 🧹 Очистити кеш (pycache, ruff, marimo, pytest)"
	@echo "  $(GREEN)make deep-clean$(RESET)   - 🧨 ПОВНЕ очищення: видалити середовище та артефакти"
	@echo ""
	@echo "$(YELLOW)🚀 Реактивні Дашборди Marimo 0.25.2 (Дослідження + Виконання):$(RESET)"
	@echo "  $(PURPLE)make run-hw1$(RESET)      - 🌍 ДЗ 1: Pre-training, Knowledge Cutoff, Alignment & Base vs Instruct"
	@echo "  $(PURPLE)make run-hw2$(RESET)      - 🎯 ДЗ 2: Контекст-інжиніринг, Тріаж відгуків & Валідація check()"
	@echo "  $(PURPLE)make run-hw3$(RESET)      - 📝 ДЗ 3: TODO-Агент з інструментами (LangChain v1 + InMemoryStore)"
	@echo "  $(PURPLE)make run-hw4$(RESET)      - 📈 ДЗ 4: RAG для фінансових звітів (Chroma + 4 чанки + RAGAS)"
	@echo "  $(PURPLE)make run-hw5$(RESET)      - 🛠️  ДЗ 5: Агент техпідтримки (LangGraph v1 + Незалежний Judge LLM)"
	@echo "  $(PURPLE)make run-fp$(RESET)       - 🌐 ФП: Агентна система збагачення даних з Excel (capitals & mountains)"
	@echo "$(CYAN)============================================================================================$(RESET)"

# ------------------------------------------------------------------------------
# 5. ІНІЦІАЛІЗАЦІЯ ІНФРАСТРУКТУРИ ТА ПЕРЕВІРКА ЗАЛІЗА
# ------------------------------------------------------------------------------
check-resources:
	@echo "$(CYAN)🔍 Аналіз апаратного забезпечення (RAM Validator)...$(RESET)"
	@echo "import platform, subprocess, sys" > .hw_check.py
	@echo "gb = 0" >> .hw_check.py
	@echo "try:" >> .hw_check.py
	@echo "    s = platform.system()" >> .hw_check.py
	@echo "    if s == 'Windows':" >> .hw_check.py
	@echo "        import ctypes" >> .hw_check.py
	@echo "        class M(ctypes.Structure): _fields_=[('l',ctypes.c_ulong),('m',ctypes.c_ulong),('t',ctypes.c_ulonglong)]" >> .hw_check.py
	@echo "        mem = M(); mem.l = ctypes.sizeof(M); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)); gb = mem.t / (1024**3)" >> .hw_check.py
	@echo "    elif s == 'Darwin':" >> .hw_check.py
	@echo "        gb = int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'])) / (1024**3)" >> .hw_check.py
	@echo "    else:" >> .hw_check.py
	@echo "        gb = int([l.split()[1] for l in open('/proc/meminfo') if 'MemTotal' in l][0]) / (1024**2)" >> .hw_check.py
	@echo "except Exception as e: print(f'Помилка детектора: {e}'); sys.exit(0)" >> .hw_check.py
	@echo "req = int('$(MIN_RAM_GB)')" >> .hw_check.py
	@echo "if gb < req: print(f'FAIL:{gb:.1f}')" >> .hw_check.py
	@echo "else: print(f'OK:{gb:.1f}')" >> .hw_check.py
	@HW_RESULT=$$(python3 .hw_check.py 2>/dev/null || python .hw_check.py 2>/dev/null); \
	rm -f .hw_check.py; \
	if echo "$$HW_RESULT" | grep -q "FAIL"; then \
		ACTUAL_RAM=$$(echo "$$HW_RESULT" | cut -d':' -f2); \
		echo "$(RED)❌ Мало ОЗП ($$ACTUAL_RAM ГБ)! Рекомендовано мінімум $(MIN_RAM_GB) ГБ.$(RESET)"; \
	elif echo "$$HW_RESULT" | grep -q "OK"; then \
		echo "$(GREEN)✅ Залізо схвалено (>= $(MIN_RAM_GB) ГБ ОЗП).$(RESET)"; \
	fi

setup: check-resources
	@echo "$(CYAN)🚀 Ініціалізація LLMOps інфраструктури...$(RESET)"
	@mkdir -p data/cache data/hw_fp_excel submissions models mlruns
	@$(PYTHON_CMD) -m pip install --upgrade pip --break-system-packages > /dev/null 2>&1 || true
	@$(PYTHON_CMD) -m pip install uv --break-system-packages > /dev/null 2>&1 || $(PYTHON_CMD) -m pip install --user uv > /dev/null 2>&1 || true
	@echo "$(YELLOW)⚡ Синхронізація залежностей (Python $(PY_VER)) через uv...$(RESET)"
	@$(UV) sync --all-extras --python $(PY_VER)
	@if [ ! -f .env ]; then \
		echo "$(YELLOW)🔐 Створюю конфіг .env із шаблону .env.example...$(RESET)"; \
		cp .env.example .env 2>/dev/null || touch .env; \
	fi
	@echo "$(GREEN)✅ Setup завершено! Введіть 'make test' або 'make export-all'.$(RESET)"

mlflow:
	@echo "$(CYAN)📈 Запуск MLflow сервера експериментів на http://127.0.0.1:5000$(RESET)"
	PYTHONPATH=. $(UV) run mlflow ui --host 127.0.0.1 --port 5000

# ------------------------------------------------------------------------------
# 6. ЗАПУСК РЕАКТИВНИХ ДАШБОРДІВ (MARIMO 0.25.2)
# ------------------------------------------------------------------------------
run-hw1:
	PYTHONPATH=. $(UV) run marimo edit notebooks_eda/hw_01_eda.py

run-hw2:
	PYTHONPATH=. $(UV) run marimo edit notebooks_eda/hw_02_eda.py

run-hw3:
	PYTHONPATH=. $(UV) run marimo edit notebooks_eda/hw_03_eda.py

run-hw4:
	PYTHONPATH=. $(UV) run marimo edit notebooks_eda/hw_04_eda.py

run-hw5:
	PYTHONPATH=. $(UV) run marimo edit notebooks_eda/hw_05_eda.py

run-fp:
	PYTHONPATH=. $(UV) run marimo edit notebooks_eda/fp_eda.py

# ------------------------------------------------------------------------------
# 7. ЕКСПОРТ АРТЕФАКТІВ ДЛЯ LMS ТА АВТОТЕСТУВАННЯ
# ------------------------------------------------------------------------------
export-all:
	@echo "$(CYAN)🎓 Експорт усіх ноутбуків (.html, .ipynb), звітів (.md) та таблиць (_enriched.xlsx)...$(RESET)"
	@mkdir -p submissions
	PYTHONPATH=. $(UV) run marimo export html notebooks_eda/hw_01_eda.py -o submissions/Oleh_Hatsenko_HW1.html -f
	PYTHONPATH=. $(UV) run marimo export html notebooks_eda/hw_02_eda.py -o submissions/Oleh_Hatsenko_HW2.html -f
	PYTHONPATH=. $(UV) run marimo export html notebooks_eda/hw_03_eda.py -o submissions/Oleh_Hatsenko_HW3.html -f
	PYTHONPATH=. $(UV) run marimo export html notebooks_eda/hw_04_eda.py -o submissions/Oleh_Hatsenko_HW4.html -f
	PYTHONPATH=. $(UV) run marimo export html notebooks_eda/hw_05_eda.py -o submissions/Oleh_Hatsenko_HW5.html -f
	PYTHONPATH=. $(UV) run marimo export html notebooks_eda/fp_eda.py -o submissions/Oleh_Hatsenko_FP.html -f
	@cp -f submissions/oleh_hatsenko_hw4.ipynb submissions/Oleh_Hatsenko_DZ7.ipynb 2>/dev/null || true
	@cp -f submissions/oleh_hatsenko_hw5.ipynb submissions/Oleh_Hatsenko_DZ9.ipynb 2>/dev/null || true
	@echo "$(GREEN)✅ Всі файли для здачі успішно згенеровано у директорію submissions/!$(RESET)"

test:
	@echo "$(CYAN)🧪 Запуск повного сьюту автотестів Pytest...$(RESET)"
	PYTHONPATH=. $(UV) run --all-extras python -m pytest -v

# ------------------------------------------------------------------------------
# 8. ОЧИЩЕННЯ КЕШІВ
# ------------------------------------------------------------------------------
clean:
	@echo "$(YELLOW)🧹 Очищення тимчасових файлів та кешу...$(RESET)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "__marimo__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(GREEN)✨ Системний кеш очищено.$(RESET)"

deep-clean: clean
	@echo "$(RED)🧨 ПОВНЕ ОЧИЩЕННЯ: Видалення середовища .venv та ML-артефактів...$(RESET)"
	rm -rf $(VENV) mlruns/ .uv_cache/
	@echo "$(GREEN)✅ Репозиторій повернуто до Zero-State!$(RESET)"

%:
	@:
