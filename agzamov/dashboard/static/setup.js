/**
 * Run Configuration Modal — form logic, validation, HTTP submission.
 */

let providers = []
let runActive = false

// --- Init ---
async function init() {
    try {
        const [defResp, provResp] = await Promise.all([
            fetch("/api/config/defaults"),
            fetch("/api/config/providers"),
        ])
        if (defResp.ok) {
            const defaults = await defResp.json()
            populateFormDefaults(defaults)
        }
        if (provResp.ok) {
            providers = await provResp.json()
        }
    } catch (e) {
        // API not available (CLI-launched dashboard) — modal still works with HTML defaults
    }

    populateModelHints()
    bindEvents()
    checkRunStatus()
}

function populateModelHints() {
    const dl = document.getElementById("model-hints")
    if (!dl) return

    // Build hints from providers API — available keys first
    if (providers.length > 0) {
        const available = providers.filter(p => p.available).flatMap(p => p.models || [])
        const unavailable = providers.filter(p => !p.available).flatMap(p => p.models || [])
        const hints = [...available, ...unavailable]
        dl.innerHTML = hints.map(h => `<option value="${h}">`).join("")
        return
    }

    // Fallback if API not available
    const fallback = [
        "claude-sonnet-4-6", "claude-opus-4-6",
        "gemini-2.5-pro", "gpt-4o", "deepseek-chat",
        "glm-4-plus", "qwen-max",
    ]
    dl.innerHTML = fallback.map(h => `<option value="${h}">`).join("")
}

function populateFormDefaults(d) {
    const set = (id, val) => {
        const el = document.getElementById(id)
        if (el && val !== undefined && val !== null) el.value = val
    }
    const setChecked = (id, val) => {
        const el = document.getElementById(id)
        if (el) el.checked = !!val
    }

    // Model
    set("cfg-model-name", d.model?.name)
    set("cfg-temperature", d.model?.temperature)
    set("cfg-max-tokens", d.model?.max_tokens)
    setChecked("cfg-thinking", d.model?.thinking)
    set("cfg-thinking-budget", d.model?.thinking_budget)
    // Chess
    set("cfg-max-moves", d.chess?.max_moves_per_game)
    // Memory
    set("cfg-memory-type", d.memory?.type)
    set("cfg-memory-endpoint", d.memory?.endpoint)
    set("cfg-memory-tokens", d.memory?.max_context_tokens)
    // Stockfish
    set("cfg-sf-path", d.stockfish?.path)
    set("cfg-sf-depth", d.stockfish?.analysis_depth)
    set("cfg-sf-threads", d.stockfish?.threads)
    set("cfg-sf-hash", d.stockfish?.hash_mb)
    // Tree search
    set("cfg-search-mode", d.tree_search?.mode)
    set("cfg-candidates", d.tree_search?.num_candidates)
    set("cfg-eval-depth", d.tree_search?.eval_depth)
    // Sanity
    set("cfg-sanity-games", d.sanity_check?.chess_games)
    set("cfg-sanity-pass", d.sanity_check?.chess_pass_threshold)
    set("cfg-sanity-error", d.sanity_check?.chess_error_threshold)
    // Budget
    set("cfg-budget-max", d.budget?.max_api_cost_usd)
    set("cfg-budget-warn", d.budget?.warn_at_pct)
    // Stats
    set("cfg-stats-sig", d.stats?.significance_threshold)
    set("cfg-stats-bootstrap", d.stats?.bootstrap_samples)
    set("cfg-stats-elo-k", d.stats?.elo_k_factor)
    // Output
    set("cfg-output-dir", d.output?.results_dir)
    setChecked("cfg-save-history", d.output?.save_game_history !== false)
    setChecked("cfg-save-memory", d.output?.save_memory_dump !== false)
    setChecked("cfg-save-sf", d.output?.save_stockfish_analysis !== false)
}

// --- Events ---
function bindEvents() {
    // Open/close
    document.getElementById("btn-new-run")?.addEventListener("click", openModal)
    document.getElementById("modal-close")?.addEventListener("click", closeModal)
    document.getElementById("btn-cancel-modal")?.addEventListener("click", closeModal)
    document.getElementById("run-modal")?.addEventListener("click", (e) => {
        if (e.target.id === "run-modal") closeModal()
    })

    // Start/stop
    document.getElementById("btn-start-run")?.addEventListener("click", submitRun)
    document.getElementById("btn-stop-run")?.addEventListener("click", stopRun)

    // Tabs
    document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"))
            document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"))
            btn.classList.add("active")
            document.getElementById(btn.dataset.tab)?.classList.add("active")
        })
    })

    // Match type toggle
    document.querySelectorAll('input[name="match_type"]').forEach(r => {
        r.addEventListener("change", () => {
            const type = document.querySelector('input[name="match_type"]:checked')?.value
            const phases = document.getElementById("phases-group")
            const opp = document.getElementById("opponent-section")
            if (phases) phases.style.display = type === "solo" ? "" : "none"
            if (opp) opp.style.display = type === "vs" ? "" : "none"
        })
    })

    // Thinking toggles
    document.getElementById("cfg-thinking")?.addEventListener("change", (e) => {
        document.getElementById("thinking-budget-group").style.display = e.target.checked ? "" : "none"
    })
    document.getElementById("cfg-opp-thinking")?.addEventListener("change", (e) => {
        document.getElementById("opp-thinking-budget-group").style.display = e.target.checked ? "" : "none"
    })

    // Model name → key indicator
    document.getElementById("cfg-model-name")?.addEventListener("input", () => updateKeyIndicator("cfg-model-name", "model-key-status"))
    document.getElementById("cfg-opp-model")?.addEventListener("input", () => updateKeyIndicator("cfg-opp-model", "opp-key-status"))

    // Search mode toggle
    document.getElementById("cfg-search-mode")?.addEventListener("change", (e) => {
        document.getElementById("tree-search-opts").style.display =
            (e.target.value === "tree") ? "" : "none"
    })

    // Escape key
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") closeModal()
    })
}

function updateKeyIndicator(inputId, statusId) {
    const name = (document.getElementById(inputId)?.value || "").toLowerCase()
    const el = document.getElementById(statusId)
    if (!el || !name) { if (el) el.textContent = ""; return }
    const prov = providers.find(p => name.startsWith(p.prefix))
    if (prov) {
        el.textContent = prov.available ? "KEY OK" : "KEY MISSING"
        el.className = "key-indicator " + (prov.available ? "key-ok" : "key-missing")
    } else {
        el.textContent = ""
    }
}

// --- Modal ---
function openModal() {
    if (runActive) return
    document.getElementById("run-modal").style.display = "flex"
    document.getElementById("modal-warnings").style.display = "none"
    updateKeyIndicator("cfg-model-name", "model-key-status")
}

function closeModal() {
    document.getElementById("run-modal").style.display = "none"
}

// --- Build config payload ---
function val(id) { return document.getElementById(id)?.value || "" }
function num(id) { return parseFloat(document.getElementById(id)?.value) || 0 }
function int(id) { return parseInt(document.getElementById(id)?.value) || 0 }
function checked(id) { return document.getElementById(id)?.checked || false }

function buildPayload() {
    const matchType = document.querySelector('input[name="match_type"]:checked')?.value || "solo"
    const nGames = int("cfg-n-games") || 10

    let phases = [0]
    if (matchType === "solo") {
        phases = [...document.querySelectorAll('input[name="phase"]:checked')]
            .map(cb => parseInt(cb.value))
        if (phases.length === 0) phases = [0]
    }

    const config = {
        name: val("cfg-run-name"),
        phases,
        model: {
            name: val("cfg-model-name"),
            temperature: num("cfg-temperature"),
            max_tokens: int("cfg-max-tokens"),
            thinking: checked("cfg-thinking"),
            thinking_budget: int("cfg-thinking-budget"),
        },
        memory: {
            type: val("cfg-memory-type"),
            endpoint: val("cfg-memory-endpoint"),
            max_context_tokens: int("cfg-memory-tokens"),
            consolidation_trigger: "every_game",
        },
        chess: {
            variant: "chess960",
            games_phase_1: nGames,
            games_phase_2: nGames,
            games_phase_3: nGames,
            max_moves_per_game: int("cfg-max-moves"),
            time_tracking: true,
        },
        sanity_check: {
            chess_games: int("cfg-sanity-games"),
            chess_pass_threshold: num("cfg-sanity-pass"),
            chess_error_threshold: num("cfg-sanity-error"),
        },
        stockfish: {
            path: val("cfg-sf-path"),
            analysis_depth: int("cfg-sf-depth"),
            chess960_mode: true,
            threads: int("cfg-sf-threads"),
            hash_mb: int("cfg-sf-hash"),
        },
        stats: {
            significance_threshold: num("cfg-stats-sig"),
            bootstrap_samples: int("cfg-stats-bootstrap"),
            elo_k_factor: int("cfg-stats-elo-k"),
            tau_window_size: 20,
            tau_threshold: 0.95,
        },
        output: {
            results_dir: val("cfg-output-dir") || "./results",
            save_game_history: checked("cfg-save-history"),
            save_memory_dump: checked("cfg-save-memory"),
            save_stockfish_analysis: checked("cfg-save-sf"),
            report_format: "markdown",
        },
        budget: {
            max_api_cost_usd: num("cfg-budget-max"),
            cost_tracking: true,
            warn_at_pct: int("cfg-budget-warn"),
        },
        tree_search: {
            mode: val("cfg-search-mode"),
            num_candidates: int("cfg-candidates"),
            eval_depth: int("cfg-eval-depth"),
            sf_play_depth: int("cfg-sf-depth"),
        },
        synthetic_patterns: { enabled: false, chess_constraints: [], poker_constraints: [] },
    }

    const payload = { match_type: matchType, n_games: nGames, config }

    if (matchType === "vs") {
        payload.opponent_model = {
            name: val("cfg-opp-model"),
            temperature: num("cfg-opp-temperature"),
            max_tokens: int("cfg-opp-max-tokens"),
            thinking: checked("cfg-opp-thinking"),
            thinking_budget: int("cfg-opp-thinking-budget"),
        }
    }

    return payload
}

// --- Submit ---
async function submitRun() {
    const payload = buildPayload()

    if (!payload.config.model.name) {
        showWarning("Model name is required.")
        return
    }

    const btn = document.getElementById("btn-start-run")
    btn.disabled = true
    btn.textContent = "Starting..."

    try {
        const resp = await fetch("/api/run", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        })
        const result = await resp.json()

        if (!resp.ok) {
            showWarning(result.error || "Failed to start run")
            btn.disabled = false
            btn.textContent = "Start Run"
            return
        }

        if (result.warnings?.length) {
            showWarning(result.warnings.join("\n"), "warning")
        }

        closeModal()
        setRunActive(true)
    } catch (e) {
        showWarning("Network error: " + e.message)
        btn.disabled = false
        btn.textContent = "Start Run"
    }
}

async function stopRun() {
    try {
        await fetch("/api/run/stop", { method: "POST" })
        document.getElementById("btn-stop-run").textContent = "STOPPING..."
    } catch (e) {
        console.error("Stop failed:", e)
    }
}

function showWarning(text, level = "error") {
    const el = document.getElementById("modal-warnings")
    el.style.display = "block"
    el.className = "modal-warnings " + level
    el.textContent = text
}

function setRunActive(active) {
    runActive = active
    const btn = document.getElementById("btn-new-run")
    const stop = document.getElementById("btn-stop-run")
    const start = document.getElementById("btn-start-run")
    if (btn) btn.disabled = active
    if (stop) stop.style.display = active ? "" : "none"
    if (!active) {
        if (stop) stop.textContent = "STOP RUN"
        if (start) { start.disabled = false; start.textContent = "Start Run" }
    }
}

async function checkRunStatus() {
    try {
        const resp = await fetch("/api/run/status")
        if (resp.ok) {
            const data = await resp.json()
            setRunActive(data.status === "running")
        }
    } catch (e) { /* API not available */ }
}

// --- Called by app.js for run lifecycle events ---
export function onRunEvent(data) {
    if (data.type === "game_start" || data.type === "run_info") {
        setRunActive(true)
    } else if (data.type === "run_complete" || data.type === "run_cancelled" || data.type === "run_error") {
        setRunActive(false)
        if (data.type === "run_error" && data.error) {
            console.error("Run error:", data.error)
        }
    }
}

// Boot
init()
