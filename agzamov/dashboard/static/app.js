/**
 * Agzamov Test — Live Dashboard Client
 *
 * Connects to WebSocket, updates cm-chessboard, eval bar,
 * move list, and commentary in real time.
 */

import {Chessboard, FEN, COLOR} from "./cm-chessboard/src/Chessboard.js"

// --- State ---
let board = null
let ws = null
let seriesScore = { w: 0, d: 0, l: 0 }
let totalErrors = 0
let moveRows = []
let currentMoveNum = 0
let whiteTotalMs = 0    // accumulated thinking time
let blackTotalMs = 0

// --- Init board ---
function initBoard() {
    board = new Chessboard(document.getElementById("board"), {
        position: FEN.start,
        orientation: COLOR.white,
        assetsUrl: "/static/cm-chessboard/assets/",
        style: {
            cssClass: "default",
            showCoordinates: true,
            borderType: "none",
            pieces: {
                file: "pieces/staunty.svg",
                tileSize: 40
            },
            animationDuration: 350
        }
    })
}

// --- Eval bar ---
function updateEvalBar(cp) {
    if (cp == null) return
    const clamped = Math.max(-600, Math.min(600, cp / 100))
    const pct = 50 + (clamped / 6) * 50
    document.getElementById("eval-bar-white").style.height = pct + "%"

    let label
    if (Math.abs(cp) >= 9000) {
        const moves = Math.ceil((10000 - Math.abs(cp)))
        label = cp > 0 ? "M" + moves : "-M" + moves
    } else {
        const val = (cp / 100).toFixed(1)
        label = cp >= 0 ? "+" + val : val
    }
    document.getElementById("eval-label").textContent = label
}

// --- Clocks ---
function formatClock(ms) {
    const totalSec = ms / 1000
    if (totalSec < 60) return totalSec.toFixed(1) + "s"
    const min = Math.floor(totalSec / 60)
    const sec = (totalSec % 60).toFixed(0).padStart(2, "0")
    return `${min}:${sec}`
}

function updateClocks() {
    document.querySelector("#player-bottom .player-clock").textContent = formatClock(whiteTotalMs)
    document.querySelector("#player-top .player-clock").textContent = formatClock(blackTotalMs)
}

// --- Move list ---
function addMove(data) {
    const ply = data.ply
    const moveNum = Math.ceil(ply / 2)

    if (data.side === "white") {
        currentMoveNum = moveNum
        moveRows.push({
            num: moveNum,
            white: data.move_uci,
            whiteTime: data.wall_ms,
            whiteTag: data.move_tag || "",
            black: "",
            blackTime: 0,
            blackTag: ""
        })
    } else {
        if (moveRows.length > 0 && moveRows[moveRows.length - 1].num === moveNum) {
            const row = moveRows[moveRows.length - 1]
            row.black = data.move_uci
            row.blackTime = data.wall_ms
            row.blackTag = data.move_tag || ""
        } else {
            moveRows.push({
                num: moveNum,
                white: "...",
                whiteTime: 0,
                whiteTag: "",
                black: data.move_uci,
                blackTime: data.wall_ms,
                blackTag: data.move_tag || ""
            })
        }
    }

    renderMoves()
}

function tagClass(tag) {
    if (!tag) return ""
    if (tag.includes("BLUNDER")) return "tag-blunder"
    if (tag.includes("MISTAKE")) return "tag-mistake"
    if (tag.includes("INACCURACY")) return "tag-inaccuracy"
    if (tag.includes("GREAT")) return "tag-great"
    return ""
}

function formatTime(ms) {
    if (!ms) return ""
    return (ms / 1000).toFixed(1) + "s"
}

function renderMoves() {
    const el = document.getElementById("moves-list")
    const visible = moveRows.slice(-20)
    el.innerHTML = visible.map(r => {
        const wTag = r.whiteTag ? `<span class="move-tag ${tagClass(r.whiteTag)}">${r.whiteTag}</span>` : ""
        const bTag = r.blackTag ? `<span class="move-tag ${tagClass(r.blackTag)}">${r.blackTag}</span>` : ""
        return `<div class="move-row">
            <span class="move-num">${r.num}.</span>
            <span class="move-white">${r.white}</span>
            <span class="move-time">${formatTime(r.whiteTime)}</span>
            ${wTag}
            <span class="move-black">${r.black}</span>
            <span class="move-time">${formatTime(r.blackTime)}</span>
            ${bTag}
        </div>`
    }).join("")
    el.scrollTop = el.scrollHeight
}

// --- Commentary ---
function addCommentary(text) {
    if (!text) return
    const el = document.getElementById("commentary")
    const entry = document.createElement("div")
    entry.className = "commentary-entry"
    entry.textContent = text
    el.appendChild(entry)
    // Keep last 5
    while (el.children.length > 5) {
        el.removeChild(el.firstChild)
    }
    // Force scroll to bottom
    requestAnimationFrame(() => {
        el.scrollTop = el.scrollHeight
    })
}

// --- Event handlers ---
function onRunInfo(data) {
    document.getElementById("phase-label").textContent = data.model_label
    document.getElementById("matchup").textContent = data.run_name
    document.getElementById("game-counter").textContent =
        `temp=${data.temperature} | memory=${data.memory_type}`
}

function onGameStart(data) {
    // Reset board
    board.setPosition(data.starting_fen, false)

    // Reset moves & clocks
    moveRows = []
    currentMoveNum = 0
    whiteTotalMs = 0
    blackTotalMs = 0
    document.getElementById("moves-list").innerHTML = ""
    document.getElementById("commentary").innerHTML = ""

    // Update UI
    document.getElementById("matchup").innerHTML =
        `<span style="color:#f0d9b5">\u2588</span> ${data.white_name}  vs  ${data.black_name} <span style="color:#555">\u2588</span>`
    document.getElementById("game-counter").textContent =
        `${data.game_id}  |  Chess960 #${data.position_id}`
    document.getElementById("phase-label").textContent = `Phase ${data.phase}`

    // Player labels
    document.querySelector("#player-bottom .player-name").textContent = data.white_name
    document.querySelector("#player-top .player-name").textContent = data.black_name
    updateClocks()

    // Reset eval bar
    updateEvalBar(0)
}

function onMove(data) {
    // Animate board
    board.setPosition(data.fen, true)

    // Update eval bar
    updateEvalBar(data.eval_cp)

    // Accumulate clocks
    if (data.side === "white") {
        whiteTotalMs += data.wall_ms
    } else {
        blackTotalMs += data.wall_ms
    }
    updateClocks()

    // Add to move list
    addMove(data)

    // Commentary
    if (data.comment) {
        addCommentary(data.comment)
    }

    // Move info bar
    document.getElementById("move-time").textContent =
        `Last: ${formatTime(data.wall_ms)}  (${data.agent_name})`

    // Track errors
    if (data.was_error) {
        totalErrors++
        document.getElementById("errors-count").textContent = `Errors: ${totalErrors}`
    }
}

function onGameEnd(data) {
    if (data.result === "1-0") seriesScore.w++
    else if (data.result === "0-1") seriesScore.l++
    else seriesScore.d++

    document.getElementById("series-score").textContent =
        `W ${seriesScore.w} — D ${seriesScore.d} — L ${seriesScore.l}`

    const reason = data.result_reason || ""
    addCommentary(`Game over: ${data.result} (${reason}) — ${data.total_moves} moves`)
}

function onPhaseStart(data) {
    document.getElementById("phase-label").textContent =
        `Phase ${data.phase}: ${data.phase_name || ""}`
    seriesScore = { w: 0, d: 0, l: 0 }
    totalErrors = 0
    document.getElementById("series-score").textContent = "W 0 — D 0 — L 0"
    document.getElementById("errors-count").textContent = "Errors: 0"
}

// --- WebSocket ---
function connect() {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:"
    ws = new WebSocket(`${protocol}//${location.host}/ws`)

    ws.onopen = () => {
        document.getElementById("status-indicator").textContent = "LIVE"
        document.getElementById("status-indicator").className = "connected"
    }

    ws.onclose = () => {
        document.getElementById("status-indicator").textContent = "DISCONNECTED"
        document.getElementById("status-indicator").className = ""
        setTimeout(connect, 2000)
    }

    ws.onerror = () => { ws.close() }

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data)
        switch (data.type) {
            case "run_info":
                onRunInfo(data)
                break
            case "game_start":
                onGameStart(data)
                break
            case "move":
                onMove(data)
                break
            case "game_end":
                onGameEnd(data)
                break
            case "phase_start":
                onPhaseStart(data)
                break
        }
    }
}

// --- Boot ---
initBoard()
connect()
