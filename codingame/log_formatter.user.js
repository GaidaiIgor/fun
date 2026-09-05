// ==UserScript==
// @name         CodinGame log reformatter
// @namespace    https://www.codingame.com/
// @version      1.3
// @description  Adds a button next to the CodinGame console that copies its log with every turn reordered to start with the own bot's streams.
// @match        https://*.codingame.com/*
// @all-frames   true
// @grant        GM_setClipboard
// @grant        GM_registerMenuCommand
// @run-at       document-idle
// ==/UserScript==

(function () {
    "use strict";

    const BOT_NAME = "IGPro";
    const ERROR = "Standard Error Stream:", OUTPUT = "Standard Output Stream:", SUMMARY = "Game Summary:";
    const MARKERS = [ERROR, OUTPUT, SUMMARY];
    const BUTTON_ID = "cg-log-reformatter";
    const LABEL = "Copy formatted log";

    // Groups the console text into turns and prints each of them as a "Turn X/Y:" header followed by the own error
    // stream, the own output stream, the opponent output stream and the game summary, whichever player the bot is.
    function reformat(text) {
        const blocks = [];
        for (const raw of text.split("\n")) {
            const line = raw.trim();
            if (MARKERS.includes(line)) blocks.push([line, []]);
            else if (blocks.length && line) blocks[blocks.length - 1][1].push(line);
        }

        const summaries = [];
        for (const [marker, content] of blocks) if (marker === SUMMARY) summaries.push(...content);
        const opponent = opponentName(summaries);

        // The bot's error stream always precedes its own output stream, so its position in the first turn tells which player the bot is.
        const botIsFirst = blocks.length > 0 && blocks[0][0] === ERROR;
        const turns = [];
        let turn = null;
        for (const [marker, content] of blocks) {
            const full = marker === ERROR ? turn && turn.errors : marker === OUTPUT ? turn && turn.outputs.length === 2 : turn && turn.summary;
            if (!turn || full) turns.push(turn = {errors: null, outputs: [], summary: null});
            if (marker === ERROR) turn.errors = content;
            else if (marker === OUTPUT) turn.outputs.push(content);
            else turn.summary = content;
        }

        const lines = [];
        turns.forEach((turn, index) => {
            // Player 1's output block carries the turn number and the turn total; without them the turns are numbered by their order.
            const first = turn.outputs[0] || [];
            const numbered = first.length > 2 && /^\d+$/.test(first[first.length - 1]) && /^\d+$/.test(first[first.length - 2]);
            const [number, total] = numbered ? first.slice(-2).map(Number) : [index + 1, turns.length];
            if (numbered) turn.outputs[0] = first.slice(0, -2);
            const ordered = botIsFirst ? turn.outputs : turn.outputs.slice().reverse();
            lines.push(`Turn ${number}/${total}:`, `${BOT_NAME} error:`, ...turn.errors || [], `${BOT_NAME} output:`, ...ordered[0] || []);
            lines.push(`${opponent} output:`, ...ordered[1] || []);
            if (turn.summary) lines.push(SUMMARY, ...turn.summary);
            lines.push("");
        });
        return lines.join("\n");
    }

    // Extracts the opponent name from the game summary lines, whose leading player name may hold spaces. The name grows
    // word by word for as long as the lines it starts agree on the next word and that word is not one the own bot's
    // lines use right after its own name.
    function opponentName(lines) {
        const verbs = new Set(), others = [], counts = {};
        for (const line of lines) {
            if (line.startsWith(BOT_NAME + " ")) verbs.add(line.split(/\s+/)[1]);
            else others.push(line.split(/\s+/));
        }
        for (const words of others) counts[words[0]] = (counts[words[0]] || 0) + 1;

        const ranked = Object.entries(counts).sort((a, b) => b[1] - a[1]);
        if (!ranked.length) return "Opponent";
        const name = [ranked[0][0]];
        for (;;) {
            const rest = others.filter(words => words.length > name.length && name.every((word, index) => words[index] === word));
            const following = new Set(rest.map(words => words[name.length]));
            if (following.size !== 1) return name.join(" ");
            const [word] = following;
            if (verbs.has(word)) return name.join(" ");
            name.push(word);
        }
    }

    // Returns the deepest element that still holds every stream marker of the page, so the lookup depends on the log
    // text only, not on CodinGame class names. Counting the markers keeps the descent from stopping on a single turn.
    function logContainer() {
        const pattern = new RegExp(MARKERS.join("|"), "g");
        const count = element => (element.textContent.replace(/\u00A0/g, " ").match(pattern) || []).length;
        let container = document.body;
        if (!container || count(container) < 2) return null;
        for (let child = [...container.children].find(c => count(c) === count(container)); child; child = [...container.children].find(c => count(c) === count(container))) container = child;
        return container;
    }

    function copy(button) {
        const container = logContainer();
        if (!container) return;
        GM_setClipboard(reformat(container.innerText.replace(/\u00A0/g, " ")));
        if (!button) return;
        button.textContent = "Copied!";
        setTimeout(() => button.textContent = LABEL, 1500);
    }

    function inject() {
        if (document.getElementById(BUTTON_ID)) return;
        const container = logContainer();
        if (!container || container === document.body) return;

        const button = document.createElement("button");
        button.id = BUTTON_ID;
        button.textContent = LABEL;
        button.style.cssText = "position:relative;z-index:9999;margin:4px;padding:4px 10px;font:12px sans-serif;cursor:pointer";
        button.addEventListener("click", event => {
            event.preventDefault();
            event.stopPropagation();
            copy(button);
        });
        container.parentElement.insertBefore(button, container.nextSibling);
    }

    let pending = null;
    new MutationObserver(() => {
        clearTimeout(pending);
        pending = setTimeout(inject, 500);
    }).observe(document.body, {childList: true, subtree: true});
    GM_registerMenuCommand(LABEL, () => copy(null));
    inject();
})();
