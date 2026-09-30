// ==UserScript==
// @name         CodinGame log copier
// @namespace    https://www.codingame.com/
// @version      1.7
// @description  Adds buttons next to the CodinGame console to copy its raw log or format it into numbered months.
// @match        https://*.codingame.com/*
// @all-frames   true
// @grant        GM_setClipboard
// @grant        GM_registerMenuCommand
// @run-at       document-idle
// ==/UserScript==

(function () {
    "use strict";

    const ERROR = "Standard Error Stream:", OUTPUT = "Standard Output Stream:", SUMMARY = "Game Summary:";
    const MARKERS = [ERROR, OUTPUT, SUMMARY];
    const BUTTON_ID = "cg-log-reformatter";
    const LABELS = ["Copy log", "Copy formatted log"];

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

    function copy(button, formatted) {
        const container = logContainer();
        if (!container) return;
        GM_setClipboard(formatted ? reformat(container.innerText) : container.innerText);
        if (!button) return;
        button.textContent = "Copied!";
        setTimeout(() => button.textContent = LABELS[Number(formatted)], 1500);
    }

    // Numbers each stream block in text as a month, removes counter pairs and separates the final game information.
    function reformat(text) {
        const blocks = text.match(/^Standard Error Stream:[\s\S]*?(?=^Standard Error Stream:|(?![\s\S]))/gm);
        return blocks.map((block, index) => {
            let content = block.replace(/\r?\n[ \t]*[+-]?\d+[ \t]*\r?\n[ \t]*[+-]?\d+[ \t]*(?:\r?\n[ \t]*)*$/, "");
            const [streams, information] = content.split(/\r?\n(?=Game information:)/);
            content = information === undefined ? streams : streams.replace(/\r?\n[ \t]*[+-]?\d+[ \t]*\r?\n[ \t]*[+-]?\d+[ \t]*$/, "");
            return `Month ${index + 1}:\n${content}\n\n${information === undefined ? "" : information.trimEnd()}`;
        }).join("");
    }

    function inject() {
        if (document.getElementById(BUTTON_ID)) return;
        const container = logContainer();
        if (!container || container === document.body) return;

        for (const formatted of [true, false]) {
            const button = document.createElement("button");
            button.id = formatted ? `${BUTTON_ID}-formatted` : BUTTON_ID;
            button.textContent = LABELS[Number(formatted)];
            button.style.cssText = "position:relative;z-index:9999;margin:4px;padding:4px 10px;font:12px sans-serif;cursor:pointer";
            button.addEventListener("click", event => {
                event.preventDefault();
                event.stopPropagation();
                copy(button, formatted);
            });
            container.parentElement.insertBefore(button, container.nextSibling);
        }
    }

    let pending = null;
    new MutationObserver(() => {
        clearTimeout(pending);
        pending = setTimeout(inject, 500);
    }).observe(document.body, {childList: true, subtree: true});
    for (const formatted of [false, true]) {
        GM_registerMenuCommand(LABELS[Number(formatted)], () => copy(null, formatted));
    }
    inject();
})();
