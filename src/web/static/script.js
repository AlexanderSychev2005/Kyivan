const dict = {
    ru: {
        desc_title: "Исследуйте тексты",
        header_subtitle: "ДРЕВНЕРУССКАЯ ЭПИГРАФИКА",
        desc_text: "Введите древнерусский текст с пропусками, чтобы восстановить утерянные символы и определить диалект и время написания.<br><br>Используйте знак вопроса (<strong>?</strong>) для каждого символа, который модель должна восстановить. Используйте символ решетки (<strong>#</strong>) для восстановления текста неизвестной длины. Дефис (<strong>-</strong>) отмечает символы, заведомо не подлежащие восстановлению (модель их не трогает, как и реальные нечитаемые места в источнике).",
        mode_label: "Режим анализа",
        mode_full: "Восстановление + атрибуция",
        mode_attribution: "Только атрибуция",
        keyboard_label: "Древнерусская клавиатура",
        btn_analyze: "Анализировать текст",
        res_title: "Результаты анализа",
        attn_legend: "Значимость символов:",
        loading: "Анализируем текст...",
        temp_label: "Температура (креативность)",
        temp_formulaic: "Формульная",
        temp_creative: "Креативная",
        mode_restore: "Значимость: Реставрация",
        mode_date: "Значимость: Датировка",
        mode_region: "Значимость: Диалект",
        region_label: "Диалект / Регион",
        date_label: "Датировка",
        res_empty: "Здесь появятся результаты восстановления и атрибутики текста.",
        unk_label: "Прогноз длины лакуны:",
        unk_multi: "> 1 символа",
        unk_single: "1 символ",
        similar_label: "Похожие документы",
        similar_empty: "Похожие документы не найдены.",
        similar_hint: "Ближайшие по эмбеддингу документы во всём корпусе (train/eval/test). Клик — полный текст.",
        restoration_label: "Реставрация",
    },
    en: {
        desc_title: "Explore Texts",
        header_subtitle: "OLD EAST SLAVIC EPIGRAPHY",
        desc_text: "Enter your Old East Slavic text below to restore missing characters, and attribute the text to its original dialect and time of writing.<br><br>Use a question mark (<strong>?</strong>) for each character you want the model to predict. Use a single hash (<strong>#</strong>) to predict a gap of unknown length. A dash (<strong>-</strong>) marks characters known to be unrecoverable (the model leaves these untouched, matching a real editor's own unresolved mark).",
        mode_label: "Analysis mode",
        mode_full: "Restoration + attribution",
        mode_attribution: "Attribution only",
        keyboard_label: "Ancient Cyrillic Keyboard",
        btn_analyze: "Analyze Text",
        res_title: "Analysis Results",
        attn_legend: "Character Saliency:",
        loading: "Analyzing text...",
        temp_label: "Temperature (Creativity)",
        temp_formulaic: "Formulaic",
        temp_creative: "Creative",
        mode_restore: "Saliency: Restoration",
        mode_date: "Saliency: Dating",
        mode_region: "Saliency: Dialect",
        region_label: "Dialect / Region",
        date_label: "Dating",
        res_empty: "Restoration and attribution results will appear here.",
        unk_label: "Gap size prediction:",
        unk_multi: "> 1 character",
        unk_single: "1 character",
        similar_label: "Similar Documents",
        similar_empty: "No similar documents found.",
        similar_hint: "Nearest neighbors by document embedding, across the whole corpus (train/eval/test). Click for the full text.",
        restoration_label: "Restoration",
    }
};

let currentLang = 'ru';

function changeLanguage(lang) {
    currentLang = lang;
    const d = dict[lang];
    document.querySelectorAll('[data-i18n]').forEach(el => {
        const k = el.getAttribute('data-i18n');
        if (d[k] !== undefined) el.innerHTML = d[k];
    });
}

function currentMode() {
    const el = document.querySelector('input[name="analysis-mode"]:checked');
    return el ? el.value : 'full';
}

// Temperature/iterative only affect restoration -- hide them once
// attribution-only is selected rather than leaving controls that do nothing.
function onModeChange() {
    document.getElementById('restoration-controls').classList.toggle('hidden', currentMode() === 'attribution');
}

const ancientChars = ['+','·',':','҃','ѫ','ѭ','ѧ','ѩ','ꙗ','є','ѥ','ѣ','ѹ','ꙋ','ѕ','ꙁ','ꙩ','ѡ','ѿ','ѯ','ӏ','і','ï','ѳ'];

// Initialize keyboard
const kb = document.getElementById('virtualKeyboard');
if (kb) {
    ancientChars.forEach(ch => {
        const btn = document.createElement('button');
        btn.type = 'button'; 
        btn.className = 'kbd-btn'; 
        btn.innerText = ch;
        btn.onclick = () => insertChar(ch); 
        kb.appendChild(btn);
    });
}

function insertChar(char) {
    const ta = document.getElementById('input-text');
    if (!ta) return;
    const s = ta.selectionStart, e = ta.selectionEnd;
    ta.value = ta.value.slice(0,s) + char + ta.value.slice(e);
    ta.selectionStart = ta.selectionEnd = s + char.length;
    ta.focus(); 
    updateCounter();
}

const inputText = document.getElementById('input-text');
const charCount = document.getElementById('charCount');

if (inputText && charCount) {
    inputText.addEventListener('input', updateCounter);
}

function updateCounter() {
    charCount.textContent = `${inputText.value.length} / 1000`;
}

// State
let currentResponse = null;
let activeSaliencyMode = 'restore'; // 'restore', 'date', or 'region'
let activeRestorationIndex = null;
let regionChartInstance = null;

// The color palette
const colors = {
    cinnabar: '#D13426',
    limetree: '#D4A373',
    charcoal: '#2C2825',
    gold: '#DAA520',
    chainmail: '#A3A3A3',
    birch: '#F9F4E8'
};

async function analyzeText() {
    const text = inputText.value.trim();
    if (!text) return;

    // UI Updates
    document.getElementById('empty-state').classList.add('hidden');
    document.getElementById('results-container').classList.add('hidden');
    document.getElementById('error-message').classList.add('hidden');
    document.getElementById('loading-spinner').classList.remove('hidden');
    document.getElementById('analyze-btn').disabled = true;

    try {
        const temp = parseFloat(document.getElementById('temp-slider').value) || 1.0;
        const isIterative = document.getElementById('iterative-toggle') ? document.getElementById('iterative-toggle').checked : false;
        const mode = currentMode();

        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, temperature: temp, iterative: isIterative, mode })
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();
        currentResponse = data;
        
        renderResults();
        
        document.getElementById('loading-spinner').classList.add('hidden');
        document.getElementById('results-container').classList.remove('hidden');
    } catch (error) {
        console.error(error);
        const errEl = document.getElementById('error-message');
        errEl.textContent = "Ошибка при анализе текста: " + error.message;
        errEl.classList.remove('hidden');
        document.getElementById('loading-spinner').classList.add('hidden');
    } finally {
        document.getElementById('analyze-btn').disabled = false;
    }
}

function switchSaliencyMode(mode) {
    activeSaliencyMode = mode;

    const btns = { restore: document.getElementById('mode-restore'), date: document.getElementById('mode-date'), region: document.getElementById('mode-region') };
    Object.entries(btns).forEach(([m, btn]) => {
        if (!btn) return;
        if (m === mode) {
            btn.style.borderColor = colors.cinnabar;
            btn.style.color = colors.cinnabar;
        } else {
            btn.style.borderColor = '#D6CBBA';
            btn.style.color = colors.charcoal;
        }
    });

    if (mode === 'restore' && currentResponse) {
        // Auto-select the first restored position if none chosen yet.
        const restorable = currentResponse.restorations.filter(r => !r.is_unk);
        if (activeRestorationIndex === null && restorable.length > 0) {
            activeRestorationIndex = restorable[0].token_index;
        }
    }

    renderText();
}

function renderResults() {
    // Nothing to show in "restore" saliency mode without any restorations
    // (attribution-only mode, or a "full" query with no '?'/'#' at all) --
    // fall back to dating saliency instead of an empty/stale highlight.
    const hasRestorations = currentResponse.restorations && currentResponse.restorations.length > 0;
    document.getElementById('mode-restore').classList.toggle('hidden', !hasRestorations);
    activeRestorationIndex = null;
    switchSaliencyMode(hasRestorations ? 'restore' : 'date');

    renderRegionChart(currentResponse.region_probs);
    renderDateHistogram(currentResponse.date_probs);
    renderSimilar(currentResponse.similar_documents);
}

function renderText() {
    if (!currentResponse) return;

    const container = document.getElementById('output-text');
    container.innerHTML = '';

    let saliency = [];
    if (activeSaliencyMode === 'date') {
        saliency = currentResponse.date_saliency;
    } else if (activeSaliencyMode === 'region') {
        saliency = currentResponse.region_saliency;
    } else if (activeSaliencyMode === 'restore' && activeRestorationIndex !== null) {
        const tokenData = currentResponse.restorations.find(t => t.token_index === activeRestorationIndex);
        if (tokenData && tokenData.saliency) {
            saliency = tokenData.saliency;
        }
    }

    let maxScore = 0;
    if (saliency && saliency.length > 0) {
        // Skip SOS for the max calc so the scale isn't dominated by it.
        const textScores = saliency.slice(1);
        if (textScores.length > 0) maxScore = Math.max(...textScores);
    }

    // Tokens contain SOS, characters, and EOS.
    currentResponse.tokens.forEach((token, i) => {
        // Skip special tokens in rendering
        if (token === "[SOS]" || token === "[EOS]" || token === "[PAD]") return;

        const span = document.createElement('span');

        const restData = currentResponse.restorations.find(t => t.token_index === i);

        if (restData) {
            span.textContent = restData.is_unk ? "#" : (restData.iterative_filled_char || restData.top_k[0].char);
            span.className = 'highlight-restored';
            if (restData.is_unk) {
                span.style.color = colors.cinnabar;
            }

            span.onmouseenter = (e) => showTooltip(e, restData);
            span.onmouseleave = hideTooltip;
            span.onclick = () => {
                if (!restData.is_unk) {
                    activeSaliencyMode = 'restore';
                    switchSaliencyMode('restore');
                    activeRestorationIndex = i;
                    renderText();
                }
            };

            if (activeSaliencyMode === 'restore' && activeRestorationIndex === i) {
                span.style.backgroundColor = colors.gold;
                span.style.color = '#fff';
            }
        } else {
            // No restData for a [-]/[#] here means attribution-only mode
            // skipped restoration entirely -- show the user's own marker
            // back rather than the raw internal token. [UNK] means "known
            // unrecoverable" (the user's own '-'), the opposite of a pending
            // mask -- '?' would misleadingly suggest the model is still
            // guessing here. '…' matches how the same marker is shown
            // everywhere else (similar-documents preview).
            const rawDisplay = { "[UNK]": "…", "[-]": "?", "[#]": "#" };
            span.textContent = rawDisplay[token] ?? token;
            span.className = 'token-context';

            if (saliency && i < saliency.length && maxScore > 0) {
                let alpha = Math.min(0.9, (saliency[i] / maxScore) * 0.9);
                span.style.backgroundColor = `rgba(209, 52, 38, ${alpha})`;
            }
        }

        container.appendChild(span);
    });
}

function renderRegionChart(regionData) {
    const ctx = document.getElementById('regionChart').getContext('2d');

    if (regionChartInstance) {
        regionChartInstance.destroy();
    }

    // Horizontal bar, sorted by probability -- same layout akkadian/src/web
    // uses for its own metadata heads, and reads straight off item.region
    // (the backend's own label) rather than a hand-maintained label array:
    // that array used to assume a ['NW','SW','OES','CS'] order while
    // REGION_NAMES in app.py actually returns ['OES','CS','NW','SW'], so the
    // old doughnut's legend entries didn't match the slice they labeled.
    const sorted = [...regionData].sort((a, b) => b.prob - a.prob);
    const topRegion = sorted[0].region;

    regionChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: sorted.map(item => item.region),
            datasets: [{
                data: sorted.map(item => item.prob),
                backgroundColor: sorted.map(item => item.region === topRegion ? colors.cinnabar : colors.limetree),
                borderRadius: 3,
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { callbacks: { label: (c) => `${(c.raw * 100).toFixed(1)}%` } }
            },
            scales: {
                x: { min: 0, max: 1, ticks: { callback: (v) => `${(v * 100).toFixed(0)}%` } },
                y: { ticks: { font: { family: 'Inter', size: 12 }, autoSkip: false } }
            }
        }
    });
}

function renderDateHistogram(dateData) {
    const container = document.getElementById('date-histogram');
    container.innerHTML = '';
    
    const probs = dateData.map(item => item.prob);

    // Must match BUCKETS_START in src/data_pipeline/prepare_datasets.py (800
    // AD, 20 bins of 50 years) -- this used to say 1000, silently shifting
    // every displayed date 200 years later than what the model actually
    // predicted.
    const startYear = 800;
    const binSize = 50;
    
    let maxProb = Math.max(...probs);
    let bestBinIndex = probs.indexOf(maxProb);
    
    const bestStart = startYear + (bestBinIndex * binSize);
    document.getElementById('best-date').textContent = `${bestStart} - ${bestStart + binSize} ${currentLang === 'ru' ? 'гг.' : 'CE'}`;
    
    probs.forEach((prob, i) => {
        const year = startYear + (i * binSize);
        
        const col = document.createElement('div');
        col.className = 'hist-bar-col';
        
        const bar = document.createElement('div');
        bar.className = 'hist-bar';
        if (i === bestBinIndex) bar.classList.add('active');
        
        const maxBarHeightPx = 135;
        const heightPx = maxProb > 0 ? (prob / maxProb) * maxBarHeightPx : 0;
        bar.style.height = `${Math.max(2, heightPx)}px`;
        bar.title = `${year}-${year+binSize}: ${(prob*100).toFixed(1)}%`;
        
        const label = document.createElement('div');
        label.className = 'hist-label';
        if (i % 2 === 0) {
            label.textContent = year;
        }
        
        col.appendChild(bar);
        col.appendChild(label);
        container.appendChild(col);
    });
}

// Tooltip logic
const tooltip = document.getElementById('tooltip');

function showTooltip(e, data) {
    let html = '';
    if (data.is_unk) {
        const pctMulti = (data.prob_multi * 100).toFixed(1);
        const pctSingle = (data.prob_single * 100).toFixed(1);
        const d = dict[currentLang];
        html = `
            <div style="margin-bottom: 5px; font-weight: 500; font-size: 0.9em; color: var(--gold);">${d.unk_label}</div>
            <table style="border-spacing: 0 4px; border-collapse: separate;">
                <tr>
                    <td style="font-weight:bold; font-size: 0.9em; padding-right: 10px;">${d.unk_multi}</td>
                    <td style="width: 80px;">
                        <div class="prob-bar" style="width: ${pctMulti}%; background-color: var(--cinnabar);"></div>
                    </td>
                    <td style="color: rgba(255,255,255,0.7); font-size: 0.9em; text-align: right; padding-left: 8px;">${pctMulti}%</td>
                </tr>
                <tr>
                    <td style="font-weight:bold; font-size: 0.9em; padding-right: 10px;">${d.unk_single}</td>
                    <td style="width: 80px;">
                        <div class="prob-bar" style="width: ${pctSingle}%"></div>
                    </td>
                    <td style="color: rgba(255,255,255,0.7); font-size: 0.9em; text-align: right; padding-left: 8px;">${pctSingle}%</td>
                </tr>
            </table>
        `;
    } else {
        html = '<table style="border-spacing: 0 4px; border-collapse: separate;">';
        data.top_k.forEach(item => {
            const pct = (item.prob * 100).toFixed(1);
            html += `
                <tr>
                    <td style="font-weight:bold; font-size: 1.2em; padding-right: 10px;">${item.char}</td>
                    <td style="width: 80px;">
                        <div class="prob-bar" style="width: ${pct}%"></div>
                    </td>
                    <td style="color: rgba(255,255,255,0.7); font-size: 0.9em; text-align: right; padding-left: 8px;">${pct}%</td>
                </tr>
            `;
        });
        html += '</table>';
    }
    
    tooltip.innerHTML = html;
    tooltip.classList.remove('hidden');
    
    // Positioning
    const rect = e.target.getBoundingClientRect();
    tooltip.style.left = rect.left + window.scrollX + 'px';
    tooltip.style.top = (rect.bottom + window.scrollY + 5) + 'px';
}

function hideTooltip() {
    tooltip.classList.add('hidden');
}

// ---------- Similar documents ----------

// doc_extra.json's own 'title' (gramoty.ru's city-scoped citation, e.g.
// "Грамота № Пск. 1" for a Pskov letter) wins when present -- the
// birchbark_classes.jsonl-derived doc_id's own number isn't city-scoped
// (birchbark_classes.jsonl uses "Пск. 1" fine, but a plain "Грамота №{n}"
// guess from doc_id alone would mislabel any non-Novgorod letter). Falls
// back to that guess for the ~1/3 of letters gramoty.ru's own site doesn't
// have a page for, and for every other source, which just show their raw doc_id.
function docLabel(doc) {
    if (doc.title) return doc.title;
    const m = /^birchbark_(.+)$/.exec(doc.doc_id || '');
    return m ? `Грамота №${m[1]}` : (doc.doc_id || '');
}

// Only NKRYA/birchbark/epigraphica actually tag genre -- RNC's own rows
// always carry the literal placeholder "unknown" (never populated at
// source), which would otherwise show up as real-looking noise ("· unknown
// ·") next to genuinely informative tags on every other source.
function realCategory(category) {
    return category && category !== "unknown" ? category : null;
}

function cleanDocText(text) {
    return (text || '').replace(/\[UNK\]/g, '…');
}

function renderSimilar(docs) {
    const grid = document.getElementById('similar-grid');
    grid.innerHTML = '';
    const d = dict[currentLang];

    if (!docs || docs.length === 0) {
        grid.innerHTML = `<div style="color:var(--chainmail); font-size:0.85rem;">${d.similar_empty}</div>`;
        return;
    }

    docs.forEach(doc => grid.appendChild(similarCard(doc)));
}

function similarCard(doc) {
    const el = document.createElement('div');
    el.className = 'similar-card';

    const pct = Math.round(doc.score * 100);
    const dateStr = doc.date_interval ? `${doc.date_interval[0]}–${doc.date_interval[1]}` : '';
    const tags = [doc.source_dataset, doc.macro_dialect, realCategory(doc.category), doc.region_city, doc.genre, doc.object_type, dateStr].filter(Boolean).join(' · ');
    const cleanText = cleanDocText(doc.text);
    const preview = cleanText.slice(0, 140) + (cleanText.length > 140 ? '…' : '');

    el.innerHTML = `
        <div class="similar-card-id">${docLabel(doc)}</div>
        <div class="similar-card-tags">${tags}</div>
        ${doc.summary ? `<div class="similar-card-summary">${doc.summary}</div>` : ''}
        <div class="similar-card-preview">${preview || '(нет текста)'}</div>
        <div class="similar-score-row">
            <span class="similar-score-bar-wrap"><span class="similar-score-bar" style="width:${pct}%"></span></span>
            <span class="similar-score-pct">${pct}%</span>
        </div>`;
    el.onclick = () => openDocModal(doc);
    return el;
}

function openDocModal(doc) {
    const body = document.getElementById('doc-modal-body');
    const dateStr = doc.date_interval ? `${doc.date_interval[0]}–${doc.date_interval[1]} гг.` : null;
    const tags = [doc.source_dataset, doc.macro_dialect, realCategory(doc.category), doc.region_city, doc.genre, doc.object_type, dateStr].filter(Boolean).join(' · ');
    // Same "label ↗" external-link pattern as akkadian/src/web's own sourceUrl()/tabletLink() --
    // doc_extra.json only has a source_url for the ~68% of birchbark letters
    // gramoty.ru's own site has a page for, and for epigraphica; other
    // sources just have no link here.
    const sourceLine = doc.source_url
        ? `<p class="modal-source"><a href="${doc.source_url}" target="_blank" rel="noopener">Источник ↗</a></p>` : '';
    body.innerHTML = `
        <div class="modal-title">${docLabel(doc)}</div>
        <div class="modal-tags">${tags || '(нет метаданных)'}</div>
        ${sourceLine}
        ${doc.summary ? `<div class="modal-section-label">Описание</div><div class="modal-text">${doc.summary}</div>` : ''}
        ${doc.place ? `<div class="modal-section-label">Место</div><div class="modal-text">${doc.place}</div>` : ''}
        <div class="modal-section-label">Текст</div>
        <div class="modal-text">${cleanDocText(doc.text) || '(нет текста)'}</div>
        ${doc.translation_ru ? `<div class="modal-section-label">Перевод</div><div class="modal-text">${doc.translation_ru}</div>` : ''}
        <div class="modal-section-label">Сходство с запросом</div>
        <div class="modal-text">${Math.round(doc.score * 100)}% (косинусное сходство эмбеддингов)</div>`;
    document.getElementById('doc-modal').classList.remove('hidden');
}

function closeDocModal() {
    document.getElementById('doc-modal').classList.add('hidden');
}

// index.html's data-i18n elements ship with a static (and easily stale --
// see git history) fallback string; changeLanguage() was previously only
// wired to the language <select>'s onchange, which never fires on initial
// load or when re-selecting the already-active language. Running it once
// here makes the dict the single source of truth for every data-i18n
// element from the very first paint.
changeLanguage(currentLang);
