const dict = {
    ru: {
        desc_title: "Исследуйте тексты",
        header_subtitle: "ДРЕВНЕРУССКАЯ ЭПИГРАФИКА",
        desc_text: "Введите древнерусский текст с пропусками, чтобы восстановить утерянные символы и определить диалект и время написания.<br><br>Используйте знак вопроса (<strong>?</strong>) для каждого символа, который модель должна восстановить. Используйте символ решетки (<strong>#</strong>) для восстановления текста неизвестной длины. Символ дефиса (<strong>-</strong>) используйте для любых невосстанавливаемых пропусков в тексте.",
        ex_label: "Примеры из грамот",
        ex_default: "-- Выбрать пример --",
        ex_1: "Реставрация букв: Степанъ...",
        ex_2: "Молитва с пропусками и лакунами",
        ex_3: "Письмо с сильными повреждениями",
        keyboard_label: "Древнерусская клавиатура",
        btn_analyze: "Анализировать текст",
        res_title: "Результаты анализа",
        attn_legend: "Внимание модели:",
        loading: "Анализируем текст...",
        temp_label: "Температура (креативность)",
        temp_formulaic: "Формульная",
        temp_creative: "Креативная",
        mode_attr: "Внимание: Атрибутика",
        mode_rest: "Внимание: Реставрация",
        region_label: "Диалект / Регион",
        date_label: "Датировка",
        res_empty: "Здесь появятся результаты восстановления и атрибутики текста.",
        unk_label: "Прогноз длины лакуны:",
        unk_multi: "> 1 символа",
        unk_single: "1 символ"
    },
    en: {
        desc_title: "Explore Texts",
        header_subtitle: "OLD EAST SLAVIC EPIGRAPHY",
        desc_text: "Enter your Old East Slavic text below to restore missing characters, and attribute the text to its original dialect and time of writing.<br><br>Use a question mark (<strong>?</strong>) for each character you want the model to predict. Use a single hash (<strong>#</strong>) to predict text sequences of unknown length. Use a dash (<strong>-</strong>) for any missing sections or characters in your text that do not need restoring.",
        ex_label: "Birchbark Examples",
        ex_default: "-- Select example --",
        ex_1: "Letter restoration: Stepan...",
        ex_2: "Prayer with gaps and lacunae",
        ex_3: "Heavily damaged letter",
        keyboard_label: "Ancient Cyrillic Keyboard",
        btn_analyze: "Analyze Text",
        res_title: "Analysis Results",
        attn_legend: "Model Attention:",
        loading: "Analyzing text...",
        temp_label: "Temperature (Creativity)",
        temp_formulaic: "Formulaic",
        temp_creative: "Creative",
        mode_attr: "Attention: Attributes",
        mode_rest: "Attention: Restoration",
        region_label: "Dialect / Region",
        date_label: "Dating",
        res_empty: "Restoration and attribution results will appear here.",
        unk_label: "Gap size prediction:",
        unk_multi: "> 1 character",
        unk_single: "1 character"
    }
};

const examples = {
    ex1: "Степанъ тивꙋнъ наѱл? пѧт҇н? Рости славль",
    ex2: "ги҃ помози рабомъ сво??? павлоу с?мь#нови",
    ex3: "отъ ??гола и ото говѣна ко дурьдѣв# п?иш#"
};

let currentLang = 'ru';

function changeLanguage(lang) {
    currentLang = lang;
    const d = dict[lang];
    document.querySelectorAll('[data-i18n]').forEach(el => {
        const k = el.getAttribute('data-i18n');
        if (d[k] !== undefined) el.innerHTML = d[k];
    });
    
    // Update chart labels if it exists
    if (regionChartInstance) {
        regionChartInstance.data.labels = currentLang === 'ru' 
            ? ['Новгородский', 'Юго-Западный', 'Древневост.', 'Церковнослав.']
            : ['Novgorod', 'South-Western', 'Old East Slavic', 'Church Slavonic'];
        regionChartInstance.update();
    }
}

function loadExample() {
    const val = document.getElementById('examples-select').value;
    const ta = document.getElementById('input-text');
    if (val && examples[val]) {
        ta.value = examples[val];
        updateCounter();
    }
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
let activeAttentionMode = 'attr'; // 'attr' or 'rest'
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

        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, temperature: temp })
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

function switchAttentionMode(mode) {
    activeAttentionMode = mode;
    
    // Update button styles
    const btnAttr = document.getElementById('mode-attr');
    const btnRest = document.getElementById('mode-rest');
    
    if (mode === 'attr') {
        btnAttr.style.borderColor = colors.cinnabar;
        btnAttr.style.color = colors.cinnabar;
        btnRest.style.borderColor = '#D6CBBA';
        btnRest.style.color = colors.charcoal;
        activeRestorationIndex = null; // reset
    } else {
        btnRest.style.borderColor = colors.cinnabar;
        btnRest.style.color = colors.cinnabar;
        btnAttr.style.borderColor = '#D6CBBA';
        btnAttr.style.color = colors.charcoal;
        
        // Auto-select first restoration if available
        if (currentResponse && currentResponse.tokens.length > 0) {
            activeRestorationIndex = currentResponse.tokens[0].token_index;
        }
    }
    
    renderText();
}

function renderResults() {
    renderText();
    renderRegionChart(currentResponse.region_probs);
    renderDateHistogram(currentResponse.date_probs);
}

function renderText() {
    if (!currentResponse) return;
    
    const container = document.getElementById('output-text');
    container.innerHTML = '';
    
    let attentionWeights = [];
    if (activeAttentionMode === 'attr') {
        attentionWeights = currentResponse.sos_attention;
    } else if (activeAttentionMode === 'rest' && activeRestorationIndex !== null) {
        const tokenData = currentResponse.restorations.find(t => t.token_index === activeRestorationIndex);
        if (tokenData && tokenData.attention) {
            attentionWeights = tokenData.attention;
        }
    }
    
    let maxWeight = 0;
    if (attentionWeights && attentionWeights.length > 0) {
        // Skip SOS and EOS for max weight calc to make text highlights visible
        // (usually SOS has very high self-attention)
        const textWeights = attentionWeights.slice(1, -1);
        if (textWeights.length > 0) maxWeight = Math.max(...textWeights);
    }
    
    // Tokens contain SOS, characters, and EOS.
    currentResponse.tokens.forEach((token, i) => {
        // Skip special tokens in rendering
        if (token === "[SOS]" || token === "[EOS]" || token === "[PAD]") return;
        
        const span = document.createElement('span');
        
        const restData = currentResponse.restorations.find(t => t.token_index === i);
        
        if (restData) {
            span.textContent = restData.is_unk ? "#" : restData.top_k[0].char;
            span.className = 'highlight-restored';
            if (restData.is_unk) {
                span.style.color = colors.cinnabar;
            }
            
            span.onmouseenter = (e) => showTooltip(e, restData);
            span.onmouseleave = hideTooltip;
            span.onclick = () => {
                if (activeAttentionMode !== 'rest') {
                    switchAttentionMode('rest');
                }
                activeRestorationIndex = i;
                renderText();
            };
            
            if (activeAttentionMode === 'rest' && activeRestorationIndex === i) {
                span.style.backgroundColor = colors.gold;
                span.style.color = '#fff';
            }
        } else {
            span.textContent = token === "[UNK]" ? "?" : token;
            span.className = 'token-context';
            
            if (attentionWeights && i < attentionWeights.length && maxWeight > 0) {
                // Normalize using text maxWeight
                let alpha = Math.min(0.9, (attentionWeights[i] / maxWeight) * 0.9);
                span.style.backgroundColor = `rgba(212, 163, 115, ${alpha})`;
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
    
    // API returns array of objects: {region: "...", prob: 0.1}
    const probs = regionData.map(item => item.prob);
    
    regionChartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: currentLang === 'ru' 
                ? ['Новгородский', 'Юго-Западный', 'Древневост.', 'Церковнослав.']
                : ['Novgorod', 'South-Western', 'Old East Slavic', 'Church Slavonic'],
            datasets: [{
                data: probs,
                backgroundColor: [
                    colors.cinnabar,
                    colors.limetree,
                    colors.gold,
                    colors.charcoal
                ],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '65%',
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        font: { family: 'Inter', size: 11 },
                        color: colors.chainmail,
                        boxWidth: 12
                    }
                }
            }
        }
    });
}

function renderDateHistogram(dateData) {
    const container = document.getElementById('date-histogram');
    container.innerHTML = '';
    
    const probs = dateData.map(item => item.prob);
    
    const startYear = 1000;
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
        
        const maxBarHeightPx = 25;
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
