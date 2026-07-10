/**
 * Adaptive Model Dispatcher — Dashboard Application Logic
 * =============================================
 * Loads run_report.json and renders charts + task table.
 * Supports drag-and-drop for file loading.
 */

// ---- Chart instances (for cleanup on reload) ----
let categoryChartInstance = null;
let modelChartInstance = null;
let sourceChartInstance = null;

// ---- Color palette ----
const COLORS = {
    indigo:  '#6366f1',
    violet:  '#8b5cf6',
    cyan:    '#06b6d4',
    emerald: '#10b981',
    amber:   '#f59e0b',
    red:     '#ef4444',
    pink:    '#ec4899',
    blue:    '#3b82f6',
};

const CATEGORY_COLORS = {
    factual_knowledge:        COLORS.indigo,
    mathematical_reasoning:   COLORS.violet,
    sentiment_classification: COLORS.pink,
    text_summarization:       COLORS.cyan,
    named_entity_recognition: COLORS.emerald,
    code_debugging:           COLORS.red,
    logical_reasoning:        COLORS.amber,
    code_generation:          COLORS.blue,
    unknown:                  '#5a5e72',
};

const MODEL_COLORS = [
    COLORS.indigo, COLORS.cyan, COLORS.violet,
    COLORS.emerald, COLORS.pink, COLORS.amber,
];

const SOURCE_COLORS = {
    deterministic:    COLORS.amber,
    local_model:      COLORS.emerald,
    gemma_bonus:      COLORS.pink,
    default_model:    COLORS.indigo,
    reasoning_model:  COLORS.violet,
    code_model:       COLORS.cyan,
    error:            COLORS.red,
    timeout:          COLORS.red,
    error_disallowed: COLORS.red,
    error_future:     COLORS.red,
};

// ---- Chart.js global config ----
Chart.defaults.color = '#9ca0b0';
Chart.defaults.borderColor = 'rgba(99, 102, 241, 0.08)';
Chart.defaults.font.family = "'Inter', sans-serif";

// ---- Utility ----
function formatNumber(n) {
    if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
    if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
    return n.toLocaleString();
}

function formatDuration(seconds) {
    if (seconds < 1) return (seconds * 1000).toFixed(0) + 'ms';
    if (seconds < 60) return seconds.toFixed(1) + 's';
    const m = Math.floor(seconds / 60);
    const s = (seconds % 60).toFixed(0);
    return `${m}m ${s}s`;
}

function prettyCategoryName(cat) {
    if (!cat) return '—';
    return cat.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function prettyModelName(model) {
    if (!model || model === 'none') return '—';
    // Extract last part: "accounts/fireworks/models/minimax-m3" -> "minimax-m3"
    const parts = model.split('/');
    return parts[parts.length - 1];
}

function prettySourceName(source) {
    if (!source) return '—';
    return source.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

// ---- Data loading ----
async function loadReport() {
    try {
        // Try loading from relative path (served alongside dashboard)
        const paths = [
            '../local_test/run_report.json',
            './run_report.json',
            '../run_report.json',
        ];

        let data = null;
        for (const path of paths) {
            try {
                const resp = await fetch(path);
                if (resp.ok) {
                    data = await resp.json();
                    break;
                }
            } catch (e) { /* try next */ }
        }

        if (data) {
            renderDashboard(data);
        } else {
            setStatus('No data', true);
        }
    } catch (err) {
        console.error('Failed to load report:', err);
        setStatus('Load failed', true);
    }
}

function loadFromFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        try {
            const data = JSON.parse(e.target.result);
            renderDashboard(data);
        } catch (err) {
            console.error('Failed to parse file:', err);
            setStatus('Parse error', true);
        }
    };
    reader.readAsText(file);
}

// ---- Status badge ----
function setStatus(text, isError = false) {
    const badge = document.getElementById('statusBadge');
    const statusText = badge.querySelector('.status-text');
    statusText.textContent = text;
    badge.classList.toggle('error', isError);
}

// ---- Main render ----
function renderDashboard(data) {
    // Summary cards
    document.getElementById('totalTasks').textContent = data.total_tasks || 0;
    document.getElementById('totalDuration').textContent = formatDuration(data.total_duration_seconds || 0);
    document.getElementById('totalTokens').textContent = formatNumber(data.total_tokens || 0);
    document.getElementById('correctedCount').textContent = data.corrected_count || 0;

    setStatus(`Run: ${new Date(data.run_id).toLocaleTimeString()}`, false);

    // Charts
    renderCategoryChart(data);
    renderModelChart(data);
    renderSourceChart(data);

    // Table
    renderTaskTable(data.tasks || []);
}

// ---- Category Distribution Chart ----
function renderCategoryChart(data) {
    const ctx = document.getElementById('categoryChart').getContext('2d');
    if (categoryChartInstance) categoryChartInstance.destroy();

    const stats = data.category_stats || {};
    const labels = Object.keys(stats).map(prettyCategoryName);
    const values = Object.values(stats).map(s => s.count);
    const colors = Object.keys(stats).map(k => CATEGORY_COLORS[k] || '#5a5e72');
    const totalTasks = values.reduce((a, b) => a + b, 0);

    // Custom plugin to draw total task count in the center
    const centerTextPlugin = {
        id: 'centerText',
        afterDraw(chart) {
            const { ctx, chartArea: { left, right, top, bottom, width, height } } = chart;
            ctx.save();
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';

            // Draw count
            ctx.font = '800 1.8rem "Inter", sans-serif';
            ctx.fillStyle = '#ffffff';
            ctx.fillText(totalTasks, left + width / 2, top + height / 2 - 6);

            // Draw label
            ctx.font = '600 0.68rem "Inter", sans-serif';
            ctx.fillStyle = '#5a5e72';
            ctx.fillText('TOTAL TASKS', left + width / 2, top + height / 2 + 16);
            ctx.restore();
        }
    };

    categoryChartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: colors.map(c => c + '30'),
                borderColor: colors,
                borderWidth: 1.5,
                hoverOffset: 4,
                spacing: values.length > 1 ? 5 : 0,
                borderRadius: 4,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '80%',
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        padding: 14,
                        usePointStyle: true,
                        pointStyleWidth: 8,
                        boxHeight: 8,
                        font: { size: 10, weight: '500' },
                        color: '#9ca0b8',
                    },
                },
                tooltip: {
                    backgroundColor: '#0c0c14',
                    borderColor: 'rgba(99, 102, 241, 0.25)',
                    borderWidth: 1,
                    padding: 10,
                    cornerRadius: 8,
                    titleFont: { size: 11, weight: '700' },
                    bodyFont: { size: 11 },
                    displayColors: false,
                },
            },
        },
        plugins: [centerTextPlugin],
    });
}

// ---- Model Token Usage Chart ----
function renderModelChart(data) {
    const ctx = document.getElementById('modelChart').getContext('2d');
    if (modelChartInstance) modelChartInstance.destroy();

    const usage = data.model_usage || {};
    const labels = Object.keys(usage).map(prettyModelName);
    const tokens = Object.values(usage).map(u => u.tokens);
    const calls = Object.values(usage).map(u => u.calls);

    // Create custom linear gradients for the model bars
    const barGradients = labels.map(label => {
        const name = label.toLowerCase();
        const grad = ctx.createLinearGradient(0, 0, 0, 300);
        if (name.includes('minimax')) {
            grad.addColorStop(0, 'rgba(99, 102, 241, 0.55)');
            grad.addColorStop(1, 'rgba(99, 102, 241, 0.02)');
        } else if (name.includes('kimi')) {
            grad.addColorStop(0, 'rgba(6, 182, 212, 0.55)');
            grad.addColorStop(1, 'rgba(6, 182, 212, 0.02)');
        } else if (name.includes('gemma')) {
            grad.addColorStop(0, 'rgba(236, 72, 153, 0.55)');
            grad.addColorStop(1, 'rgba(236, 72, 153, 0.02)');
        } else {
            grad.addColorStop(0, 'rgba(139, 92, 246, 0.55)');
            grad.addColorStop(1, 'rgba(139, 92, 246, 0.02)');
        }
        return grad;
    });

    const borderColors = labels.map(label => {
        const name = label.toLowerCase();
        if (name.includes('minimax')) return 'rgba(99, 102, 241, 0.9)';
        if (name.includes('kimi')) return 'rgba(6, 182, 212, 0.9)';
        if (name.includes('gemma')) return 'rgba(236, 72, 153, 0.9)';
        return 'rgba(139, 92, 246, 0.9)';
    });

    // Create line gradient for API calls area fill
    const lineFillGradient = ctx.createLinearGradient(0, 0, 0, 300);
    lineFillGradient.addColorStop(0, 'rgba(245, 158, 11, 0.12)');
    lineFillGradient.addColorStop(1, 'rgba(245, 158, 11, 0.0)');

    modelChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Tokens Spent',
                    data: tokens,
                    backgroundColor: barGradients,
                    borderColor: borderColors,
                    borderWidth: 1.5,
                    borderRadius: 6,
                    borderSkipped: false,
                    yAxisID: 'y',
                    barThickness: labels.length > 4 ? 'flex' : 32,
                },
                {
                    label: 'API Calls',
                    data: calls,
                    type: 'line',
                    borderColor: 'rgba(245, 158, 11, 0.95)',
                    backgroundColor: lineFillGradient,
                    fill: true,
                    borderWidth: 2.5,
                    pointRadius: 4.5,
                    pointHoverRadius: 7,
                    pointBackgroundColor: '#06060b',
                    pointBorderColor: 'rgba(245, 158, 11, 1)',
                    pointBorderWidth: 2,
                    tension: 0.35,
                    yAxisID: 'y1',
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
                y: {
                    type: 'linear',
                    position: 'left',
                    title: { display: true, text: 'Tokens', font: { size: 10, weight: '600' }, color: '#5a5e72' },
                    grid: { color: 'rgba(255, 255, 255, 0.02)' },
                    border: { display: false },
                    ticks: { color: '#5a5e72', font: { size: 9 } },
                },
                y1: {
                    type: 'linear',
                    position: 'right',
                    title: { display: true, text: 'API Calls', font: { size: 10, weight: '600' }, color: '#5a5e72' },
                    grid: { drawOnChartArea: false },
                    border: { display: false },
                    ticks: { color: '#5a5e72', font: { size: 9 }, stepSize: 1 },
                },
                x: {
                    grid: { display: false },
                    border: { display: false },
                    ticks: { color: '#9ca0b8', font: { size: 10, weight: '500' } },
                },
            },
            plugins: {
                legend: {
                    align: 'end',
                    labels: {
                        padding: 14,
                        usePointStyle: true,
                        pointStyleWidth: 8,
                        boxHeight: 8,
                        font: { size: 10, weight: '500' },
                        color: '#9ca0b8',
                    },
                },
                tooltip: {
                    backgroundColor: '#0c0c14',
                    borderColor: 'rgba(99, 102, 241, 0.25)',
                    borderWidth: 1,
                    padding: 10,
                    cornerRadius: 8,
                    titleFont: { size: 11, weight: '700' },
                    bodyFont: { size: 11 },
                },
            },
        },
    });
}

// ---- Routing Sources Chart ----
function renderSourceChart(data) {
    const ctx = document.getElementById('sourceChart').getContext('2d');
    if (sourceChartInstance) sourceChartInstance.destroy();

    const tasks = data.tasks || [];
    const sourceCounts = {};
    tasks.forEach(t => {
        const src = t.source || 'unknown';
        sourceCounts[src] = (sourceCounts[src] || 0) + 1;
    });

    const labels = Object.keys(sourceCounts).map(prettySourceName);
    const values = Object.values(sourceCounts);
    const colors = Object.keys(sourceCounts).map(k => SOURCE_COLORS[k] || '#5a5e72');
    const totalSources = values.length;

    // Use a clean concentric doughnut instead of polar area to match category distribution
    const centerTextPlugin = {
        id: 'centerText',
        afterDraw(chart) {
            const { ctx, chartArea: { left, right, top, bottom, width, height } } = chart;
            ctx.save();
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';

            // Draw count
            ctx.font = '800 1.8rem "Inter", sans-serif';
            ctx.fillStyle = '#ffffff';
            ctx.fillText(totalSources, left + width / 2, top + height / 2 - 6);

            // Draw label
            ctx.font = '600 0.68rem "Inter", sans-serif';
            ctx.fillStyle = '#5a5e72';
            ctx.fillText('SOURCES USED', left + width / 2, top + height / 2 + 16);
            ctx.restore();
        }
    };

    sourceChartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: colors.map(c => c + '30'),
                borderColor: colors,
                borderWidth: 1.5,
                hoverOffset: 4,
                spacing: values.length > 1 ? 5 : 0,
                borderRadius: 4,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '80%',
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        padding: 14,
                        usePointStyle: true,
                        pointStyleWidth: 8,
                        boxHeight: 8,
                        font: { size: 10, weight: '500' },
                        color: '#9ca0b8',
                    },
                },
                tooltip: {
                    backgroundColor: '#0c0c14',
                    borderColor: 'rgba(99, 102, 241, 0.25)',
                    borderWidth: 1,
                    padding: 10,
                    cornerRadius: 8,
                    titleFont: { size: 11, weight: '700' },
                    bodyFont: { size: 11 },
                    displayColors: false,
                },
            },
        },
        plugins: [centerTextPlugin],
    });
}

// ---- Task Table ----
function renderTaskTable(tasks) {
    const tbody = document.getElementById('taskTableBody');

    if (!tasks.length) {
        tbody.innerHTML = '<tr><td colspan="8" class="empty-state">No tasks in report.</td></tr>';
        return;
    }

    tbody.innerHTML = tasks.map(t => {
        const source = t.source || 'unknown';
        const badgeClass = `badge badge-${source.replace(/\s+/g, '_')}`;
        const tokens = t.tokens_spent || 0;
        let tokenClass = 'tokens-cell';
        if (tokens === 0) tokenClass += ' tokens-zero';
        else if (tokens < 200) tokenClass += ' tokens-low';
        else if (tokens < 500) tokenClass += ' tokens-med';
        else tokenClass += ' tokens-high';

        return `<tr>
            <td>${t.task_id || '—'}</td>
            <td><span class="badge-cat">${prettyCategoryName(t.category)}</span></td>
            <td><span class="${badgeClass}">${prettySourceName(source)}</span></td>
            <td>${prettyModelName(t.model_used)}</td>
            <td class="${tokenClass}">${formatNumber(tokens)}</td>
            <td class="latency-cell">${t.latency_seconds != null ? t.latency_seconds.toFixed(3) + 's' : '—'}</td>
            <td class="${t.was_corrected ? 'corrected-yes' : 'corrected-no'}">${t.was_corrected ? '✓ Yes' : '—'}</td>
            <td class="prompt-preview" title="${(t.prompt_preview || '').replace(/"/g, '&quot;')}">${t.prompt_preview || '—'}</td>
        </tr>`;
    }).join('');
}

// ---- Interactive Slide Deck Switcher ----
function switchSlide(index) {
    const tabs = document.querySelectorAll('.deck-tab');
    const slides = document.querySelectorAll('.deck-slide');

    tabs.forEach((tab, i) => {
        if (i === index) {
            tab.classList.add('active');
        } else {
            tab.classList.remove('active');
        }
    });

    slides.forEach((slide, i) => {
        if (i === index) {
            slide.classList.add('active');
        } else {
            slide.classList.remove('active');
        }
    });
}


// ---- Drag & Drop ----
const dropOverlay = document.getElementById('dropOverlay');

document.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropOverlay.classList.add('active');
});

document.addEventListener('dragleave', (e) => {
    if (e.relatedTarget === null || !document.body.contains(e.relatedTarget)) {
        dropOverlay.classList.remove('active');
    }
});

document.addEventListener('drop', (e) => {
    e.preventDefault();
    dropOverlay.classList.remove('active');

    const files = e.dataTransfer?.files;
    if (files && files.length > 0) {
        loadFromFile(files[0]);
    }
});

dropOverlay.addEventListener('dragleave', () => {
    dropOverlay.classList.remove('active');
});

// ---- Init ----
document.addEventListener('DOMContentLoaded', () => {
    loadReport();
});
