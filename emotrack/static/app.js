// Общие функции дашборда «ЭмоТрек»

const EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"];
const EMOTION_RU = {
  angry: "Злость", disgust: "Отвращение", fear: "Страх", happy: "Радость",
  neutral: "Нейтрально", sad: "Грусть", surprise: "Удивление",
};
// Фиксированный цвет за каждой эмоцией (не зависит от порядка на графике)
const EMOTION_COLOR = {
  angry: "#e34948", disgust: "#008300", fear: "#6250d6", happy: "#eda100",
  neutral: "#2a78d6", sad: "#1baf7a", surprise: "#eb6834",
};
// Цвета участников — по порядку появления
const SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"];
const LEVEL_RU = { green: "высокий", yellow: "средний", red: "низкий" };

function levelOf(v) {
  if (v === null || v === undefined) return null;
  return v >= 70 ? "green" : v >= 40 ? "yellow" : "red";
}

function badge(v) {
  const lvl = levelOf(v);
  if (!lvl) return '<span class="badge grey">нет данных</span>';
  return `<span class="badge ${lvl}">${v.toFixed(1)} · ${LEVEL_RU[lvl]}</span>`;
}

function indexBar(v) {
  const lvl = levelOf(v) || "red";
  const w = Math.max(0, Math.min(100, v || 0));
  return `<div class="index-cell"><b>${(v ?? 0).toFixed(1)}</b>
          <div class="bar ${lvl}" style="flex:1"><span style="width:${w}%"></span></div></div>`;
}

function fmtTime(sec) {
  sec = Math.round(sec || 0);
  const m = Math.floor(sec / 60), s = sec % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

function topEmotion(probs) {
  if (!probs) return "—";
  let best = EMOTIONS[0];
  for (const e of EMOTIONS) if (probs[e] > probs[best]) best = e;
  return `${EMOTION_RU[best]} (${Math.round(probs[best] * 100)} %)`;
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

// Плагин Chart.js: цветные полосы уровней вовлечённости на фоне графика
const levelBands = {
  id: "levelBands",
  beforeDatasetsDraw(chart) {
    const { ctx, chartArea: a, scales: { y } } = chart;
    if (!y) return;
    const bands = [[70, 100, "rgba(12,163,12,0.07)"], [40, 70, "rgba(250,178,25,0.10)"], [0, 40, "rgba(208,59,59,0.07)"]];
    ctx.save();
    for (const [lo, hi, color] of bands) {
      const top = y.getPixelForValue(hi), bottom = y.getPixelForValue(lo);
      ctx.fillStyle = color;
      ctx.fillRect(a.left, top, a.right - a.left, bottom - top);
    }
    ctx.restore();
  },
};

if (window.Chart) {
  Chart.defaults.font.family = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
  Chart.defaults.color = "#52514e";
  Chart.defaults.borderColor = "#e1e0d9";
}
