const MODEL_LABELS = {
  "mlp": "MLP Baseline",
  "mlp-obl": "MLP + OBL",
  "cnn": "CNN Baseline",
  "cnn-obl": "CNN + OBL",
};
const MODEL_ORDER = ["mlp", "mlp-obl", "cnn", "cnn-obl"];

const DATASET_LABELS = {
  "mnist": "MNIST",
  "fashion-mnist": "Fashion-MNIST",
  "cifar10": "CIFAR-10",
};
const DATASET_ORDER = ["mnist", "fashion-mnist", "cifar10"];

const COLORS = {
  "mlp": "#0a6f6f",
  "mlp-obl": "#d86b2c",
  "cnn": "#3a7d44",
  "cnn-obl": "#1f4e8c",
};

let chartInstances = [];

function resetCharts() {
  chartInstances.forEach((chart) => chart.destroy());
  chartInstances = [];
}

function trackChart(chart) {
  if (chart) {
    chartInstances.push(chart);
  }
  return chart;
}

function clearElement(element) {
  if (element) {
    element.innerHTML = "";
  }
}

function labelDataset(name) {
  if (!name) return "unknown";
  return DATASET_LABELS[name] || name;
}

function sortByPriority(names, priority) {
  const order = new Map(priority.map((value, index) => [value, index]));
  return names.slice().sort((a, b) => {
    const aOrder = order.has(a) ? order.get(a) : Infinity;
    const bOrder = order.has(b) ? order.get(b) : Infinity;
    if (aOrder !== bOrder) return aOrder - bOrder;
    return a.localeCompare(b);
  });
}

function formatDatasetInfo(datasetName, datasetInfo) {
  const label = labelDataset(datasetName);
  if (!datasetInfo) return label;
  const train = datasetInfo.train_size ?? "-";
  const val = datasetInfo.val_size ?? "-";
  const test = datasetInfo.test_size ?? "-";
  if (train === "-" && val === "-" && test === "-") {
    return label;
  }
  return `${label} (train ${train}, val ${val}, test ${test})`;
}

function formatPercent(stats, digits = 2) {
  if (!stats || stats.mean === undefined) return "-";
  const mean = (stats.mean * 100).toFixed(digits);
  const std = stats.std !== undefined ? (stats.std * 100).toFixed(digits) : "0.00";
  return `${mean}% +/- ${std}%`;
}

function formatNumber(stats, digits = 2, suffix = "") {
  if (!stats || stats.mean === undefined) return "-";
  const mean = stats.mean.toFixed(digits);
  const std = stats.std !== undefined ? stats.std.toFixed(digits) : "0.00";
  return `${mean} +/- ${std}${suffix}`;
}

function formatSeconds(stats) {
  if (!stats || stats.mean === undefined) return "-";
  const mean = stats.mean;
  const std = stats.std || 0;
  const meanMin = mean / 60;
  return `${mean.toFixed(1)}s (+/- ${std.toFixed(1)}s) / ${meanMin.toFixed(2)}m`;
}

function formatParamPair(totalStats, trainableStats) {
  if (!totalStats || totalStats.mean === undefined) return "-";
  const total = totalStats.mean.toFixed(0);
  if (!trainableStats || trainableStats.mean === undefined) return total;
  const trainable = trainableStats.mean.toFixed(0);
  return `${total} / ${trainable}`;
}

function formatDelta(value, higherIsBetter, suffix = "%") {
  if (value === undefined || value === null) {
    return { label: "-", className: "delta-neutral" };
  }
  const sign = value >= 0 ? "+" : "";
  const formatted = `${sign}${value.toFixed(2)}${suffix}`;
  const isGood = higherIsBetter ? value >= 0 : value <= 0;
  return { label: formatted, className: isGood ? "delta-good" : "delta-bad" };
}

function addRunCard(container, key, run) {
  const card = document.createElement("div");
  card.className = "card";

  const title = document.createElement("h4");
  title.textContent = MODEL_LABELS[key] || key;
  card.appendChild(title);

  const acc = document.createElement("p");
  acc.textContent = `Test acc: ${formatPercent(run.summary.aggregate?.test_accuracy)}`;
  card.appendChild(acc);

  const step = document.createElement("p");
  step.textContent = `Step time: ${formatNumber(run.summary.aggregate?.final_train_step_time_ms, 2, " ms")}`;
  card.appendChild(step);

  const wall = document.createElement("p");
  wall.textContent = `Wall time: ${formatSeconds(run.summary.aggregate?.wall_time_sec)}`;
  card.appendChild(wall);

  const params = document.createElement("p");
  params.textContent = `Params: ${formatParamPair(run.param_count, run.trainable_param_count)}`;
  card.appendChild(params);

  const epoch = document.createElement("p");
  epoch.textContent = `Epoch to 97% val: ${run.epoch_to_97 ?? "-"}`;
  card.appendChild(epoch);

  container.appendChild(card);
}

function buildBarChart(ctx, labels, values, colors, label) {
  return trackChart(new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label,
          data: values,
          backgroundColor: colors,
          borderRadius: 6,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
      },
      scales: {
        y: { beginAtZero: true },
      },
    },
  }));
}

function buildLineChart(ctx, series) {
  return trackChart(new Chart(ctx, {
    type: "line",
    data: {
      labels: series.labels,
      datasets: series.datasets,
    },
    options: {
      responsive: true,
      plugins: { legend: { position: "bottom" } },
      interaction: { intersect: false, mode: "index" },
      scales: { y: { beginAtZero: false } },
    },
  }));
}

function buildScatterChart(ctx, points) {
  return trackChart(new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: points,
    },
    options: {
      responsive: true,
      plugins: { legend: { position: "bottom" } },
      scales: {
        x: { title: { display: true, text: "Step Time (ms)" } },
        y: { title: { display: true, text: "Test Accuracy" } },
      },
    },
  }));
}

function buildGroupedBarChart(ctx, labels, datasets, yLabel) {
  return trackChart(new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets,
    },
    options: {
      responsive: true,
      plugins: { legend: { position: "bottom" } },
      scales: {
        y: { beginAtZero: true, title: { display: !!yLabel, text: yLabel || "" } },
      },
    },
  }));
}

function buildCurveDataset(run, labelPrefix, split = "val", metric = "accuracy") {
  if (!run) return null;
  const curve = run.curves?.[split];
  if (!curve || !curve.epoch?.length) return null;
  return {
    labels: curve.epoch,
    data: curve[metric],
    label: labelPrefix,
    borderColor: COLORS[run.model] || "#666",
    backgroundColor: "transparent",
    tension: 0.2,
  };
}

async function main() {
  const runCards = document.getElementById("run-cards");
  const generatedAt = document.getElementById("generated-at");
  const datasetInfo = document.getElementById("dataset-info");
  const datasetSelect = document.getElementById("dataset-select");
  const groupSelect = document.getElementById("group-select");
  const groupInfo = document.getElementById("group-info");
  const envDetails = document.getElementById("env-details");
  const derivedTable = document.getElementById("derived-metrics");
  const analysisSections = document.getElementById("analysis-sections");
  const scorecard = document.querySelector("#scorecard tbody");
  const deltaCards = document.getElementById("delta-cards");
  const banner = document.getElementById("constraint-banner");

  let report;
  try {
    const response = await fetch("data/report.json");
    report = await response.json();
  } catch (err) {
    runCards.innerHTML = `<div class="panel empty">Report data not found. Run build_report.py first.</div>`;
    return;
  }

  generatedAt.textContent = report.generated_at || "-";

  const datasets = report.datasets || {};
  const datasetNames = sortByPriority(Object.keys(datasets), DATASET_ORDER);
  if (datasetNames.length === 0) {
    runCards.innerHTML = `<div class="panel empty">No runs detected. Execute experiments to populate metrics.</div>`;
    return;
  }

  function populateDatasetSelect() {
    if (!datasetSelect) return datasetNames[0];
    clearElement(datasetSelect);
    datasetNames.forEach((name) => {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = labelDataset(name);
      datasetSelect.appendChild(option);
    });
    datasetSelect.value = datasetNames[0];
    return datasetSelect.value;
  }

  function populateGroupSelect(datasetName) {
    const datasetBlock = datasets[datasetName];
    const groups = datasetBlock?.groups || {};
    const groupNames = Object.keys(groups);
    if (!groupSelect) {
      return datasetBlock?.latest_group || groupNames[0];
    }
    clearElement(groupSelect);
    if (groupNames.length === 0) {
      return null;
    }
    groupNames.sort();
    const latest = datasetBlock?.latest_group;
    if (latest && groupNames.includes(latest)) {
      const filtered = groupNames.filter((name) => name !== latest);
      groupNames.splice(0, groupNames.length, latest, ...filtered);
    }
    groupNames.forEach((name) => {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      groupSelect.appendChild(option);
    });
    const selected = latest && groupNames.includes(latest) ? latest : groupNames[0];
    groupSelect.value = selected;
    return selected;
  }

  function renderGroup(datasetName, groupName) {
    resetCharts();
    clearElement(runCards);
    clearElement(envDetails);
    clearElement(derivedTable);
    clearElement(analysisSections);
    clearElement(scorecard);
    clearElement(deltaCards);
    clearElement(banner);
    clearElement(document.getElementById("insights"));
    clearElement(document.getElementById("comparisons"));

    const datasetBlock = datasets[datasetName];
    const group = datasetBlock?.groups?.[groupName];
    if (!group) {
      runCards.innerHTML = `<div class="panel empty">No runs detected for the selected dataset/group.</div>`;
      return;
    }

    const runs = group.runs || {};
    const runKeys = MODEL_ORDER.filter((key) => runs[key]);
    if (runKeys.length === 0) {
      runCards.innerHTML = `<div class="panel empty">No runs detected for the selected dataset/group.</div>`;
      return;
    }

    const groupLabel = group.run_group || groupName || "-";
    if (datasetInfo) {
      const info = group.dataset || datasetBlock?.dataset;
      datasetInfo.textContent = formatDatasetInfo(datasetName, info);
    }
    if (groupInfo) {
      const latestSuffix = datasetBlock?.latest_group === groupLabel ? " (latest)" : "";
      groupInfo.textContent = `${groupLabel}${latestSuffix}`;
    }

    runKeys.forEach((key) => addRunCard(runCards, key, runs[key]));

    const guardrail = group.guardrail ?? report.guardrail ?? 0.97;
    if (banner) {
      const warnings = [];
      runKeys.forEach((key) => {
        const accMean = runs[key].summary.aggregate?.test_accuracy?.mean;
        if (accMean !== undefined && accMean !== null && accMean < guardrail) {
          warnings.push(
            `${MODEL_LABELS[key] || key} below guardrail: ${(accMean * 100).toFixed(2)}%`
          );
        }
      });
      Object.entries(group.comparisons || {}).forEach(([pair, cmp]) => {
        const paramDelta = cmp.param_delta_pct;
        if (paramDelta !== undefined && paramDelta !== null && Math.abs(paramDelta) > 10) {
          warnings.push(`${pair.toUpperCase()} param delta ${paramDelta.toFixed(1)}% (rebalance)`);
        }
      });
      if (warnings.length === 0) {
        banner.classList.add("good");
        banner.innerHTML = "<strong>All guardrails satisfied.</strong> No accuracy or comparability warnings detected.";
      } else {
        banner.classList.remove("good");
        banner.innerHTML = `<strong>Constraints to review</strong><ul>${warnings
          .map((item) => `<li>${item}</li>`)
          .join("")}</ul>`;
      }
    }

    if (scorecard) {
      runKeys.forEach((key) => {
        const run = runs[key];
        const row = document.createElement("tr");
        row.className = key.includes("obl") ? "scorecard-variant" : "scorecard-baseline";
        const acc = run.summary.aggregate?.test_accuracy;
        const step = run.summary.aggregate?.final_train_step_time_ms;
        const wall = run.summary.aggregate?.wall_time_sec;
        const accPerWall = run.derived?.accuracy_per_wall_time;
        const totalParams = run.param_count;
        const trainableParams = run.trainable_param_count;
        const accMean = acc?.mean;
        const guardrailStatus = accMean !== undefined && accMean !== null && accMean >= guardrail;
        row.innerHTML = `
          <td>${MODEL_LABELS[key] || key}</td>
          <td>${formatParamPair(totalParams, trainableParams)}</td>
          <td>${formatPercent(acc)}</td>
          <td>${formatNumber(step, 2, " ms")}</td>
          <td>${formatSeconds(wall)}</td>
          <td>${accPerWall === undefined || accPerWall === null ? "-" : accPerWall.toFixed(4)}</td>
          <td class="${guardrailStatus ? "guardrail-pass" : "guardrail-fail"}">
            ${guardrailStatus ? "PASS" : "BELOW"}
          </td>
        `;
        scorecard.appendChild(row);
      });
    }

    const firstRun = runs[runKeys[0]];
    if (firstRun && firstRun.env) {
      const env = firstRun.env;
      const torchConfig = env.torch_config ? env.torch_config.split("\n").slice(0, 6).join("<br />") : "-";
      envDetails.innerHTML = `
        <div><strong>Platform:</strong> ${env.platform || "-"}</div>
        <div><strong>CPU:</strong> ${env.processor || "-"}</div>
        <div><strong>Threads:</strong> ${env.num_threads ?? "-"}</div>
        <div><strong>MKLDNN:</strong> ${env.mkldnn ? "enabled" : "disabled"}</div>
        <div><strong>Torch:</strong> ${env.torch || "-"} / TorchVision: ${env.torchvision || "-"}</div>
        <div><strong>Build:</strong><br />${torchConfig}</div>
      `;
    }

    const labels = runKeys.map((k) => MODEL_LABELS[k] || k);
    const colors = runKeys.map((k) => COLORS[k] || "#999");
    const accuracyValues = runKeys.map((k) => {
      const mean = runs[k].summary.aggregate?.test_accuracy?.mean;
      return mean === undefined || mean === null ? null : mean * 100;
    });
    const stepValues = runKeys.map((k) => runs[k].summary.aggregate?.final_train_step_time_ms?.mean ?? null);
    const wallValues = runKeys.map((k) => runs[k].summary.aggregate?.wall_time_sec?.mean ?? null);

    buildBarChart(document.getElementById("accuracy-chart"), labels, accuracyValues, colors, "Accuracy");
    buildBarChart(document.getElementById("step-time-chart"), labels, stepValues, colors, "Step Time");
    buildBarChart(document.getElementById("wall-time-chart"), labels, wallValues, colors, "Wall Time");

    const deltaLabels = ["MLP", "CNN"];
    const deltaMetrics = [
      { key: "test_accuracy_delta", label: "Acc delta (pp)", scale: 100, color: "#0a6f6f" },
      { key: "step_time_delta_pct", label: "Step time delta (%)", scale: 1, color: "#d86b2c" },
      { key: "wall_time_delta_pct", label: "Wall time delta (%)", scale: 1, color: "#3a7d44" },
      { key: "param_delta_pct", label: "Param delta (%)", scale: 1, color: "#1f4e8c" },
    ];
    const deltaDatasets = deltaMetrics.map((metric) => {
      const values = ["mlp", "cnn"].map((pair) => {
        const cmp = group.comparisons?.[pair];
        if (!cmp || cmp[metric.key] === undefined || cmp[metric.key] === null) {
          return null;
        }
        return cmp[metric.key] * metric.scale;
      });
      return { label: metric.label, data: values, backgroundColor: metric.color, borderRadius: 6 };
    });
    buildGroupedBarChart(document.getElementById("delta-chart"), deltaLabels, deltaDatasets, "Delta");

    const varianceDatasets = [
      {
        label: "Test acc std (pp)",
        data: runKeys.map((k) => {
          const std = runs[k].derived?.test_accuracy_std;
          return std === undefined || std === null ? null : std * 100;
        }),
        backgroundColor: "#0a6f6f",
        borderRadius: 6,
      },
      {
        label: "Max val std (pp)",
        data: runKeys.map((k) => {
          const std = runs[k].derived?.val_accuracy_std_max;
          return std === undefined || std === null ? null : std * 100;
        }),
        backgroundColor: "#d86b2c",
        borderRadius: 6,
      },
    ];
    buildGroupedBarChart(document.getElementById("variance-chart"), labels, varianceDatasets, "Std (pp)");

    const scatterPoints = runKeys
      .map((k) => {
        const step = runs[k].summary.aggregate?.final_train_step_time_ms?.mean;
        const acc = runs[k].summary.aggregate?.test_accuracy?.mean;
        if (step === undefined || step === null || acc === undefined || acc === null) {
          return null;
        }
        return {
          label: MODEL_LABELS[k] || k,
          data: [{ x: step, y: acc }],
          backgroundColor: COLORS[k] || "#999",
        };
      })
      .filter((point) => point !== null);
    buildScatterChart(document.getElementById("tradeoff-chart"), scatterPoints);

    const mlpSeries = [];
    const mlpBase = buildCurveDataset(runs["mlp"], "MLP val");
    const mlpObl = buildCurveDataset(runs["mlp-obl"], "MLP+OBL val");
    if (mlpBase) mlpSeries.push(mlpBase);
    if (mlpObl) mlpSeries.push(mlpObl);
    if (mlpSeries.length > 0) {
      buildLineChart(document.getElementById("mlp-curve"), {
        labels: mlpSeries[0].labels,
        datasets: mlpSeries,
      });
    }

    const mlpLossSeries = [];
    const mlpLossBase = buildCurveDataset(runs["mlp"], "MLP val", "val", "loss");
    const mlpLossObl = buildCurveDataset(runs["mlp-obl"], "MLP+OBL val", "val", "loss");
    if (mlpLossBase) mlpLossSeries.push(mlpLossBase);
    if (mlpLossObl) mlpLossSeries.push(mlpLossObl);
    if (mlpLossSeries.length > 0) {
      buildLineChart(document.getElementById("mlp-loss-curve"), {
        labels: mlpLossSeries[0].labels,
        datasets: mlpLossSeries,
      });
    }

    const cnnSeries = [];
    const cnnBase = buildCurveDataset(runs["cnn"], "CNN val");
    const cnnObl = buildCurveDataset(runs["cnn-obl"], "CNN+OBL val");
    if (cnnBase) cnnSeries.push(cnnBase);
    if (cnnObl) cnnSeries.push(cnnObl);
    if (cnnSeries.length > 0) {
      buildLineChart(document.getElementById("cnn-curve"), {
        labels: cnnSeries[0].labels,
        datasets: cnnSeries,
      });
    }

    const cnnLossSeries = [];
    const cnnLossBase = buildCurveDataset(runs["cnn"], "CNN val", "val", "loss");
    const cnnLossObl = buildCurveDataset(runs["cnn-obl"], "CNN+OBL val", "val", "loss");
    if (cnnLossBase) cnnLossSeries.push(cnnLossBase);
    if (cnnLossObl) cnnLossSeries.push(cnnLossObl);
    if (cnnLossSeries.length > 0) {
      buildLineChart(document.getElementById("cnn-loss-curve"), {
        labels: cnnLossSeries[0].labels,
        datasets: cnnLossSeries,
      });
    }

    const epochSeries = [];
    runKeys.forEach((key) => {
      const curve = runs[key].curves?.train;
      if (!curve || !curve.epoch?.length) return;
      epochSeries.push({
        label: `${MODEL_LABELS[key] || key} epoch time`,
        data: curve.epoch_time_sec,
        borderColor: COLORS[key] || "#999",
        backgroundColor: "transparent",
        tension: 0.2,
      });
    });
    if (epochSeries.length > 0) {
      buildLineChart(document.getElementById("epoch-time-chart"), {
        labels: runs[runKeys[0]].curves?.train?.epoch || [],
        datasets: epochSeries,
      });
    }

    const insightsList = document.getElementById("insights");
    if (group.insights && group.insights.length > 0) {
      group.insights.forEach((text) => {
        const li = document.createElement("li");
        li.textContent = text;
        insightsList.appendChild(li);
      });
    } else {
      insightsList.innerHTML = "<li>No insights generated yet.</li>";
    }

    const comparisons = document.getElementById("comparisons");
    Object.entries(group.comparisons || {}).forEach(([key, cmp]) => {
      const accDelta = cmp.test_accuracy_delta;
      const stepDelta = cmp.step_time_delta_pct;
      const wallDelta = cmp.wall_time_delta_pct;
      const div = document.createElement("div");
      div.className = "comparison-item";
      div.innerHTML = `
        <strong>${key.toUpperCase()}</strong><br />
        Accuracy delta: ${accDelta === undefined || accDelta === null ? "-" : (accDelta * 100.0).toFixed(2)} pp<br />
        Step time delta: ${stepDelta === undefined || stepDelta === null ? "-" : stepDelta.toFixed(1)}%<br />
        Wall time delta: ${wallDelta === undefined || wallDelta === null ? "-" : wallDelta.toFixed(1)}%
      `;
      comparisons.appendChild(div);
    });

    const comparisonPairs = [
      { label: "MLP", base: "mlp", variant: "mlp-obl", key: "mlp" },
      { label: "CNN", base: "cnn", variant: "cnn-obl", key: "cnn" },
    ];

    if (deltaCards) {
      comparisonPairs.forEach((pair) => {
        const cmp = group.comparisons?.[pair.key];
        if (!cmp) return;
        const card = document.createElement("div");
        card.className = "delta-card";
        const accDelta = cmp.test_accuracy_delta;
        const stepDelta = cmp.step_time_delta_pct;
        const wallDelta = cmp.wall_time_delta_pct;
        const paramDelta = cmp.param_delta_pct;
        const accBadge = accDelta === undefined || accDelta === null
          ? { label: "-", className: "delta-neutral" }
          : formatDelta(accDelta * 100, true, " pp");
        const stepBadge = formatDelta(stepDelta, false);
        const wallBadge = formatDelta(wallDelta, false);
        const paramBadge = formatDelta(paramDelta, false);
        card.innerHTML = `
          <h4>${pair.label} OBL vs Baseline</h4>
          <div class="delta-row"><span>Accuracy</span><span class="delta-badge ${accBadge.className}">${accBadge.label}</span></div>
          <div class="delta-row"><span>Step Time</span><span class="delta-badge ${stepBadge.className}">${stepBadge.label}</span></div>
          <div class="delta-row"><span>Wall Time</span><span class="delta-badge ${wallBadge.className}">${wallBadge.label}</span></div>
          <div class="delta-row"><span>Params</span><span class="delta-badge ${paramBadge.className}">${paramBadge.label}</span></div>
        `;
        deltaCards.appendChild(card);
      });
    }

    runKeys.forEach((key) => {
      const run = runs[key];
      const derived = run.derived || {};
      const row = document.createElement("div");
      row.className = "derived-row";
      row.innerHTML = `
        <strong>${MODEL_LABELS[key] || key}</strong>
        Generalization gap: ${derived.generalization_gap === null || derived.generalization_gap === undefined ? "-" : (derived.generalization_gap * 100).toFixed(2)} pp<br />
        Test std: ${derived.test_accuracy_std === null || derived.test_accuracy_std === undefined ? "-" : (derived.test_accuracy_std * 100).toFixed(2)} pp<br />
        Accuracy / wall-sec: ${derived.accuracy_per_wall_time === null || derived.accuracy_per_wall_time === undefined ? "-" : derived.accuracy_per_wall_time.toFixed(4)}<br />
        Accuracy / param: ${derived.accuracy_per_param === null || derived.accuracy_per_param === undefined ? "-" : derived.accuracy_per_param.toExponential(2)}<br />
        Max val std: ${derived.val_accuracy_std_max === null || derived.val_accuracy_std_max === undefined ? "-" : (derived.val_accuracy_std_max * 100).toFixed(2)} pp
      `;
      derivedTable.appendChild(row);
    });

    if (analysisSections && group.analysis_sections) {
      group.analysis_sections.forEach((section) => {
        const block = document.createElement("div");
        block.className = "analysis-section";
        const title = document.createElement("h4");
        title.textContent = section.title;
        block.appendChild(title);
        const list = document.createElement("ul");
        section.items.forEach((item) => {
          const li = document.createElement("li");
          li.textContent = item;
          list.appendChild(li);
        });
        block.appendChild(list);
        analysisSections.appendChild(block);
      });
    }
  }

  const initialDataset = populateDatasetSelect();
  const initialGroup = populateGroupSelect(initialDataset);
  renderGroup(initialDataset, initialGroup);

  if (datasetSelect) {
    datasetSelect.addEventListener("change", () => {
      const datasetName = datasetSelect.value;
      const groupName = populateGroupSelect(datasetName);
      renderGroup(datasetName, groupName);
    });
  }

  if (groupSelect) {
    groupSelect.addEventListener("change", () => {
      const datasetName = datasetSelect ? datasetSelect.value : initialDataset;
      renderGroup(datasetName, groupSelect.value);
    });
  }
}

main();
