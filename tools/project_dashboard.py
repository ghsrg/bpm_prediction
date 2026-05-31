"""
project_dashboard.py
====================
BPM Prediction — Project Dashboard with Research Metro Map.

Routes:
  /       — main status dashboard (capabilities, debt, ADRs, etc.)
  /map    — Research Journey Metro Map (SVG, interactive)
  /api/data — JSON data for the dashboard

Usage:
    .\.venv-modern\Scripts\python.exe tools\project_dashboard.py [--port 7878]
"""

import argparse
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

TOOLS_DIR    = Path(__file__).resolve().parent
PROJECT_ROOT = TOOLS_DIR.parent
STATE_FILE   = PROJECT_ROOT / "docs" / "current" / "project-state.md"
DEBT_FILE    = PROJECT_ROOT / "docs" / "current" / "architecture-debt.md"
ADR_DIR      = PROJECT_ROOT / "docs" / "adr"


# ---------------------------------------------------------------------------
# Markdown parsers (unchanged from previous version)
# ---------------------------------------------------------------------------

def _read(path):
    try: return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError: return ""

def _meta(text):
    meta, inside = {}, False
    for line in text.splitlines():
        if line.strip().startswith("## Metadata"): inside = True; continue
        if inside:
            if line.startswith("##"): break
            m = re.match(r"-\s+`(.+?)`:\s+`?(.+?)`?\s*$", line.strip())
            if m: meta[m.group(1)] = m.group(2)
    return meta

def _sections(text):
    sections, h2, h3, body = [], None, None, []
    def flush():
        if h3 and h2 is not None:
            h2["children"].append({"title": h3, "body": "\n".join(body).strip()})
        elif h2 is not None:
            h2["body"] = "\n".join(body).strip()
    for line in text.splitlines():
        if line.startswith("### "): flush(); body=[]; h3=line[4:].strip()
        elif line.startswith("## "): flush(); body=[]; h3=None; h2={"title":line[3:].strip(),"body":"","children":[]}; sections.append(h2)
        else: body.append(line)
    flush()
    return sections

def _kv(body):
    r = {}
    for line in body.splitlines():
        m = re.match(r"\s*-\s+`(.+?)`:\s+`?(.+?)`?\s*$", line)
        if m: r[m.group(1)] = m.group(2)
    return r

def parse_state():
    text = _read(STATE_FILE)
    if not text: return {"meta":{},"capabilities":[],"runtime_invariants":[],"recent_updates":[],"priorities":[]}
    meta = _meta(text); secs = _sections(text)
    caps, inv, upds, prios = [], [], [], []
    for s in secs:
        t = s["title"]
        if t == "Implemented Capabilities":
            for c in s["children"]:
                kv = _kv(c["body"]); caps.append({"name":c["title"],"status":kv.get("status","unknown"),"body":c["body"]})
        elif t == "Runtime Invariants":
            for c in s["children"]:
                kv = _kv(c["body"]); inv.append({"name":c["title"],"adr":kv.get("adr",""),"rule":kv.get("rule",""),"body":c["body"]})
        elif t == "Current Priorities":
            for line in s["body"].splitlines():
                m = re.match(r"\d+\.\s+`(.+?)`", line)
                if m: prios.append(m.group(1))
        elif t.startswith("Runtime Update"):
            upds.append({"title":t,"body":s["body"].strip()})
    return {"meta":meta,"capabilities":caps,"runtime_invariants":inv,"recent_updates":upds,"priorities":prios}

def parse_debt():
    text = _read(DEBT_FILE)
    if not text: return {"items":[]}
    items = []
    for s in _sections(text):
        if not s["title"].startswith("P"): continue
        for c in s["children"]:
            kv = _kv(c["body"]); items.append({
                "name":c["title"],"category":s["title"],
                "status":kv.get("status","unknown"),"priority":kv.get("priority","P?"),
                "adr":kv.get("adr",""),"current_behavior":kv.get("current_behavior",""),
                "target_state":kv.get("target_state",""),"body":c["body"]})
    return {"items":items}

def parse_adrs():
    adrs = []
    if not ADR_DIR.exists(): return adrs
    for path in sorted(ADR_DIR.glob("*.md")):
        if path.name == "README.md": continue
        text = _read(path); lines = text.splitlines()
        title=lines[0].lstrip("# ").strip() if lines else path.stem
        status=date=context=decision=""
        inc=ind=False
        for line in lines[1:]:
            if line.startswith("Status:"): status=line.split(":",1)[1].strip()
            elif line.startswith("Date:"): date=line.split(":",1)[1].strip()
            elif line.startswith("## Context"): inc=True;ind=False
            elif line.startswith("## Decision"): ind=True;inc=False
            elif line.startswith("## "): inc=ind=False
            elif inc: context+=line+"\n"
            elif ind: decision+=line+"\n"
        adrs.append({"id":path.stem,"title":title,"status":status,"date":date,
                     "context":context.strip(),"decision":decision.strip()})
    return adrs

def _last_mod():
    import datetime
    paths=[STATE_FILE,DEBT_FILE]+list(ADR_DIR.glob("*.md") if ADR_DIR.exists() else [])
    mt=max((p.stat().st_mtime for p in paths if p.exists()),default=0)
    return datetime.datetime.fromtimestamp(mt).strftime("%Y-%m-%d %H:%M") if mt else "N/A"

def build_data():
    state=parse_state(); debt=parse_debt(); adrs=parse_adrs()
    items=debt["items"]
    return {
        "state":state,"debt":debt,"adrs":adrs,
        "last_refreshed":_last_mod(),
        "debt_summary":{
            "P0_active":sum(1 for x in items if x["priority"]=="P0" and x["status"]=="active"),
            "P1_active":sum(1 for x in items if x["priority"]=="P1" and x["status"]=="active"),
            "closed":sum(1 for x in items if x["status"]=="closed"),
        }
    }


# ---------------------------------------------------------------------------
# Metro Map HTML
# ---------------------------------------------------------------------------

MAP_HTML = r"""<!DOCTYPE html>
<html lang="uk">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Research Journey — BPM Prediction</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
:root {
  --bg:#0d1117; --surface:#161b22; --surface2:#1c2128; --border:#30363d;
  --text:#e6edf3; --muted:#8b949e; --dim:#484f58;
  --c-trunk:#58a6ff; --c-fusion:#f85149; --c-ablation:#e3b341;
  --c-candidate:#3fb950; --c-method:#bc8cff; --c-infra:#8b949e;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); overflow: hidden; height: 100vh; }

.topbar {
  height: 52px; background: var(--surface); border-bottom: 1px solid var(--border);
  display: flex; align-items: center; padding: 0 20px; gap: 16px; position: relative; z-index: 50;
}
.topbar-title { font-size: 14px; font-weight: 700; }
.topbar-sub { font-size: 11px; color: var(--muted); }
.back-btn {
  background: var(--surface2); border: 1px solid var(--border); color: var(--muted);
  padding: 4px 12px; border-radius: 5px; cursor: pointer; font-size: 11px; font-family: inherit;
  text-decoration: none; display: flex; align-items: center; gap: 5px; transition: all .15s;
}
.back-btn:hover { color: var(--text); border-color: var(--c-trunk); }
.spacer { flex: 1; }

/* Line toggle pills */
.line-toggles { display: flex; gap: 6px; flex-wrap: wrap; }
.lt { display: flex; align-items: center; gap: 5px; padding: 4px 10px; border-radius: 20px;
  border: 1px solid; cursor: pointer; font-size: 10px; font-weight: 600; transition: all .2s; user-select: none; }
.lt.off { opacity: .35; }
.lt-dot { width: 8px; height: 8px; border-radius: 50%; }

/* Map viewport */
.map-outer {
  width: 100vw; height: calc(100vh - 52px);
  overflow: auto; position: relative; cursor: grab;
}
.map-outer.dragging { cursor: grabbing; }

.map-canvas {
  position: relative;
  /* sized by JS */
}

/* SVG */
#metro-svg { display: block; }

/* Station hit areas */
.station-hit {
  cursor: pointer; position: absolute;
  border-radius: 50%; transform: translate(-50%, -50%);
  z-index: 10;
}
.station-hit:hover + .station-label { opacity: 1 !important; }

/* Station labels */
.station-label {
  position: absolute; pointer-events: none;
  font-size: 10px; font-weight: 500; color: var(--text);
  text-align: center; line-height: 1.3; white-space: nowrap;
  z-index: 5; transition: opacity .15s;
}

/* Detail panel */
.detail-panel {
  position: fixed; right: 0; top: 52px; bottom: 0; width: 340px;
  background: var(--surface); border-left: 1px solid var(--border);
  transform: translateX(100%); transition: transform .25s ease-out;
  z-index: 100; overflow-y: auto; display: flex; flex-direction: column;
}
.detail-panel.open { transform: translateX(0); }
.dp-header {
  padding: 18px 20px; border-bottom: 1px solid var(--border); position: sticky; top: 0;
  background: var(--surface);
}
.dp-line-tag {
  font-size: 9px; font-weight: 700; letter-spacing: .8px; text-transform: uppercase;
  padding: 2px 8px; border-radius: 10px; display: inline-block; margin-bottom: 8px;
}
.dp-title { font-size: 16px; font-weight: 700; margin-bottom: 4px; }
.dp-date { font-size: 11px; color: var(--muted); }
.dp-status-badge {
  display: inline-block; font-size: 10px; font-weight: 600; padding: 2px 8px;
  border-radius: 8px; margin-top: 6px;
}
.dp-body { padding: 18px 20px; flex: 1; }
.dp-desc { font-size: 12px; color: var(--muted); line-height: 1.7; margin-bottom: 14px; }
.dp-details-title {
  font-size: 9px; font-weight: 600; text-transform: uppercase; letter-spacing: .6px;
  color: var(--dim); margin-bottom: 8px; padding-bottom: 5px; border-bottom: 1px solid var(--border);
}
.dp-detail-item {
  display: flex; gap: 8px; margin-bottom: 6px; font-size: 11px; color: var(--muted); line-height: 1.5;
}
.dp-detail-item::before { content: '·'; color: var(--dim); flex-shrink: 0; }
.dp-close {
  position: absolute; top: 16px; right: 16px;
  background: none; border: none; color: var(--muted); cursor: pointer; font-size: 16px;
  transition: color .15s;
}
.dp-close:hover { color: var(--text); }

/* Time axis labels */
.time-label {
  position: absolute; font-size: 9px; color: var(--dim); font-weight: 600;
  letter-spacing: .5px; text-transform: uppercase; text-align: center;
  transform: translateX(-50%); z-index: 2;
}

/* Lane labels */
.lane-label {
  position: absolute; right: 0; font-size: 9px; font-weight: 600;
  letter-spacing: .4px; text-align: right; z-index: 2;
  transform: translateY(-50%);
}

/* Legend */
.legend {
  position: fixed; bottom: 20px; left: 20px; z-index: 50;
  background: rgba(22,27,34,.92); border: 1px solid var(--border);
  border-radius: 10px; padding: 12px 16px; backdrop-filter: blur(8px);
}
.legend-title { font-size: 9px; font-weight: 700; text-transform: uppercase;
  letter-spacing: .8px; color: var(--dim); margin-bottom: 10px; }
.legend-item { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; font-size: 10px; color: var(--muted); }
.legend-line { width: 24px; height: 3px; border-radius: 2px; flex-shrink: 0; }
.legend-line.dashed { background: none; border-top: 2px dashed currentColor; height: 0; }

/* Tooltip */
.tooltip {
  position: fixed; z-index: 200; background: var(--surface2);
  border: 1px solid var(--border); border-radius: 6px;
  padding: 6px 10px; font-size: 11px; color: var(--text); pointer-events: none;
  opacity: 0; transition: opacity .15s; max-width: 200px; line-height: 1.4;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
</head>
<body>

<div class="topbar">
  <a class="back-btn" href="/">&#8592; Dashboard</a>
  <div>
    <div class="topbar-title">Research Journey — Metro Map</div>
    <div class="topbar-sub">BPM Prediction / Topology-Conditioned Zero-Shot · MVP2.5 Stage 4.2</div>
  </div>
  <div class="spacer"></div>
  <div class="line-toggles" id="line-toggles"></div>
</div>

<div class="map-outer" id="map-outer">
  <div class="map-canvas" id="map-canvas">
    <svg id="metro-svg"></svg>
  </div>
</div>

<div class="legend">
  <div class="legend-title">Умовні позначення</div>
  <div class="legend-item"><div class="legend-line" style="background:#8b949e"></div>Реалізовано</div>
  <div class="legend-item"><div class="legend-line dashed" style="color:#8b949e"></div>Експеримент / заплановано</div>
  <div class="legend-item"><svg width="24" height="14"><circle cx="12" cy="7" r="6" fill="none" stroke="#8b949e" stroke-width="2"/><circle cx="12" cy="7" r="3" fill="#8b949e"/></svg>Ключове рішення</div>
  <div class="legend-item"><svg width="24" height="14"><circle cx="12" cy="7" r="6" fill="#8b949e" stroke="none"/><text x="12" y="11" text-anchor="middle" fill="white" font-size="8">★</text></svg>Поточна позиція</div>
</div>

<div class="detail-panel" id="detail-panel">
  <div class="dp-header">
    <button class="dp-close" onclick="closeDetail()">&#x2715;</button>
    <div class="dp-line-tag" id="dp-tag"></div>
    <div class="dp-title" id="dp-title"></div>
    <div class="dp-date" id="dp-date"></div>
    <div class="dp-status-badge" id="dp-status"></div>
  </div>
  <div class="dp-body">
    <div class="dp-desc" id="dp-desc"></div>
    <div class="dp-details-title">Деталі</div>
    <div id="dp-details"></div>
  </div>
</div>

<div class="tooltip" id="tooltip"></div>

<script>
/* ============================================================
   DATA
   ============================================================ */
const LINES_DEF = [
  { id:'trunk',     name:'Dissertation Trunk',     shortName:'Trunk',      color:'#58a6ff', lane:0 },
  { id:'fusion',    name:'Fusion Architecture',    shortName:'Fusion',     color:'#f85149', lane:1 },
  { id:'ablation',  name:'Ablation / Experimental',shortName:'Ablation',   color:'#e3b341', lane:2 },
  { id:'candidate', name:'Candidate Contract',     shortName:'Candidate',  color:'#3fb950', lane:3 },
  { id:'method',    name:'Learning Methodology',   shortName:'Methodology',color:'#bc8cff', lane:4 },
  { id:'infra',     name:'Infrastructure',         shortName:'Infra',      color:'#8b949e', lane:5 },
];

// x is the "time position" (arbitrary units, will be scaled)
// lane maps to y
const STATIONS = [
  /* ── TRUNK ─────────────────────────────────────────────── */
  { id:'hyp', line:'trunk', x:80, label:'Research\nHypothesis', type:'origin',
    date:'Feb 2026', status:'done',
    desc:'Гіпотеза дисертації: модель може навчитись використовувати topology процесу як умовний prior у cold-start/zero-shot режимі нової версії BPMN. Стара версія дає логи, нова — тільки структуру.',
    details:[
      'Тема: topology-conditioned zero-shot predictive monitoring under structural process drift',
      'Бізнес-припущення: нова BPMN версія може бути доступна до появи логів vN',
      'Ціль: довести що structural BPMN topology дає корисний prior для next-activity prediction',
      'Ключова відмінність від простого structural augmentation: модель повинна реально залежати від topology',
    ]
  },
  { id:'mvp1', line:'trunk', x:210, label:'MVP1\nBaseline GNN', type:'transfer',
    date:'Mar 10 2026', status:'done',
    desc:'MVP1 released: повний pipeline від XES event log до GATv2 з MLflow tracking. Встановлює baseline метрики без структурного сигналу.',
    details:[
      'XES Adapter → DTO → PrefixSlice → GraphBuilder → BaselineGATv2/GCN',
      'MLflow 2.x logging, YAML config з include/merge, early stopping',
      'Checkpoint resume/retrain, encoder freeze',
      'drift_window_sliding evaluation',
      'Expanded eval metrics: OOS, ECE, top-k accuracy',
    ]
  },
  { id:'mvp2', line:'trunk', x:340, label:'MVP2\nBPMN Structure', type:'transfer',
    date:'Mar 11–13 2026', status:'done',
    desc:'MVP2: введення BPMN topology як додаткового структурного сигналу. Перший dual model, OOS метрики, visualize-graph tool.',
    details:[
      'BPMN topology ingestion з XML',
      'Перший dual model для structural path',
      'OOS (out-of-structure) метрики',
      'visualize-graph та visualize-topology CLI tools',
      'Documentation для MVP2 architecture',
    ]
  },
  { id:'mvp25', line:'trunk', x:470, label:'MVP2.5\nEOPKG Runtime', type:'xlarge',
    date:'Mar 14 – Apr 2026', status:'done',
    desc:'MVP2.5: повна offline/runtime separation через IKnowledgeGraphPort. EOPKGGATv2/GCN моделі. Neo4j, stats snapshots, graph cache.',
    details:[
      'Offline topology preparation: ingest-topology, sync-topology',
      'Offline stats snapshots: sync-stats, sync-stats-backfill',
      'IKnowledgeGraphPort: file / in_memory / neo4j backends',
      'EOPKGGATv2 та EOPKGGCN model families з structural tensors',
      'allowed_target_mask, structural_edge_index, struct_x, struct_node_to_class_index',
      'graph_dataset_cache + sharded disk spill',
      'ADRs: 0002 (offline separation), 0003 (immutable snapshots), 0004 (strict_asof)',
    ]
  },
  { id:'stage41', line:'trunk', x:640, label:'Stage 4.1\nAlignment Gate', type:'transfer',
    date:'Apr–May 2026', status:'done',
    desc:'Stats contract v1. Producer-side alignment gate. Topology projection diagnostics. Закриває головний research-grade debt по leakage.',
    details:[
      'Activity-topology alignment gate: research_strict profile',
      'stats_contract v1: producer quality gates, metadata.stats_index',
      'topology_projection diagnostics для collapse_for_prediction',
      'ADR-0006: research-grade alignment gate',
      'ADR-0007: topology projection alignment',
      'Snapshot-aware Neo4j payload caching',
      'Deduplicated structural payload shards',
    ]
  },
  { id:'stage42', line:'trunk', x:830, label:'Stage 4.2\nLearning Strategy', type:'xlarge',
    date:'May 2026', status:'done',
    desc:'Ключовий перехід: від ablation fusion modes до methodology навчання. EOPKGTopologyConditioned + candidate_id training. MLflow 3.',
    details:[
      'Upgrade до MLflow 3 (signature compatibility)',
      'topology_conditioned learning strategy',
      'wrong_version_topology negatives (known versions)',
      'drop_edges_same_version: local corruption',
      'version-weighted CE + rehearsal retention',
      'candidate_id training: set-aware CE над [B,C_v]',
      'topology-homogeneous DataLoader batching',
      'ADR-0008: CLI composition root boundary',
    ]
  },
  { id:'current', line:'trunk', x:990, label:'Zero-Shot\nValidation ★', type:'current',
    date:'Зараз', status:'current',
    desc:'Поточна позиція: topology_native candidate identity реалізовано. Задача: провести повну dissertation-grade валідацію zero-shot гіпотези.',
    details:[
      'candidate_identity_mode=topology_native реалізовано',
      'Фікс target alignment (candidate_ids + labels matching)',
      'candidate_id CE training готовий',
      'Потрібна: full eval_drift validation на loan dataset',
      'Dissertation claim: structural drift → measurable zero-shot benefit',
      'Next: semantic candidate grounding (P2 debt)',
    ]
  },
  { id:'future_zs', line:'trunk', x:1120, label:'versioned\nzero_shot', type:'future',
    date:'Майбутнє', status:'planned',
    desc:'experiment.mode=versioned_zero_shot: повна оркестрація zero-shot evaluation через versioned train/eval split.',
    details:[
      'Окремий experiment orchestration step',
      'known version train → unseen version eval',
      'Потрібний: candidate-native evaluation contract',
      'Потрібний: semantic/topological candidate prototypes',
    ]
  },

  /* ── FUSION ARCHITECTURE (red) — branches from mvp2 ──── */
  { id:'classmean', line:'fusion', x:400, label:'ClassMean\nAttn / Concat', type:'station',
    date:'Mar–Apr 2026', status:'done',
    desc:'Перший structural fusion для EOPKGGATv2: ClassMeanAttention та ClassMeanConcat. Backward-compatible, mean-pools structural node states by activity class.',
    details:[
      'ClassMeanAttention: attention-weighted mean pooling per class',
      'ClassMeanConcat: concatenation variant',
      'Backward-compatible з existing baseline path',
      'Baseline для порівняння більш складних fusion modes',
    ]
  },
  { id:'classaware', line:'fusion', x:510, label:'ClassAware\nStructuralScoring', type:'transfer',
    date:'May 7 2026', status:'done',
    desc:'Новий fusion: bilinear prefix→structure scorer. Structural identity embeddings + stats_projection(struct_x). Node-to-class LogSumExp pooling.',
    details:[
      'Bilinear prefix-to-structure scorer',
      'Structural identity enriched by stats_projection(struct_x) * structural_stats_beta',
      'Node-to-class LogSumExp pooling → class logits',
      'Per-sample LayerNorm + observed-logit scale alignment',
      'Optional structural auxiliary loss (structural_aux_loss_enabled)',
      'Trainer diagnostics: scale, gate, aux loss metrics',
    ]
  },
  { id:'structprior', line:'fusion', x:620, label:'Structural\nPriorEncoder', type:'station',
    date:'May 2026', status:'done',
    desc:'Еталон-like fusion mode: mean-pools structural GNN node states → struct_context. Fuses [obs_context || struct_context] перед classifier.',
    details:[
      'Keeps observed prefix encoder as primary path',
      'Mean-pools structural GNN → struct_context',
      'Supports structural_prior_fusion: concat / gated_concat',
      'Trainer diagnostics: context scale, gate open ratio',
    ]
  },
  { id:'structxattn', line:'fusion', x:730, label:'StructXAttn\nCross-Attention', type:'station',
    date:'May 2026', status:'done',
    desc:'Token-level structural cross-attention: observed IG nodes query structural GNN states перед pooling.',
    details:[
      'struct_xattn_layers: post_conv2 / after_each_conv',
      'Observed nodes query structural node states (cross-attention)',
      'Optional correct-vs-corrupted topology contrastive objective',
      'struct_xattn_merge_mode: pre_norm_context recommended for drift',
      'struct_xattn_delta_ratio_max cap проти LayerNorm amplification',
    ]
  },

  /* ── ABLATION / EXPERIMENTAL (orange) — branches from mvp25 */
  { id:'tse', line:'ablation', x:540, label:'TopologyState\nEncoder', type:'experimental',
    date:'May 14 2026', status:'experimental',
    desc:'Early/input-level fusion: prefix execution state projected onto structural nodes ПЕРЕД structural GNN. Ablation для fusion-level analysis.',
    details:[
      'Consumes struct_prefix_state_x [B, |V|, 6]',
      'Projects prefix execution state → structural nodes',
      'Runs structural GNN over enriched structural nodes',
      'Призначений як ablation, не canonical drift-transfer mechanism',
      'Потребує struct_prefix_state_x у cache (schema v4+)',
    ]
  },
  { id:'tsge', line:'ablation', x:660, label:'TopologyStateGraph\nEncoder', type:'experimental',
    date:'May 16 2026', status:'experimental',
    desc:'Article-like structural graph baseline: classifies structural graph context after message passing. Etalon для same-version topology usefulness.',
    details:[
      'Projects prefix state → structural nodes',
      'Runs structural message passing (GNN)',
      'Mean-pools topology node states → graph-level context',
      'Classifies structural graph context only',
      'Висновок: корисний як baseline, але не як final drift mechanism',
    ]
  },

  /* ── CANDIDATE CONTRACT (green) — branches from mvp25 ─── */
  { id:'fixedproj', line:'candidate', x:550, label:'fixed_projection\ncompat', type:'station',
    date:'May 2026', status:'done',
    desc:'Stage 2 compatibility path: candidate logits projected back до fixed [B, C_train] для сумісності з existing CE/mask/calibration/drift metrics.',
    details:[
      'EOPKGTopologyConditioned Stage 2 foundation',
      'forward_candidate() → CandidatePredictionOutput',
      'candidate_logits [B, C_v] → sparse fixed-label [B, C_train]',
      'CandidatePredictionOutput: node_logits, candidate_class_index, node_to_candidate_index',
      'Backward compatible з existing metric pipeline',
    ]
  },
  { id:'candidid', line:'candidate', x:680, label:'candidate_id\nTraining', type:'transfer',
    date:'May 25 2026', status:'done',
    desc:'True candidate-level training: set-aware CE над topology-local [B, C_v] candidate axis. Перша справжня кандидатна оптимізація.',
    details:[
      'Set-aware CE: всі candidate nodes mapped до target label = target set',
      'Topology-homogeneous DataLoader batching (process_version + stats_snapshot)',
      'candidate_target_in_candidate_set_rate, candidate_missing_target_rate diagnostics',
      'single_topology_required fail-fast guard',
      'topology_segments per shard → efficient batching без hydration',
      'Eval/drift: maps candidate probabilities back до global classes',
    ]
  },
  { id:'topnative', line:'candidate', x:830, label:'Topology-Native\nIdentity', type:'transfer',
    date:'May 26–28 2026', status:'done',
    desc:'candidate_identity_mode=topology_native: BPMN candidates поза train activity vocab зберігаються як valid prediction targets.',
    details:[
      'candidate_ids, candidate_labels, candidate_class_index per topology',
      'candidate_is_unseen flag для майбутніх BPMN нод',
      'struct_node_to_candidate_index, candidate_allowed_target_mask',
      'Фікс target alignment: match by candidate_ids AND candidate_labels',
      'candidate_class_index=-1 → fixed-label metric як compatibility diagnostic',
      'Дозволяє zero-shot без виключення нових activities через vocab OOV',
    ]
  },

  /* ── LEARNING METHODOLOGY (purple) — branches from stage41 */
  { id:'learnstd', line:'method', x:660, label:'standard\nstrategy', type:'station',
    date:'May 21 2026', status:'done',
    desc:'learning_strategy=standard: зберігає поточну поведінку тренера. Default baseline для порівняння з topology_conditioned.',
    details:[
      'Default behavior, без змін у trainer',
      'Базова лінія для controlled comparison',
      'Дозволяє точно ізолювати ефект topology_conditioned',
    ]
  },
  { id:'learntopo', line:'method', x:780, label:'topology_\nconditioned', type:'transfer',
    date:'May 21–25 2026', status:'done',
    desc:'topology_conditioned strategy: навчання явно тисне на модель враховувати topology через wrong-version negatives та structural corruption.',
    details:[
      'wrong_version_topology negatives (known versions only — no leakage)',
      'drop_edges_same_version: фізичне видалення structural edges',
      'train-time allowed-set loss',
      'version-weighted CE / rehearsal retention для старих версій',
      'Dissertation contribution: методологія навчання topology dependency',
    ]
  },
  { id:'futuremethod', line:'method', x:1070, label:'versioned_\nzero_shot', type:'future',
    date:'Майбутнє', status:'planned',
    desc:'experiment.mode=versioned_zero_shot: оркестрація evaluation де train = old versions, eval = new unseen version.',
    details:[
      'Requires candidate-native evaluation contract',
      'Separate experiment orchestration from current runner',
      'Version-aware train/eval split без leakage',
    ]
  },

  /* ── INFRASTRUCTURE (gray) — branches from mvp1 ─────── */
  { id:'xes_pipe', line:'infra', x:270, label:'XES→Graph\nPipeline', type:'station',
    date:'Mar 2026', status:'done',
    desc:'Базовий data pipeline: XES Adapter, DTOs, PrefixSlice, GraphBuilder, GraphTensorContract.',
    details:[
      'Streaming XES adapter з typed extras parsing',
      'PrefixSlice DTO, stateless PrefixPolicy',
      'GraphTensorContract, class-weighted training',
      'MLflow tracker adapter, YAML include loader',
    ]
  },
  { id:'neo4j_s', line:'infra', x:400, label:'Neo4j +\nStats Snapshots', type:'station',
    date:'Mar–Apr 2026', status:'done',
    desc:'Neo4j graph backend + immutable JSON stats snapshots. Camunda SQL adapter.',
    details:[
      'IKnowledgeGraphPort: file / in_memory / neo4j',
      'sync-stats + sync-stats-backfill tools',
      'immutable_json_asof snapshot policy',
      'Camunda SQL adapter для production logs',
      'add-version2xes, simulate-versioned-log tools',
    ]
  },
  { id:'cache_s', line:'infra', x:520, label:'Graph Cache\n+ Disk Spill', type:'station',
    date:'Mar–Apr 2026', status:'done',
    desc:'Graph dataset disk cache + sharded spill для великих запусків. max_ram_gb soft RSS guard.',
    details:[
      'graph_dataset_cache_policy: none / dto / full',
      'Sharded disk spill з deduplicated structural payloads',
      'max_ram_gb soft RSS guard → flush до shards',
      'Cache schema v4: drift metadata, struct_prefix_state_x',
    ]
  },
  { id:'shard_s', line:'infra', x:640, label:'Shard\nOptimization ×250', type:'station',
    date:'May 29 2026', status:'done',
    desc:'Критична оптимізація продуктивності: 250x speedup. Epoch time: ~60 годин → нормальний runtime.',
    details:[
      'max_cached_shards auto-scaling → кожен шард завантажується 1 раз/епоха',
      'O(1) attribute extraction із _slice_dict (без to_data_list())',
      'training.sharded_dataset_cached_shards config key',
      'topology_segments per shard entry для candidate-id batching',
    ]
  },
];

/* ── CONNECTIONS ─────────────────────────────────────────── */
// type: 'segment' (same lane), 'branch' (down to new lane), 'merge' (up to trunk/other)
// style: 'solid' | 'dashed'
const CONNECTIONS = [
  // Trunk
  { a:'hyp',       b:'mvp1',     style:'solid'  },
  { a:'mvp1',      b:'mvp2',     style:'solid'  },
  { a:'mvp2',      b:'mvp25',    style:'solid'  },
  { a:'mvp25',     b:'stage41',  style:'solid'  },
  { a:'stage41',   b:'stage42',  style:'solid'  },
  { a:'stage42',   b:'current',  style:'solid'  },
  { a:'current',   b:'future_zs',style:'dashed' },

  // Fusion — branch from mvp2, merge at stage42
  { a:'mvp2',      b:'classmean',  style:'solid' },
  { a:'classmean', b:'classaware', style:'solid' },
  { a:'classaware',b:'structprior',style:'solid' },
  { a:'structprior',b:'structxattn',style:'solid'},
  { a:'structxattn',b:'stage42',   style:'solid' },

  // Ablation — branch from mvp25, no merge
  { a:'mvp25',     b:'tse',        style:'dashed' },
  { a:'tse',       b:'tsge',       style:'dashed' },

  // Candidate — branch from mvp25, merge at current
  { a:'mvp25',     b:'fixedproj',  style:'solid' },
  { a:'fixedproj', b:'candidid',   style:'solid' },
  { a:'candidid',  b:'topnative',  style:'solid' },
  { a:'topnative', b:'current',    style:'solid' },

  // Methodology — branch from stage41, merge at stage42
  { a:'stage41',   b:'learnstd',   style:'solid' },
  { a:'learnstd',  b:'learntopo',  style:'solid' },
  { a:'learntopo', b:'stage42',    style:'solid' },
  { a:'current',   b:'futuremethod',style:'dashed'},

  // Infra — branch from mvp1, support line
  { a:'mvp1',      b:'xes_pipe',   style:'solid' },
  { a:'xes_pipe',  b:'neo4j_s',    style:'solid' },
  { a:'neo4j_s',   b:'cache_s',    style:'solid' },
  { a:'cache_s',   b:'shard_s',    style:'solid' },
];

/* ── TIME MARKERS ─────────────────────────────────────────── */
const TIME_MARKERS = [
  { x:80,   label:'Feb\n2026' },
  { x:210,  label:'Mar\n10' },
  { x:340,  label:'Mar\n11' },
  { x:470,  label:'Mar\n14+' },
  { x:640,  label:'Apr–\nMay' },
  { x:830,  label:'May\n21+' },
  { x:990,  label:'Now' },
  { x:1120, label:'Future' },
];

/* ============================================================
   LAYOUT CONSTANTS
   ============================================================ */
const SCALE    = 1.0;   // zoom factor
const LANE_H   = 90;    // px between lanes
const LANE_TOP = 80;    // y of first lane
const PAD_L    = 60;
const PAD_R    = 120;
const PAD_TOP  = 40;
const PAD_BOT  = 120;

const LANE_LABELS = [
  'Dissertation Trunk',
  'Fusion Architecture',
  'Ablation / Experimental',
  'Candidate Contract',
  'Learning Methodology',
  'Infrastructure',
];

/* ============================================================
   UTILS
   ============================================================ */
const lineByIdMap = {};
LINES_DEF.forEach(l => lineByIdMap[l.id] = l);

const stationMap = {};
STATIONS.forEach(s => stationMap[s.id] = s);

function laneY(lane) { return PAD_TOP + LANE_TOP + lane * LANE_H; }
function stationX(s)  { return PAD_L + s.x; }
function stationY(s)  { return laneY(LINES_DEF.find(l=>l.id===s.line).lane); }

function stationRadius(s) {
  if (s.type==='xlarge')  return 14;
  if (s.type==='origin')  return 12;
  if (s.type==='transfer' || s.type==='current') return 11;
  if (s.type==='experimental' || s.type==='future') return 7;
  return 8;
}

/* ============================================================
   SVG BUILDER
   ============================================================ */
const NS = 'http://www.w3.org/2000/svg';
function el(tag, attrs={}) {
  const e = document.createElementNS(NS, tag);
  for (const [k,v] of Object.entries(attrs)) e.setAttribute(k,v);
  return e;
}

/* ============================================================
   STATE
   ============================================================ */
const lineVisible = {};
LINES_DEF.forEach(l => lineVisible[l.id] = true);

/* ============================================================
   RENDER
   ============================================================ */
function render() {
  const svg = document.getElementById('metro-svg');
  svg.innerHTML = '';

  // Canvas size
  const maxX = Math.max(...STATIONS.map(stationX)) + PAD_R;
  const maxY = laneY(LINES_DEF.length - 1) + LANE_H;
  const W = maxX;
  const H = maxY + PAD_BOT;

  svg.setAttribute('width', W);
  svg.setAttribute('height', H);
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);

  const canvas = document.getElementById('map-canvas');
  canvas.style.width  = W + 'px';
  canvas.style.height = H + 'px';

  // Remove old hit areas and labels
  canvas.querySelectorAll('.station-hit,.station-label,.time-label,.lane-label').forEach(e=>e.remove());

  // ── Background ──
  const bg = el('rect', {x:0, y:0, width:W, height:H, fill:'#0d1117'});
  svg.appendChild(bg);

  // ── Grid lines ──
  LINES_DEF.forEach((line, i) => {
    const y = laneY(i);
    const g = el('line', { x1:PAD_L, y1:y, x2:W-20, y2:y, stroke:'#21262d', 'stroke-width':'1', 'stroke-dasharray':'4 6' });
    svg.appendChild(g);
  });

  // ── Lane labels (left side) ──
  LINES_DEF.forEach((line, i) => {
    if (!lineVisible[line.id]) return;
    const y = laneY(i);
    const div = document.createElement('div');
    div.className = 'lane-label';
    div.style.cssText = `top:${y}px;right:${W - PAD_L + 8}px;color:${line.color};opacity:.7`;
    div.textContent = line.shortName || line.name;
    canvas.appendChild(div);
  });

  // ── Time markers ──
  TIME_MARKERS.forEach(m => {
    const x = PAD_L + m.x;
    const lineEl = el('line', { x1:x, y1:PAD_TOP+10, x2:x, y2:H-PAD_BOT+30,
      stroke:'#21262d', 'stroke-width':'1', 'stroke-dasharray':'2 8' });
    svg.appendChild(lineEl);
    const lines = m.label.split('\n');
    lines.forEach((line, j) => {
      const div = document.createElement('div');
      div.className = 'time-label';
      div.style.cssText = `left:${x}px;top:${H-PAD_BOT+36+j*13}px`;
      div.textContent = line;
      canvas.appendChild(div);
    });
  });

  // ── Connections ──
  CONNECTIONS.forEach(conn => {
    const a = stationMap[conn.a];
    const b = stationMap[conn.b];
    if (!a || !b) return;

    const lineA = lineByIdMap[a.line];
    const lineB = lineByIdMap[b.line];
    const color = lineA.color; // use source line color

    // visibility: hide if BOTH endpoints' lines are hidden
    if (!lineVisible[a.line] && !lineVisible[b.line]) return;
    const opacity = (lineVisible[a.line] && lineVisible[b.line]) ? 1 : 0.15;

    const x1 = stationX(a), y1 = stationY(a);
    const x2 = stationX(b), y2 = stationY(b);
    const r  = stationRadius(a);
    const r2 = stationRadius(b);

    let pathD;
    const dx = x2 - x1, dy = y2 - y1;
    const startX = x1 + (dx > 0 ? r : -r) * 0.6;
    const endX   = x2 - (dx > 0 ? r2 : -r2) * 0.6;

    if (Math.abs(dy) < 2) {
      // same lane — straight line
      pathD = `M ${startX} ${y1} L ${endX} ${y2}`;
    } else {
      // cross-lane — bezier
      const cx1 = startX + (endX-startX)*0.35;
      const cx2 = endX   - (endX-startX)*0.35;
      pathD = `M ${startX} ${y1} C ${cx1} ${y1} ${cx2} ${y2} ${endX} ${y2}`;
    }

    const strokeW = (a.type==='xlarge'||b.type==='xlarge') ? 3.5 : 2.5;

    // glow
    const glowPath = el('path', {
      d: pathD, fill:'none',
      stroke: color, 'stroke-width': strokeW+6,
      'stroke-dasharray': conn.style==='dashed' ? '8 6' : 'none',
      opacity: 0.15 * opacity, 'stroke-linecap': 'round'
    });
    svg.appendChild(glowPath);

    // main path
    const mainPath = el('path', {
      d: pathD, fill:'none',
      stroke: color, 'stroke-width': strokeW,
      'stroke-dasharray': conn.style==='dashed' ? '8 6' : 'none',
      opacity: opacity * (conn.style==='dashed' ? 0.55 : 1),
      'stroke-linecap': 'round'
    });
    svg.appendChild(mainPath);
  });

  // ── Stations ──
  STATIONS.forEach(s => {
    if (!lineVisible[s.line]) return;
    const line = lineByIdMap[s.line];
    const x = stationX(s), y = stationY(s);
    const r = stationRadius(s);

    // glow ring for important stations
    if (['transfer','xlarge','origin','current'].includes(s.type)) {
      const glow = el('circle', { cx:x, cy:y, r:r+6, fill:`${line.color}20` });
      svg.appendChild(glow);
    }

    // outer ring (for transfer nodes)
    if (['transfer','xlarge'].includes(s.type)) {
      const ring = el('circle', { cx:x, cy:y, r:r+3, fill:'none', stroke:line.color, 'stroke-width':'1.5', opacity:'0.5' });
      svg.appendChild(ring);
    }

    // main circle
    let fillColor, strokeColor, strokeW;
    if (s.type === 'current') {
      fillColor = line.color; strokeColor = '#fff'; strokeW = 2.5;
    } else if (s.type === 'future') {
      fillColor = '#0d1117'; strokeColor = line.color; strokeW = 1.5;
    } else if (s.type === 'experimental') {
      fillColor = `${line.color}40`; strokeColor = line.color; strokeW = 1.5;
    } else if (['transfer','xlarge','origin'].includes(s.type)) {
      fillColor = '#0d1117'; strokeColor = line.color; strokeW = 2.5;
    } else {
      fillColor = line.color; strokeColor = 'none'; strokeW = 0;
    }

    const circle = el('circle', {
      cx:x, cy:y, r,
      fill: fillColor, stroke: strokeColor, 'stroke-width': strokeW
    });
    svg.appendChild(circle);

    // Inner dot for hollow circles
    if (['transfer','xlarge','origin'].includes(s.type)) {
      const inner = el('circle', { cx:x, cy:y, r:Math.max(2,r-5), fill:line.color });
      svg.appendChild(inner);
    }

    // Current star
    if (s.type === 'current') {
      const star = el('text', { x, y:y+4, 'text-anchor':'middle', 'font-size':'10',
        fill:'#0d1117', 'font-weight':'700', 'pointer-events':'none' });
      star.textContent = '★';
      svg.appendChild(star);
    }

    // Experimental: dashed border
    if (s.type === 'experimental') {
      const dash = el('circle', { cx:x, cy:y, r:r+4, fill:'none',
        stroke:line.color, 'stroke-width':'1', 'stroke-dasharray':'3 3', opacity:'0.6' });
      svg.appendChild(dash);
    }

    // ── Label ──
    const labelLines = s.label.split('\n');
    const LABEL_OFFSET = r + 10;
    const isBottom = [1,2,3,4,5].includes(line.lane); // lanes below trunk go below
    const labelTop = isBottom ? y + LABEL_OFFSET : y - LABEL_OFFSET - labelLines.length * 14;

    const div = document.createElement('div');
    div.className = 'station-label';
    const lw = 90;
    div.style.cssText = `left:${x}px;top:${labelTop}px;width:${lw}px;margin-left:${-lw/2}px`;
    div.style.color = line.color;
    if (s.type==='future' || s.type==='experimental') div.style.opacity='0.55';

    labelLines.forEach((ln, i) => {
      const span = document.createElement('div');
      span.textContent = ln;
      if (i===0) span.style.fontWeight='600';
      div.appendChild(span);
    });
    canvas.appendChild(div);

    // ── Hit area ──
    const hitR = Math.max(22, r + 10);
    const hit = document.createElement('div');
    hit.className = 'station-hit';
    hit.style.cssText = `left:${x}px;top:${y}px;width:${hitR*2}px;height:${hitR*2}px;
      margin-left:${-hitR}px;margin-top:${-hitR}px;`;
    hit.dataset.stationId = s.id;
    hit.addEventListener('mouseenter', e => showTooltip(s, e));
    hit.addEventListener('mouseleave', hideTooltip);
    hit.addEventListener('click', () => showDetail(s.id));
    canvas.appendChild(hit);
  });

  // ── Pulse animation for current station ──
  const cur = STATIONS.find(s=>s.type==='current');
  if (cur) {
    const pulse = el('circle', {
      cx: stationX(cur), cy: stationY(cur), r: stationRadius(cur)+4,
      fill:'none', stroke: lineByIdMap[cur.line].color, 'stroke-width':'2', opacity:'0'
    });
    svg.appendChild(pulse);

    // CSS pulse animation via animateTransform
    const animate = el('animate', {
      attributeName:'r', from:stationRadius(cur)+4, to:stationRadius(cur)+16,
      dur:'2s', repeatCount:'indefinite'
    });
    const animateOp = el('animate', {
      attributeName:'opacity', from:'0.6', to:'0',
      dur:'2s', repeatCount:'indefinite'
    });
    pulse.appendChild(animate);
    pulse.appendChild(animateOp);
  }
}

/* ============================================================
   LINE TOGGLES
   ============================================================ */
function buildToggles() {
  const container = document.getElementById('line-toggles');
  LINES_DEF.forEach(line => {
    const btn = document.createElement('div');
    btn.className = 'lt';
    btn.style.borderColor = line.color;
    btn.style.color = line.color;
    btn.innerHTML = `<div class="lt-dot" style="background:${line.color}"></div>${line.shortName}`;
    btn.dataset.lineId = line.id;
    btn.addEventListener('click', () => {
      lineVisible[line.id] = !lineVisible[line.id];
      btn.classList.toggle('off', !lineVisible[line.id]);
      render();
    });
    container.appendChild(btn);
  });
}

/* ============================================================
   TOOLTIP
   ============================================================ */
const tooltip = document.getElementById('tooltip');
function showTooltip(s, e) {
  tooltip.textContent = s.label.replace(/\n/,' · ');
  tooltip.style.opacity = '1';
  posTooltip(e);
}
function posTooltip(e) {
  tooltip.style.left = (e.clientX + 12) + 'px';
  tooltip.style.top  = (e.clientY - 8) + 'px';
}
function hideTooltip() { tooltip.style.opacity = '0'; }
document.addEventListener('mousemove', e => {
  if (tooltip.style.opacity==='1') posTooltip(e);
});

/* ============================================================
   DETAIL PANEL
   ============================================================ */
const STATUS_STYLE = {
  done:         { bg:'rgba(63,185,80,.15)', color:'#3fb950',  label:'✅ Реалізовано' },
  current:      { bg:'rgba(88,166,255,.2)', color:'#58a6ff', label:'★ Зараз' },
  experimental: { bg:'rgba(227,179,65,.15)',color:'#e3b341',  label:'🔬 Experimental' },
  planned:      { bg:'rgba(188,140,255,.15)',color:'#bc8cff', label:'🗓 Заплановано' },
};

function showDetail(id) {
  const s    = stationMap[id];
  const line = lineByIdMap[s.line];
  const st   = STATUS_STYLE[s.status] || STATUS_STYLE.done;

  const tag = document.getElementById('dp-tag');
  tag.textContent  = line.name;
  tag.style.background = `${line.color}25`;
  tag.style.color      = line.color;
  tag.style.border     = `1px solid ${line.color}60`;

  document.getElementById('dp-title').textContent  = s.label.replace(/\n/,' ');
  document.getElementById('dp-date').textContent   = s.date;
  document.getElementById('dp-desc').textContent   = s.desc;

  const statusEl = document.getElementById('dp-status');
  statusEl.textContent       = st.label;
  statusEl.style.background  = st.bg;
  statusEl.style.color       = st.color;
  statusEl.style.border      = `1px solid ${st.color}60`;

  const detailsEl = document.getElementById('dp-details');
  detailsEl.innerHTML = (s.details||[]).map(d =>
    `<div class="dp-detail-item">${d}</div>`
  ).join('');

  document.getElementById('detail-panel').classList.add('open');
}

function closeDetail() {
  document.getElementById('detail-panel').classList.remove('open');
}

/* ============================================================
   DRAG TO PAN
   ============================================================ */
let dragging=false, startX=0, startY=0, scrollX=0, scrollY=0;
const outer = document.getElementById('map-outer');
outer.addEventListener('mousedown', e => {
  if (e.target.classList.contains('station-hit')) return;
  dragging=true; startX=e.clientX; startY=e.clientY;
  scrollX=outer.scrollLeft; scrollY=outer.scrollTop;
  outer.classList.add('dragging');
});
document.addEventListener('mousemove', e => {
  if (!dragging) return;
  outer.scrollLeft = scrollX - (e.clientX-startX);
  outer.scrollTop  = scrollY - (e.clientY-startY);
});
document.addEventListener('mouseup', () => { dragging=false; outer.classList.remove('dragging'); });

/* ============================================================
   INIT
   ============================================================ */
buildToggles();
render();

// Scroll to show start
outer.scrollLeft = 0;
outer.scrollTop  = 0;
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main Dashboard HTML (compact version with link to map)
# ---------------------------------------------------------------------------

DASH_HTML = r"""<!DOCTYPE html>
<html lang="uk">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>BPM Prediction Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
:root {
  --bg:#0d1117;--surface:#161b22;--surface2:#1c2128;--surface3:#21262d;
  --border:#30363d;--border-s:#21262d;
  --text:#e6edf3;--muted:#8b949e;--dim:#484f58;
  --blue:#58a6ff;--blue-g:rgba(88,166,255,.15);
  --green:#3fb950;--green-g:rgba(63,185,80,.15);
  --yellow:#d29922;--yellow-g:rgba(210,153,34,.15);
  --red:#f85149;--red-g:rgba(248,81,73,.15);
  --purple:#bc8cff;--purple-g:rgba(188,140,255,.12);
  --r:10px;
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);min-height:100vh}

.hdr{background:linear-gradient(135deg,#0d1117,#161b22);border-bottom:1px solid var(--border);
  padding:14px 24px;position:sticky;top:0;z-index:100;display:flex;align-items:center;gap:14px}
.hdr-logo{width:34px;height:34px;background:linear-gradient(135deg,#58a6ff,#bc8cff);
  border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:16px}
.hdr-info{flex:1}
.hdr-title{font-size:15px;font-weight:700}
.hdr-sub{font-size:10px;color:var(--muted);margin-top:1px}
.hdr-badges{display:flex;gap:5px}
.badge{font-size:9px;font-weight:600;padding:2px 8px;border-radius:20px;border:1px solid}
.b-blue{background:var(--blue-g);border-color:var(--blue);color:var(--blue)}
.b-green{background:var(--green-g);border-color:var(--green);color:var(--green)}
.b-purple{background:var(--purple-g);border-color:var(--purple);color:var(--purple)}
.map-btn{
  background:linear-gradient(135deg,#58a6ff18,#bc8cff18);
  border:1px solid var(--purple);color:var(--purple);
  padding:7px 16px;border-radius:6px;cursor:pointer;font-size:12px;font-weight:600;
  font-family:inherit;text-decoration:none;display:flex;align-items:center;gap:6px;
  transition:all .2s;
}
.map-btn:hover{background:linear-gradient(135deg,#58a6ff30,#bc8cff30);transform:translateY(-1px)}
.ref-btn{background:var(--surface2);border:1px solid var(--border);color:var(--muted);
  padding:5px 11px;border-radius:5px;cursor:pointer;font-size:11px;font-family:inherit;transition:all .2s}
.ref-btn:hover{color:var(--text);border-color:var(--blue);background:var(--blue-g)}
.last-ref{font-size:10px;color:var(--dim)}

.layout{display:grid;grid-template-columns:240px 1fr;min-height:calc(100vh - 64px)}
.sidebar{background:var(--surface);border-right:1px solid var(--border);padding:14px 0;
  position:sticky;top:64px;height:calc(100vh - 64px);overflow-y:auto}
.sb-label{font-size:9px;font-weight:600;letter-spacing:1px;color:var(--dim);text-transform:uppercase;
  padding:10px 14px 4px}
.sb-item{display:flex;align-items:center;gap:8px;padding:6px 12px;border-radius:5px;cursor:pointer;
  color:var(--muted);font-size:12px;font-weight:500;transition:all .15s;border:1px solid transparent;
  margin:1px 7px;user-select:none}
.sb-item:hover{background:var(--surface2);color:var(--text);border-color:var(--border-s)}
.sb-item.active{background:var(--blue-g);color:var(--blue);border-color:var(--blue)}
.sb-item .ico{font-size:13px;width:17px;text-align:center}
.sb-item .cnt{margin-left:auto;font-size:9px;font-weight:700;background:var(--surface3);
  padding:1px 5px;border-radius:7px;color:var(--muted)}
.sb-map-link{display:flex;align-items:center;gap:8px;padding:8px 12px;border-radius:5px;
  margin:4px 7px;background:linear-gradient(135deg,var(--blue-g),var(--purple-g));
  border:1px solid var(--purple);color:var(--purple);font-size:12px;font-weight:600;
  text-decoration:none;transition:all .2s}
.sb-map-link:hover{filter:brightness(1.15)}

.main{padding:22px 24px;overflow-y:auto}
.section{display:none}.section.active{display:block}
.sec-title{font-size:19px;font-weight:700;margin-bottom:4px;
  background:linear-gradient(90deg,var(--text),var(--muted));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.sec-desc{font-size:11px;color:var(--muted);margin-bottom:18px}

.stats-bar{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:20px}
.sp{background:var(--surface);border:1px solid var(--border);border-radius:7px;
  padding:10px 16px;display:flex;flex-direction:column;gap:3px;min-width:90px}
.sv{font-size:24px;font-weight:700}
.sl{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.4px}
.c-gr{color:var(--green)}.c-rd{color:var(--red)}.c-bl{color:var(--blue)}
.c-yw{color:var(--yellow)}.c-pp{color:var(--purple)}

.cards-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px;margin-bottom:20px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);
  padding:14px;transition:all .2s;cursor:pointer;position:relative;overflow:hidden}
.card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;border-radius:var(--r) var(--r) 0 0}
.card.c-gr::before{background:linear-gradient(90deg,var(--green),#2ea043)}
.card.c-yw::before{background:linear-gradient(90deg,var(--yellow),#e3b341)}
.card.c-rd::before{background:linear-gradient(90deg,var(--red),#da3633)}
.card.c-bl::before{background:linear-gradient(90deg,var(--blue),#388bfd)}
.card.c-pp::before{background:linear-gradient(90deg,var(--purple),#a371f7)}
.card.c-gy::before{background:linear-gradient(90deg,var(--border),#484f58)}
.card:hover{border-color:var(--blue);transform:translateY(-2px);box-shadow:0 6px 18px rgba(0,0,0,.4)}
.card-hdr{display:flex;align-items:flex-start;gap:7px;margin-bottom:7px}
.card-ico{font-size:15px;margin-top:1px}
.card-name{font-size:11px;font-weight:600;flex:1}
.cs{font-size:9px;font-weight:600;padding:2px 6px;border-radius:8px;white-space:nowrap}
.st-implemented{background:var(--green-g);color:var(--green);border:1px solid var(--green)}
.st-experimental{background:var(--yellow-g);color:var(--yellow);border:1px solid var(--yellow)}
.st-planned{background:var(--blue-g);color:var(--blue);border:1px solid var(--blue)}
.st-active{background:var(--red-g);color:var(--red);border:1px solid var(--red)}
.st-closed{background:var(--green-g);color:var(--green);border:1px solid var(--green)}
.st-deferred{background:var(--surface3);color:var(--muted);border:1px solid var(--border)}
.st-proposed{background:var(--purple-g);color:var(--purple);border:1px solid var(--purple)}
.st-accepted{background:var(--green-g);color:var(--green);border:1px solid var(--green)}
.st-unknown{background:var(--surface3);color:var(--dim);border:1px solid var(--border)}
.card-body{font-size:10px;color:var(--muted);line-height:1.6}
.card-kv{font-size:9px;margin-top:7px}
.kv-row{display:flex;gap:5px;margin-bottom:2px}
.kv-k{color:var(--dim);min-width:80px;text-transform:uppercase;letter-spacing:.3px;font-size:8px}
.kv-v{color:var(--muted);font-family:'JetBrains Mono',monospace;font-size:9px;flex:1}

.tbl-wrap{overflow-x:auto;border-radius:var(--r);border:1px solid var(--border);margin-bottom:20px}
table{width:100%;border-collapse:collapse}
th{background:var(--surface2);text-align:left;padding:8px 11px;font-size:9px;font-weight:600;
  color:var(--dim);text-transform:uppercase;letter-spacing:.3px;border-bottom:1px solid var(--border)}
td{padding:9px 11px;font-size:10px;color:var(--muted);border-bottom:1px solid var(--border-s);vertical-align:top}
tr:last-child td{border-bottom:none}
tr:hover td{background:var(--surface2);cursor:pointer}
td.mono{font-family:'JetBrains Mono',monospace;font-size:9px;color:var(--text)}

.p-section{margin-bottom:20px}
.p-hdr{display:flex;align-items:center;gap:8px;margin-bottom:10px;
  padding-bottom:6px;border-bottom:1px solid var(--border)}
.p-title{font-size:13px;font-weight:700}
.p0{color:var(--red)}.p1{color:var(--yellow)}.p2{color:var(--muted)}

.tl{position:relative;margin-bottom:20px}
.tl::before{content:'';position:absolute;left:16px;top:0;bottom:0;width:2px;
  background:linear-gradient(to bottom,var(--blue),var(--purple),var(--border))}
.tl-item{display:flex;gap:16px;margin-bottom:16px}
.tl-dot{width:34px;height:34px;flex-shrink:0;background:var(--surface2);border:2px solid var(--blue);
  border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:13px;z-index:1}
.tl-content{flex:1;background:var(--surface);border:1px solid var(--border);border-radius:7px;padding:10px}
.tl-title{font-size:11px;font-weight:600;margin-bottom:3px}
.tl-body{font-size:10px;color:var(--muted);line-height:1.6}

.overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.78);z-index:500;
  align-items:center;justify-content:center;padding:14px}
.overlay.open{display:flex}
.modal{background:var(--surface);border:1px solid var(--border);border-radius:11px;
  max-width:620px;width:100%;max-height:82vh;overflow-y:auto;
  box-shadow:0 20px 56px rgba(0,0,0,.6);animation:mi .17s ease-out}
@keyframes mi{from{opacity:0;transform:scale(.95)}to{opacity:1;transform:scale(1)}}
.m-hdr{padding:16px 20px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:10px;
  position:sticky;top:0;background:var(--surface);border-radius:11px 11px 0 0}
.m-ico{font-size:20px}
.m-info{flex:1}
.m-title{font-size:14px;font-weight:700}
.m-sub{font-size:10px;color:var(--muted);margin-top:1px}
.m-close{background:none;border:none;color:var(--muted);cursor:pointer;font-size:17px;transition:color .15s}
.m-close:hover{color:var(--text)}
.m-body{padding:16px 20px}
.m-sec{margin-bottom:14px}
.m-sec-t{font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:.6px;
  color:var(--dim);margin-bottom:5px;padding-bottom:4px;border-bottom:1px solid var(--border-s)}
.m-text{font-size:11px;color:var(--muted);line-height:1.75;white-space:pre-wrap}
.m-tag{display:inline-block;padding:1px 6px;border-radius:3px;font-size:9px;
  font-family:'JetBrains Mono',monospace;background:var(--surface3);color:var(--blue);margin:1px}

::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
.empty{color:var(--muted);font-size:11px;padding:12px 0}
.div{height:1px;background:var(--border);margin:16px 0}
</style>
</head>
<body>
<div class="hdr">
  <div class="hdr-logo">&#x1F9E0;</div>
  <div class="hdr-info">
    <div class="hdr-title">BPM Prediction Dashboard</div>
    <div class="hdr-sub" id="phase-label">Loading...</div>
  </div>
  <div class="hdr-badges" id="hdr-badges"></div>
  <span class="last-ref" id="last-ref"></span>
  <a class="map-btn" href="/map" target="_blank">&#x1F5FA;&#xFE0F; Research Map</a>
  <button class="ref-btn" onclick="loadData()">&#8635; Refresh</button>
</div>
<div class="layout">
<nav class="sidebar">
  <a class="sb-map-link" href="/map" target="_blank">&#x1F9ED; Research Journey Map</a>
  <div class="sb-label">Status</div>
  <div class="sb-item active" onclick="nav('ov',this)"><span class="ico">&#x1F4CA;</span>Overview</div>
  <div class="sb-item" onclick="nav('upd',this)"><span class="ico">&#x26A1;</span>Updates</div>
  <div class="sb-label">Tech</div>
  <div class="sb-item" onclick="nav('caps',this)"><span class="ico">&#x2705;</span>Capabilities<span class="cnt" id="cnt-caps">—</span></div>
  <div class="sb-item" onclick="nav('debt',this)"><span class="ico">&#x26A0;&#xFE0F;</span>Tech Debt<span class="cnt" id="cnt-debt">—</span></div>
  <div class="sb-item" onclick="nav('adrs',this)"><span class="ico">&#x1F4CB;</span>ADRs<span class="cnt" id="cnt-adrs">—</span></div>
  <div class="sb-item" onclick="nav('inv',this)"><span class="ico">&#x1F512;</span>Invariants</div>
</nav>
<main class="main">

<div class="section active" id="section-ov">
  <div class="sec-title">Project Overview</div>
  <div class="sec-desc">MVP2.5 Stage 4.2 · Topology-Conditioned Zero-Shot BPM Prediction</div>
  <div class="stats-bar" id="stats-bar"></div>
  <div id="prios"></div>
</div>

<div class="section" id="section-upd">
  <div class="sec-title">Recent Updates</div>
  <div class="sec-desc">Останні зміни runtime та архітектури.</div>
  <div class="tl" id="timeline"></div>
</div>

<div class="section" id="section-caps">
  <div class="sec-title">Capabilities</div>
  <div class="sec-desc">Реалізовані можливості MVP2.5.</div>
  <div class="cards-grid" id="caps-grid"></div>
</div>

<div class="section" id="section-debt">
  <div class="sec-title">Tech Debt</div>
  <div class="sec-desc">Активний техборг. Клік для деталей.</div>
  <div id="debt-root"></div>
</div>

<div class="section" id="section-adrs">
  <div class="sec-title">ADRs</div>
  <div class="sec-desc">Architecture Decision Records.</div>
  <div class="tbl-wrap">
    <table><thead><tr><th>ID</th><th>Title</th><th>Status</th><th>Date</th><th>Decision</th></tr></thead>
    <tbody id="adr-tbody"></tbody></table>
  </div>
</div>

<div class="section" id="section-inv">
  <div class="sec-title">Runtime Invariants</div>
  <div class="sec-desc">Незмінні архітектурні правила.</div>
  <div class="cards-grid" id="inv-grid"></div>
</div>

</main>
</div>

<div class="overlay" id="overlay" onclick="cm(event)">
  <div class="modal">
    <div class="m-hdr">
      <span class="m-ico" id="m-ico"></span>
      <div class="m-info"><div class="m-title" id="m-title"></div><div class="m-sub" id="m-sub"></div></div>
      <button class="m-close" onclick="cm()">&#x2715;</button>
    </div>
    <div class="m-body" id="m-body"></div>
  </div>
</div>

<script>
let DATA=null;
const MODALS={};let mid=0;
function rm(ico,title,sub,secs){const id='m'+(++mid);MODALS[id]={ico,title,sub,secs};return id;}
function nav(id,el){
  document.querySelectorAll('.section').forEach(s=>s.classList.remove('active'));
  document.querySelectorAll('.sb-item').forEach(s=>s.classList.remove('active'));
  document.getElementById('section-'+id).classList.add('active');
  el.classList.add('active');
}
function om(id){
  const m=MODALS[id];if(!m)return;
  document.getElementById('m-ico').textContent=m.ico;
  document.getElementById('m-title').textContent=m.title;
  document.getElementById('m-sub').textContent=m.sub;
  let h='';
  for(const s of m.secs) h+=`<div class="m-sec"><div class="m-sec-t">${esc(s.t)}</div><div class="m-text">${lmd(s.b)}</div></div>`;
  document.getElementById('m-body').innerHTML=h;
  document.getElementById('overlay').classList.add('open');
}
function cm(e){if(!e||e.target===document.getElementById('overlay'))document.getElementById('overlay').classList.remove('open');}
document.addEventListener('keydown',e=>{if(e.key==='Escape')cm();});

function esc(s){return String(s??'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function lmd(s){return esc(s).replace(/`([^`]+)`/g,'<span class="m-tag">$1</span>').replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>');}

const SI={implemented:'✅',experimental:'🔬',planned:'🗓️',active:'🔴',closed:'✅',deferred:'⏸️',proposed:'💡',accepted:'✅',unknown:'❓'};
const SC={implemented:'c-gr',experimental:'c-yw',planned:'c-bl',active:'c-rd',closed:'c-gr',deferred:'c-gy',proposed:'c-pp',accepted:'c-gr',unknown:'c-gy'};
function sb(s){return `<span class="cs st-${s}">${SI[s]||'·'} ${s}</span>`;}

function renderHeader(){
  const meta=DATA.state.meta||{};
  document.getElementById('phase-label').textContent=`${meta.active_phase||''} · ${meta.status||''} · updated: ${meta.last_updated||''}`;
  document.getElementById('last-ref').textContent='docs: '+(DATA.last_refreshed||'');
  const badges=[{l:'MVP2.5',c:'b-blue'},{l:meta.active_phase||'',c:'b-blue'},{l:meta.runtime_status||'',c:meta.runtime_status==='implemented'?'b-green':'b-yellow'},{l:'zero-shot hypothesis',c:'b-purple'}];
  document.getElementById('hdr-badges').innerHTML=badges.filter(b=>b.l).map(b=>`<span class="badge ${b.c}">${esc(b.l)}</span>`).join('');
}

function renderStats(){
  const caps=DATA.state.capabilities||[];
  const debt=DATA.debt.items||[];
  const adrs=DATA.adrs||[];
  const impl=caps.filter(c=>c.status==='implemented').length;
  const actD=debt.filter(d=>d.status==='active').length;
  const clD=debt.filter(d=>d.status==='closed').length;
  const accA=adrs.filter(a=>a.status.toLowerCase()==='accepted').length;
  document.getElementById('stats-bar').innerHTML=`
    <div class="sp"><div class="sv c-gr">${impl}</div><div class="sl">Capabilities</div></div>
    <div class="sp"><div class="sv c-rd">${actD}</div><div class="sl">Active Debt</div></div>
    <div class="sp"><div class="sv c-gr">${clD}</div><div class="sl">Closed Debt</div></div>
    <div class="sp"><div class="sv c-bl">${accA}/${adrs.length}</div><div class="sl">ADRs Accepted</div></div>
  `;
  document.getElementById('cnt-caps').textContent=impl;
  document.getElementById('cnt-debt').textContent=actD;
  document.getElementById('cnt-adrs').textContent=adrs.length;

  const prios=DATA.state.priorities||[];
  let ph=`<div style="background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:14px;margin-bottom:16px">
    <div style="font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.5px;color:var(--dim);margin-bottom:10px">Current Priorities</div>
    ${prios.map((p,i)=>`<div style="display:flex;gap:8px;margin-bottom:6px;font-size:11px;color:var(--muted)">
      <span style="color:var(--blue);font-weight:700">${i+1}.</span><span>${esc(p)}</span></div>`).join('')}
  </div>`;
  document.getElementById('prios').innerHTML=ph;
}

function renderUpdates(){
  const upds=DATA.state.recent_updates||[];
  const icos=['⚡','🔧','🚀','🛠️','🔄'];const cols=['var(--blue)','var(--green)','var(--purple)','var(--yellow)','var(--teal)'];
  let h='';
  upds.slice().reverse().forEach((u,i)=>{
    h+=`<div class="tl-item"><div class="tl-dot" style="border-color:${cols[i%cols.length]}">${icos[i%icos.length]}</div>
    <div class="tl-content"><div class="tl-title">${esc(u.title)}</div>
    <div class="tl-body">${esc(u.body.slice(0,400))}${u.body.length>400?'…':''}</div></div></div>`;
  });
  document.getElementById('timeline').innerHTML=h||'<div class="empty">No updates.</div>';
}

function renderCaps(){
  const caps=DATA.state.capabilities||[];
  let h='';
  for(const c of caps){
    const mid=rm(SI[c.status]||'·',c.name,'status: '+c.status,[{t:'Description',b:c.body}]);
    h+=`<div class="card ${SC[c.status]||'c-gy'}" onclick="om('${mid}')">
      <div class="card-hdr"><span class="card-ico">${SI[c.status]||'·'}</span><span class="card-name">${esc(c.name)}</span>${sb(c.status)}</div>
      <div class="card-body">${esc(c.body.slice(0,160))}${c.body.length>160?'…':''}</div></div>`;
  }
  document.getElementById('caps-grid').innerHTML=h||'<div class="empty">No caps.</div>';
}

function renderDebt(){
  const items=DATA.debt.items||[];
  const groups={};for(const d of items)(groups[d.priority]=groups[d.priority]||[]).push(d);
  let h='';
  for(const p of['P0','P1','P2']){
    const list=groups[p]||[];if(!list.length)continue;
    const ac=list.filter(x=>x.status==='active').length;
    h+=`<div class="p-section"><div class="p-hdr"><div class="p-title ${p.toLowerCase()}">${p}</div>
      <span style="font-size:10px;color:var(--muted)">${ac} active / ${list.length}</span></div>
      <div class="cards-grid">`;
    for(const d of list){
      const mid=rm('⚠️',d.name,d.priority+' · '+d.status,[
        {t:'Current behavior',b:d.current_behavior||'—'},
        {t:'Target state',b:d.target_state||'—'},
        {t:'Details',b:d.body||'—'}]);
      h+=`<div class="card ${SC[d.status]||'c-gy'}" onclick="om('${mid}')">
        <div class="card-hdr"><span class="card-ico">${SI[d.status]||'·'}</span><span class="card-name">${esc(d.name)}</span>${sb(d.status)}</div>
        <div class="card-kv">
          ${d.adr?`<div class="kv-row"><span class="kv-k">ADR</span><span class="kv-v">${esc(d.adr)}</span></div>`:''}
          <div class="kv-row"><span class="kv-k">Current</span><span class="kv-v">${esc((d.current_behavior||'').slice(0,90))}…</span></div>
        </div></div>`;
    }
    h+=`</div></div>`;
  }
  document.getElementById('debt-root').innerHTML=h||'<div class="empty">No debt.</div>';
}

function renderAdrs(){
  const adrs=DATA.adrs||[];
  let h='';
  for(const a of adrs){
    const sc=a.status.toLowerCase()==='accepted'?'var(--green)':a.status.toLowerCase()==='proposed'?'var(--purple)':'var(--muted)';
    const mid=rm('📋',a.title,a.id+' · '+a.status,[{t:'Context',b:a.context},{t:'Decision',b:a.decision}]);
    h+=`<tr onclick="om('${mid}')"><td class="mono">${esc(a.id)}</td>
      <td style="color:var(--text);font-weight:500">${esc(a.title)}</td>
      <td><span style="color:${sc};font-weight:600;font-size:9px">${esc(a.status)}</span></td>
      <td class="mono">${esc(a.date)}</td>
      <td>${esc((a.decision||'').split('\n')[0].slice(0,80))}…</td></tr>`;
  }
  document.getElementById('adr-tbody').innerHTML=h;
}

function renderInv(){
  const inv=DATA.state.runtime_invariants||[];
  let h='';
  for(const item of inv){
    const mid=rm('🔒',item.name,'invariant',[{t:'Rule',b:item.rule},{t:'Details',b:item.body}]);
    h+=`<div class="card c-bl" onclick="om('${mid}')">
      <div class="card-hdr"><span class="card-ico">🔒</span><span class="card-name">${esc(item.name)}</span></div>
      <div class="card-body">${esc(item.rule||item.body.slice(0,140))}…</div>
      ${item.adr?`<div class="card-kv"><div class="kv-row"><span class="kv-k">ADR</span><span class="kv-v">${esc(item.adr)}</span></div></div>`:''}
    </div>`;
  }
  document.getElementById('inv-grid').innerHTML=h||'<div class="empty">No invariants.</div>';
}

async function loadData(){
  try{
    const r=await fetch('/api/data');DATA=await r.json();
    for(const k in MODALS)delete MODALS[k];mid=0;
    renderHeader();renderStats();renderUpdates();renderCaps();renderDebt();renderAdrs();renderInv();
  }catch(e){
    document.getElementById('section-ov').innerHTML=`<div style="color:var(--red);padding:24px">Error: ${esc(e.message)}</div>`;
  }
}
loadData();
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# HTTP Server
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            self._send(200, 'text/html; charset=utf-8', DASH_HTML.encode('utf-8'))
        elif self.path == '/map':
            self._send(200, 'text/html; charset=utf-8', MAP_HTML.encode('utf-8'))
        elif self.path == '/api/data':
            body = json.dumps(build_data(), ensure_ascii=False, default=str).encode('utf-8')
            self._send(200, 'application/json; charset=utf-8', body, cache=False)
        elif self.path == '/favicon.ico':
            self.send_response(204); self.end_headers()
        else:
            self._send(404, 'text/plain', b'Not found')

    def _send(self, code, ctype, body, cache=True):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        if not cache: self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception: pass

    parser = argparse.ArgumentParser(description='BPM Prediction Project Dashboard')
    parser.add_argument('--port', type=int, default=7878)
    parser.add_argument('--host', default='127.0.0.1')
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), Handler)
    url = f'http://{args.host}:{args.port}'
    print('')
    print('  BPM Prediction Dashboard')
    print('  ----------------------------------------')
    print(f'  Dashboard:  {url}/')
    print(f'  Metro Map:  {url}/map')
    print(f'  Project:    {PROJECT_ROOT}')
    print('')
    print('  Press Ctrl+C to stop.')
    print('')

    try: server.serve_forever()
    except KeyboardInterrupt: print('\n  Stopped.')


if __name__ == '__main__':
    main()
