# System Architecture Document (SAD)

**Project:** BPM Prediction Platform (`bpm_prediction`)  
**Status:** Draft v1.0  
**Authors:** Serhii Korotenko  

Цей документ надає всебічний архітектурний огляд платформи прогнозування бізнес-процесів. Він призначений для розробників, архітекторів та дослідників.

- [System Architecture Document (SAD)](#system-architecture-document-sad)
  - [1. Business Context (Бізнес-контекст)](#1-business-context-бізнес-контекст)
    - [1.1. Основні бізнес-функції](#11-основні-бізнес-функції)
    - [1.2. Зацікавлені сторони (Stakeholders)](#12-зацікавлені-сторони-stakeholders)
  - [2. System Context (Контекстна діаграма)](#2-system-context-контекстна-діаграма)


---

## 1. Business Context (Бізнес-контекст)

### 1.1. Основні бізнес-функції
Система вирішує проблему невизначеності у виконанні бізнес-процесів, надаючи інструменти для:
1.  **Predictive Monitoring:** Прогнозування наступного кроку (Next Activity) та часу завершення (Time Prediction) для активних процесів.
2.  **Structural Drift Adaptation:** Забезпечення точності прогнозів навіть при зміні версій процесів завдяки POKG.
3.  **Process Simulation Benchmarking:** Наукове порівняння ефективності різних ML-архітектур (GNN vs LSTM).

### 1.2. Зацікавлені сторони (Stakeholders)
* **Бізнес-користувачі (BPMS):** Отримують підказки (Recommendations) та попередження про порушення SLA.
* **Data Scientists:** Використовують платформу для експериментів з новими архітектурами GNN.
* **Process Owners:** Аналізують розходження між планом (BPMN) та фактом (Logs).

---

## 2. System Context (Контекстна діаграма)

Система `bpm_prediction` є "інтелектуальним ядром", яке вбудовується в ландшафт підприємства.

```mermaid
C4Context
    title System Context Diagram for BPM Prediction Platform

    Person(admin, "Data Scientist / Admin", "Configures experiments, manages POKG")
    System(bpm_pred, "BPM Prediction Core", "Fusion Graph Engine + GNN Models")
    
    System_Ext(camunda, "Camunda BPMS", "Orchestrates processes, stores runtime data (SQL)")
    System_Ext(neo4j, "Neo4j Graph DB", "Stores EPOKG (Enterprise Process-Org Graph)")
    System_Ext(files, "File Storage", "XES Logs, BPMN files, Checkpoints")

    Rel(admin, bpm_pred, "Configures YAML, Runs Pipelines")
    Rel(bpm_pred, camunda, "Reads Logs (SQL/API), Returns Predictions (REST)")
    Rel(bpm_pred, neo4j, "Reads Context, Updates Statistics")
    Rel(bpm_pred, files, "Reads XES/BPMN, Saves Models")