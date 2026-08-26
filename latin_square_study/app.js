"use strict";

const state = {
  materials: [],
  tasks: [],
  taskRecords: [],
  activeTaskIndex: 0,
  studyStartedAt: null,
};

const $ = (selector) => document.querySelector(selector);

function formatTime(seconds) {
  const wholeSeconds = Math.round(Number(seconds));
  const minutes = Math.floor(wholeSeconds / 60);
  const remainingSeconds = wholeSeconds % 60;
  const hours = Math.floor(minutes / 60);
  return hours
    ? `${String(hours).padStart(2, "0")}:${String(minutes % 60).padStart(2, "0")}:${String(remainingSeconds).padStart(2, "0")}`
    : `${String(minutes).padStart(2, "0")}:${String(remainingSeconds).padStart(2, "0")}`;
}

async function fetchJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(`Could not load ${path}.`);
  return response.json();
}

function validateMaterial(material) {
  return material && typeof material.video_id === "string" && Array.isArray(material.chapters);
}

function validateTasks(taskDefinition) {
  return taskDefinition && Array.isArray(taskDefinition.tasks) && taskDefinition.tasks.length === 3;
}

function getTaskForMaterial(material) {
  return state.tasks.find((task) => task.video_id === material.video_id);
}

function renderProgress() {
  const nav = $("#progress-nav");
  nav.replaceChildren();
  [...state.materials, "Feedback"].forEach((item, index) => {
    const label = typeof item === "string" ? item : `Task ${index + 1}`;
    const element = document.createElement("span");
    element.className = `progress-item${index === state.activeTaskIndex ? " active" : ""}${index < state.activeTaskIndex ? " done" : ""}`;
    element.textContent = label;
    nav.append(element);
  });
}

function createTaskRecord(material, task) {
  return {
    task_number: state.activeTaskIndex + 1,
    video_id: material.video_id,
    task_id: task.id || `task_${state.activeTaskIndex + 1}`,
    prompt: task.prompt,
    anonymized_segmentation_method: material.segmentation_method || null,
    started_at: new Date().toISOString(),
    completed_at: null,
    duration_seconds: null,
    selected_chapter_indices: [],
    selected_chapters: [],
    interaction_counts: {
      selection_changes: 0,
      video_link_clicks: 0,
      accordion_opens: { keywords: 0, summary: 0, transcript: 0 },
    },
  };
}

function renderTask() {
  const material = state.materials[state.activeTaskIndex];
  const task = getTaskForMaterial(material);
  const taskView = $("#task-view");
  const fragment = $("#task-template").content.cloneNode(true);
  const page = fragment.querySelector(".task-page");
  const record = createTaskRecord(material, task);
  state.taskRecords[state.activeTaskIndex] = record;

  fragment.querySelector(".task-number").textContent = `Task ${state.activeTaskIndex + 1} of 3`;
  fragment.querySelector(".task-title").textContent = task.title || `Video ${state.activeTaskIndex + 1}`;
  fragment.querySelector(".task-prompt").textContent = task.prompt;
  const videoLink = fragment.querySelector(".video-link");
  if (task.video_url) {
    videoLink.href = task.video_url;
    videoLink.textContent = task.video_label || "Open video";
    videoLink.classList.remove("hidden");
    videoLink.addEventListener("click", () => { record.interaction_counts.video_link_clicks += 1; });
  }

  const chapterList = fragment.querySelector(".chapter-list");
  material.chapters.forEach((chapter, chapterIndex) => {
    const chapterFragment = $("#chapter-template").content.cloneNode(true);
    const checkbox = chapterFragment.querySelector(".chapter-select");
    checkbox.value = String(chapterIndex);
    checkbox.setAttribute("aria-label", `Select chapter ${chapterIndex + 1}`);
    chapterFragment.querySelector(".chapter-time").textContent = `${chapterIndex + 1}. ${formatTime(chapter.start)} to ${formatTime(chapter.end)}`;
    chapterFragment.querySelector(".chapter-title").textContent = chapter.title || "Untitled segment";
    chapterFragment.querySelector(".keywords").textContent = (chapter.keywords || []).join(", ") || "No keywords available.";
    chapterFragment.querySelector(".chapter-summary").textContent = chapter.summary || "No summary available.";
    chapterFragment.querySelector(".transcript").textContent = chapter.transcript || "No transcript available.";
    checkbox.addEventListener("change", () => {
      record.interaction_counts.selection_changes += 1;
      updateSelectionCount(page);
    });
    chapterFragment.querySelectorAll("details").forEach((details) => {
      details.addEventListener("toggle", () => {
        if (details.open) record.interaction_counts.accordion_opens[details.dataset.section] += 1;
      });
    });
    chapterList.append(chapterFragment);
  });

  fragment.querySelector(".finish-task-button").addEventListener("click", () => completeTask(page, material, record));
  taskView.replaceChildren(fragment);
  $("#feedback-view").classList.add("hidden");
  renderProgress();
  updateSelectionCount(page);
}

function updateSelectionCount(page) {
  const count = page.querySelectorAll(".chapter-select:checked").length;
  page.querySelector(".selection-count").textContent = `${count} segment${count === 1 ? "" : "s"} selected`;
}

function completeTask(page, material, record) {
  const selectedIndices = [...page.querySelectorAll(".chapter-select:checked")].map((input) => Number(input.value));
  if (!selectedIndices.length) {
    window.alert("Select at least one segment before continuing.");
    return;
  }
  const completedAt = new Date();
  record.completed_at = completedAt.toISOString();
  record.duration_seconds = Math.round((completedAt.getTime() - new Date(record.started_at).getTime()) / 1000);
  record.selected_chapter_indices = selectedIndices.map((index) => index + 1);
  record.selected_chapters = selectedIndices.map((index) => {
    const chapter = material.chapters[index];
    return { chapter_index: index + 1, start: chapter.start, end: chapter.end, title: chapter.title };
  });
  state.activeTaskIndex += 1;
  if (state.activeTaskIndex < state.materials.length) renderTask();
  else renderFeedback();
}

function renderFeedback() {
  $("#task-view").replaceChildren();
  const feedbackView = $("#feedback-view");
  const fragment = $("#feedback-template").content.cloneNode(true);
  const easiestOptions = fragment.querySelector("#easiest-task-options");
  state.taskRecords.forEach((record) => {
    const label = document.createElement("label");
    label.innerHTML = `<input type="radio" name="easiest_task" value="${record.task_number}" required /> Task ${record.task_number}`;
    easiestOptions.append(label);
  });
  const form = fragment.querySelector("#feedback-form");
  const helpfulnessFieldset = fragment.querySelector("#video-helpfulness-fieldset");
  form.elements.watched_videos.forEach((input) => input.addEventListener("change", () => {
    const watched = form.elements.watched_videos.value === "yes";
    helpfulnessFieldset.classList.toggle("hidden", !watched);
    form.elements.video_helpfulness.forEach((option) => { option.required = watched; });
  }));
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    downloadResponse(new FormData(form));
  });
  feedbackView.replaceChildren(fragment);
  feedbackView.classList.remove("hidden");
  renderProgress();
  $("#status-text").textContent = "All task selections saved. Complete the feedback form to download your response.";
}

function downloadResponse(formData) {
  const completedAt = new Date();
  const feedback = Object.fromEntries(formData.entries());
  const response = {
    schema_version: 1,
    generated_at: completedAt.toISOString(),
    study_started_at: state.studyStartedAt,
    study_duration_seconds: Math.round((completedAt.getTime() - new Date(state.studyStartedAt).getTime()) / 1000),
    participant: state.materials[0].participant || null,
    tasks: state.taskRecords,
    feedback,
  };
  const blob = new Blob([JSON.stringify(response, null, 2) + "\n"], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `latin_square_response_${new Date().toISOString().replaceAll(":", "-")}.json`;
  link.click();
  URL.revokeObjectURL(url);
  $("#status-text").textContent = "Response downloaded. Send the downloaded JSON file to the researcher.";
}

async function startStudy() {
  try {
    const taskDefinition = await fetchJson("task_template.json");
    if (!validateTasks(taskDefinition)) throw new Error("The task file must contain exactly three tasks.");
    if (taskDefinition.tasks.some((task) => typeof task.material_file !== "string" || !task.material_file)) {
      throw new Error("Each task in task_template.json needs a material_file path.");
    }
    const materials = await Promise.all(taskDefinition.tasks.map((task) => fetchJson(task.material_file)));
    if (!materials.every(validateMaterial)) throw new Error("One or more assigned material files do not match the expected study format.");
    const participantIds = new Set(materials.map((material) => material.participant));
    if (participantIds.size !== 1) throw new Error("All three materials must belong to the same participant.");
    const videoIds = new Set(materials.map((material) => material.video_id));
    if (videoIds.size !== 3) throw new Error("The three materials must use three different videos.");
    state.tasks = taskDefinition.tasks;
    if (materials.some((material) => !getTaskForMaterial(material))) throw new Error("Each material video_id needs a matching task in the task-definition JSON.");
    state.materials = state.tasks.map((task) => materials.find((material) => material.video_id === task.video_id));
    state.activeTaskIndex = 0;
    state.taskRecords = [];
    state.studyStartedAt = new Date().toISOString();
    $("#loading-view").classList.add("hidden");
    $("#study-view").classList.remove("hidden");
    $("#status-text").textContent = "Complete all three tasks, then download your response.";
    renderTask();
  } catch (exception) {
    const message = exception instanceof Error ? exception.message : "Unable to load study files.";
    $("#loading-message").textContent = `${message} Start the study with start_study.bat, then open the browser window it launches.`;
    $("#status-text").textContent = "Study materials could not be loaded.";
  }
}

startStudy();