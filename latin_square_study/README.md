# Latin Square Study Website

This folder is a self-contained study website. The participant does not configure or select any files.

## Researcher setup

1. In `task_template.json`, replace the three `video_id` values, write each task prompt, and set `material_file` to the matching material JSON path. `video_url` is optional; leave it empty when no video link should be shown.
2. Create a `materials/` subfolder in this folder. Copy the three assigned JSON files into it, then set their filenames in `material_file`. Keep each filename linked to the same `video_id`.
3. Give the participant this entire configured folder. They double-click `start_study.bat`; it opens the study in their default browser with all materials already loaded.
4. The browser downloads a response JSON after the participant submits feedback. Collect that file.

The local web server is needed only because browsers block JSON reads from a directly opened `file://` webpage. It does not upload data or require a network connection. All source materials stay on the participant's computer, and the final response is generated through a local browser download.

## Response contents

The downloaded response records the selected chapters for each task, task start/completion timestamps and durations, selection changes, accordion-open counts for keywords/summary/transcript, video-link clicks, and the final feedback form values.