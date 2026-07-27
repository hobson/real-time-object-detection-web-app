// hotkeys.js — Ctrl+Enter (Cmd+Enter on Mac) to save
//
// Two-step save:
//  1. If an x-editable inline-edit popover is open, click its Save button
//     (commits the single cell without leaving the list view).
//  2. Otherwise click the first submit button on the page
//     (saves the full edit/create form and returns to the list).
//
// Loaded on every page via _BaseView.extra_js and _HomeView.extra_js.

document.addEventListener("keydown", function (e) {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        const inlineSave = document.querySelector(".editable-submit");
        if (inlineSave) {
            inlineSave.click();
        } else {
            const submit = document.querySelector(
                "input[type=submit], button[type=submit]"
            );
            if (submit) submit.click();
        }
    }
});
