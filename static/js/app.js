function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename || "transcript_results.zip";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function setProgress({visible, percent, label, note, indeterminate}) {
  const container = document.getElementById("progress-container");
  const fill = document.getElementById("progress-fill");
  const percentEl = document.getElementById("progress-percent");
  const labelEl = document.getElementById("progress-label");
  const noteEl = document.getElementById("progress-note");
  const bar = document.querySelector(".progress__bar");

  if (!container || !fill || !percentEl || !labelEl || !noteEl || !bar) return;

  if (visible) container.classList.remove("hidden"); else container.classList.add("hidden");

  if (typeof label === "string") labelEl.textContent = label;
  if (typeof note === "string") noteEl.textContent = note;

  if (indeterminate) {
    fill.classList.add("indeterminate");
    bar.setAttribute("aria-busy", "true");
    bar.setAttribute("aria-valuenow", "0");
    percentEl.textContent = "";
  } else {
    fill.classList.remove("indeterminate");
    const p = Math.max(0, Math.min(100, percent || 0));
    fill.style.width = p + "%";
    bar.setAttribute("aria-busy", "false");
    bar.setAttribute("aria-valuenow", String(p));
    percentEl.textContent = p ? (p + "%") : "";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("transcribe-form");
  const btn = document.getElementById("submit-btn");
  if (!form || !btn) return;

  form.addEventListener("submit", (ev) => {
    ev.preventDefault();

    const formData = new FormData(form);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", form.action, true);
    xhr.responseType = "blob";

    // UI: disable button & show progress
    btn.disabled = true;
    btn.dataset.originalText = btn.textContent;
    btn.textContent = "Kör transkribering...";
    setProgress({visible: true, percent: 0, label: "Laddar upp...", note: "Skickar filen till servern...", indeterminate: false});

    // Upload progress
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        const p = Math.round((e.loaded / e.total) * 100);
        setProgress({visible: true, percent: p, label: "Laddar upp...", note: `Uppladdning: ${p}%`, indeterminate: false});
      }
    };

    // After upload completes, server processes request: switch to indeterminate
    xhr.upload.onload = () => {
      setProgress({visible: true, percent: 100, label: "Bearbetar...", note: "Transkriberar och skapar filer...", indeterminate: true});
    };

    xhr.onerror = () => {
      alert("Något gick fel vid anslutningen.");
      resetUI();
    };

    xhr.onload = () => {
      if (xhr.status === 200) {
        // Try to infer filename from Content-Disposition
        let filename = "transcript_results.zip";
        const disp = xhr.getResponseHeader("Content-Disposition") || "";
        const match = /filename\*=UTF-8''([^;\n]+)|filename=([^;\n]+)/i.exec(disp);
        if (match) {
          filename = decodeURIComponent((match[1] || match[2] || "").replace(/["']/g, "")) || filename;
        }
        downloadBlob(xhr.response, filename);
        resetUI(true);
      } else {
        // Attempt to read text error (if Flask flashed)
        try {
          const reader = new FileReader();
          reader.onload = function() {
            alert("Fel: " + (reader.result || "Okänt serverfel."));
          };
          reader.readAsText(xhr.response);
        } catch (e) {
          alert("Servern returnerade fel (" + xhr.status + ").");
        }
        resetUI();
      }
    };

    xhr.send(formData);

    function resetUI(success) {
      btn.disabled = false;
      btn.textContent = btn.dataset.originalText || "Kör transkribering";
      if (success) {
        setProgress({visible: false, percent: 0, label: "Klar", note: "", indeterminate: false});
      } else {
        setProgress({visible: true, percent: 0, label: "Klar med fel", note: "Försök igen eller kontrollera loggar.", indeterminate: false});
        setTimeout(() => setProgress({visible:false, percent:0, label:"", note:"", indeterminate:false}), 2000);
      }
    }
  });
});