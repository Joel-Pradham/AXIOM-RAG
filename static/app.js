document.addEventListener('DOMContentLoaded', () => {
    const chatForm       = document.getElementById('chat-form');
    const queryInput     = document.getElementById('query-input');
    const chatContainer  = document.getElementById('chat-container');
    const uploadBtn      = document.getElementById('upload-btn');
    const documentUpload = document.getElementById('document-upload');

    // Unique session per page-load
    const sessionId = 'conv_' + Date.now() + '_' + Math.floor(Math.random() * 9999);

    // ── Helpers ──────────────────────────────────────────────────────────────

    const scrollToBottom = () => {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    };

    const appendUserMessage = (text) => {
        const div       = document.createElement('div');
        div.className   = 'message msg-user';
        div.textContent = text;
        chatContainer.appendChild(div);
        scrollToBottom();
    };

    const appendTypingIndicator = (label = 'SYNTHESIZING VECTORS') => {
        const id      = 'msg-' + Date.now();
        const div     = document.createElement('div');
        div.className = 'message msg-tutor-group';
        div.id        = id;
        div.innerHTML = `
            <div class="typing-indicator">
                ${label} <span>.</span><span>.</span><span>.</span>
            </div>`;
        chatContainer.appendChild(div);
        scrollToBottom();
        return id;
    };

    const replaceWithTutorResponse = (id, data) => {
        const el = document.getElementById(id);
        if (!el) return;

        let citationsHtml = '';
        if (data.citations && data.citations.length > 0) {
            const chips = data.citations.map(c =>
                `<div class="citation-chip">
                    <svg viewBox="0 0 24 24" width="12" height="12" stroke="currentColor" stroke-width="2" fill="none"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                    ${c.course_name || 'Source'} · Ch ${c.chapter_number || '—'} · ${c.concept_tags || ''}
                </div>`
            ).join('');
            citationsHtml = `<div class="citations">${chips}</div>`;
        }

        el.innerHTML = `
            <div class="telemetry">
                <span class="telemetry-header">
                    <svg viewBox="0 0 24 24" width="12" height="12" stroke="currentColor" stroke-width="2" fill="none"><circle cx="12" cy="12" r="3"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/><path d="M4.93 4.93a10 10 0 0 0 0 14.14"/></svg>
                    Internal Telemetry
                </span>
                ${data.internal_thought_process}
            </div>
            <div class="ai-response">
                ${data.answer}
            </div>
            ${citationsHtml}`;
        scrollToBottom();
    };

    const showError = (id, message) => {
        const el = document.getElementById(id);
        if (!el) return;
        el.innerHTML = `
            <div class="ai-response error-response">
                <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" stroke-width="2" fill="none" style="margin-right:8px;flex-shrink:0"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                <span>${message}</span>
            </div>`;
        scrollToBottom();
    };

    const showSuccess = (id, message) => {
        const el = document.getElementById(id);
        if (!el) return;
        el.innerHTML = `
            <div class="ai-response success-response">
                <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" stroke-width="2" fill="none" style="margin-right:8px;flex-shrink:0"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                <span>${message}</span>
            </div>`;
        scrollToBottom();
    };

    // ── Chunked Upload ────────────────────────────────────────────────────────

    const CHUNK_SIZE = 3.5 * 1024 * 1024;  // 3.5 MB — safely under Vercel's 4.5 MB limit

    const updateProgress = (id, percent, label) => {
        const el = document.getElementById(id);
        if (!el) return;
        el.innerHTML = `
            <div class="upload-progress">
                <div class="upload-progress-label">${label}</div>
                <div class="upload-progress-bar-track">
                    <div class="upload-progress-bar-fill" style="width:${percent}%"></div>
                </div>
                <div class="upload-progress-pct">${Math.round(percent)}%</div>
            </div>`;
        scrollToBottom();
    };

    const uploadFileChunked = async (file, loaderId) => {
        const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
        const uploadId    = 'up_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8);

        updateProgress(loaderId, 0, `Uploading '${file.name}' — preparing…`);

        // ── Phase 1: Upload all chunks ──────────────────────────────────────
        for (let i = 0; i < totalChunks; i++) {
            const start = i * CHUNK_SIZE;
            const end   = Math.min(start + CHUNK_SIZE, file.size);
            const blob  = file.slice(start, end);

            const fd = new FormData();
            fd.append('file',         new File([blob], file.name, { type: file.type }));
            fd.append('upload_id',    uploadId);
            fd.append('chunk_index',  String(i));
            fd.append('total_chunks', String(totalChunks));
            fd.append('filename',     file.name);

            const pct   = ((i + 1) / totalChunks) * 60;  // 0-60% for upload phase
            updateProgress(loaderId, pct,
                `Uploading '${file.name}' — chunk ${i + 1}/${totalChunks}`);

            let resp;
            try {
                resp = await fetch('/api/upload/chunk', { method: 'POST', body: fd });
            } catch (netErr) {
                throw new Error(`Network error on chunk ${i + 1}: ${netErr.message}`);
            }

            if (!resp.ok) {
                const errData = await resp.json().catch(() => ({ detail: resp.statusText }));
                throw new Error(`Chunk ${i + 1} rejected: ${errData.detail || resp.statusText}`);
            }
        }

        // ── Phase 2: Finalize (reassemble + ingest) ─────────────────────────
        updateProgress(loaderId, 65, `Processing '${file.name}' — indexing document…`);

        const fd2 = new FormData();
        fd2.append('upload_id',    uploadId);
        fd2.append('total_chunks', String(totalChunks));

        let finalResp;
        try {
            finalResp = await fetch('/api/upload/finalize', { method: 'POST', body: fd2 });
        } catch (netErr) {
            throw new Error(`Network error during finalization: ${netErr.message}`);
        }

        updateProgress(loaderId, 90, `Finalising index…`);

        if (!finalResp.ok) {
            const errData = await finalResp.json().catch(() => ({ detail: finalResp.statusText }));
            throw new Error(errData.detail || `Server error ${finalResp.status}`);
        }

        const result = await finalResp.json();
        updateProgress(loaderId, 100, 'Complete');
        return result;
    };

    // ── Upload event ──────────────────────────────────────────────────────────

    uploadBtn.addEventListener('click', () => documentUpload.click());

    documentUpload.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const allowed = ['.pdf', '.txt', '.md'];
        const ext = '.' + file.name.split('.').pop().toLowerCase();
        if (!allowed.includes(ext)) {
            appendUserMessage(`Rejected '${file.name}' — only PDF, TXT, and MD files are supported.`);
            documentUpload.value = '';
            return;
        }

        uploadBtn.disabled = true;
        uploadBtn.classList.add('uploading');
        appendUserMessage(`Uploading: ${file.name} (${(file.size / 1_048_576).toFixed(1)} MB)`);

        const loaderId = appendTypingIndicator('INGESTING DATA');

        try {
            const result = await uploadFileChunked(file, loaderId);
            showSuccess(loaderId,
                `${result.message}<br>
                 <small style="opacity:0.7">${result.size_mb} MB · ${result.num_chunks} indexed chunks</small>`
            );
        } catch (err) {
            showError(loaderId, `Upload failed: ${err.message}`);
        } finally {
            uploadBtn.disabled = false;
            uploadBtn.classList.remove('uploading');
            documentUpload.value = '';
        }
    });

    // ── Chat ──────────────────────────────────────────────────────────────────

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = queryInput.value.trim();
        if (!query) return;

        appendUserMessage(query);
        queryInput.value = '';

        const loaderId = appendTypingIndicator();

        try {
            const resp = await fetch('/api/chat', {
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify({ query, session_id: sessionId }),
            });

            if (!resp.ok) {
                const err = await resp.json().catch(() => ({ detail: 'Server error' }));
                throw new Error(err.detail || `HTTP ${resp.status}`);
            }

            const data = await resp.json();
            replaceWithTutorResponse(loaderId, data);

        } catch (err) {
            showError(loaderId, `Could not reach the intelligence unit: ${err.message}`);
        }
    });

    // ── Enter key sends (Shift+Enter for newline) ─────────────────────────────
    queryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event('submit'));
        }
    });
});
