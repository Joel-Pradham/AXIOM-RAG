document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const queryInput = document.getElementById('query-input');
    const chatContainer = document.getElementById('chat-container');

    // Generate unique memory tracker for this conversation flow
    const sessionId = 'conv_' + Date.now() + '_' + Math.floor(Math.random() * 1000);

    const appendUserMessage = (text) => {
        const div = document.createElement('div');
        div.className = 'message msg-user';
        div.textContent = text;
        chatContainer.appendChild(div);
        scrollToBottom();
    };

    const appendTypingIndicator = () => {
        const id = 'typing-' + Date.now();
        const div = document.createElement('div');
        div.className = 'message msg-tutor-group';
        div.id = id;
        div.innerHTML = `
            <div class="typing-indicator">
                SYNTHESIZING VECTORS <span>.</span><span>.</span><span>.</span>
            </div>
        `;
        chatContainer.appendChild(div);
        scrollToBottom();
        return id;
    };

    const replaceWithTutorResponse = (id, data) => {
        const placeholder = document.getElementById(id);
        
        let citationsHtml = '';
        if (data.citations && data.citations.length > 0) {
            const chipsHtml = data.citations.map(c => 
                `<div class="citation-chip">Source: ${c.course_name || 'Unknown'} | Ch: ${c.chapter_number || 'N'} - ${c.concept_tags || ''}</div>`
            ).join('');
            citationsHtml = `<div class="citations">${chipsHtml}</div>`;
        }

        placeholder.innerHTML = `
            <div class="telemetry">
                <span class="telemetry-header">Internal Telemetry</span>
                ${data.internal_thought_process}
            </div>
            <div class="ai-response">
                ${data.answer}
            </div>
            ${citationsHtml}
        `;
        scrollToBottom();
    };

    const scrollToBottom = () => {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    };

    const uploadBtn = document.getElementById('upload-btn');
    const documentUpload = document.getElementById('document-upload');

    uploadBtn.addEventListener('click', () => {
        documentUpload.click();
    });

    documentUpload.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        uploadBtn.classList.add('uploading');
        appendUserMessage(`Uploading document: ${file.name}...`);
        
        const loaderId = appendTypingIndicator();
        document.getElementById(loaderId).querySelector('.typing-indicator').innerHTML = `INGESTING DATA <span>.</span><span>.</span><span>.</span>`;

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Upload failed');
            
            const data = await response.json();
            document.getElementById(loaderId).innerHTML = `
                <div class="ai-response" style="border-color: var(--accent); background: var(--accent-dim);">
                    <strong>System Update:</strong> Document '${file.name}' ingested successfully. Generated ${data.num_chunks} chunks.
                </div>
            `;
            scrollToBottom();
        } catch (error) {
            document.getElementById(loaderId).innerHTML = `
                <div class="ai-response" style="border-color: var(--danger); background: rgba(255,0,0,0.1);">
                    <strong>Upload Error:</strong> System failed to process document.
                </div>
            `;
            scrollToBottom();
        } finally {
            uploadBtn.classList.remove('uploading');
            documentUpload.value = ''; // reset
        }
    });

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = queryInput.value.trim();
        if (!query) return;

        appendUserMessage(query);
        queryInput.value = '';
        
        const loaderId = appendTypingIndicator();

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query: query, session_id: sessionId })
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            replaceWithTutorResponse(loaderId, data);
            
        } catch (error) {
            document.getElementById(loaderId).innerHTML = `
                <div class="ai-response" style="border-color: var(--danger); background: rgba(255,0,0,0.1);">
                    <strong>Connection Error:</strong> Could not reach the intelligence unit. Please verify API configuration.
                </div>
            `;
            scrollToBottom();
        }
    });
});
