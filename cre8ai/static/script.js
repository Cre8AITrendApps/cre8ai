document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatBox = document.getElementById('chat-box');
    const languageToggle = document.getElementById('language-toggle'); 
    
    let currentSessionId = `session_${Date.now()}`;
    let currentLanguage = 'en';

    function startNewSession() {
        chatBox.innerHTML = '';
        const welcome = currentLanguage === 'en' 
            ? "Hello! I'm the Cre8AI Assistant. How can I help you today? ✨" 
            : "مرحباً! أنا مساعد Cre8AI. كيف يمكنني مساعدتك اليوم؟ ✨";
        
        appendMessage(welcome, 'bot-message');
    }

    languageToggle.addEventListener('change', () => {
        currentLanguage = languageToggle.checked ? 'ar' : 'en';
        document.documentElement.dir = currentLanguage === 'ar' ? 'rtl' : 'ltr';
        document.getElementById('lang-label').textContent = currentLanguage === 'ar' ? 'العربية' : 'English';
        userInput.placeholder = currentLanguage === 'ar' ? 'اطرح سؤالاً...' : 'Ask a question...';
        startNewSession(); // This reloads the greeting in the correct language
    });

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = userInput.value.trim();
        if (!query) return;

        appendMessage(query, 'user-message');
        userInput.value = '';
        
        const thinking = appendMessage("...", "bot-message");

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    query: query,
                    session_id: currentSessionId,
                    client_id: "cre8ai", // Sent internally
                    language: currentLanguage 
                }),
            });

            const data = await response.json();
            thinking.remove();
            appendMessage(data.answer, 'bot-message');

        } catch (error) {
            thinking.remove();
            appendMessage("Error connecting to server.", "bot-message");
        }
    });

    function appendMessage(text, className) {
        const div = document.createElement('div');
        div.className = `message ${className}`;
        div.innerHTML = `<div class="content">${text}</div>`;
        chatBox.appendChild(div);
        chatBox.scrollTop = chatBox.scrollHeight;
        return div;
    }

    document.getElementById('refresh-button').addEventListener('click', startNewSession);
    startNewSession();
});
