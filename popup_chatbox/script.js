document.addEventListener("DOMContentLoaded", function() {
    const chatPopup = document.getElementById("chat-popup");
    const chatboxToggle = document.getElementById("chatbox-toggle");
    const chatboxClose = document.getElementById("chatbox-close");
    const messageInput = document.getElementById("message-input");
    const sendButton = document.getElementById("send-button");
    const chatMessages = document.getElementById("chat-messages");

    chatboxToggle.addEventListener("click", function() {
        chatPopup.style.display = "block";
        chatboxToggle.style.display = "none"; // Hide the button when chatbox is opened
    });

    chatboxClose.addEventListener("click", function() {
        chatPopup.style.display = "none";
        chatboxToggle.style.display = "block"; // Show the button when chatbox is closed
    });

    sendButton.addEventListener("click", function() {
        const messageText = messageInput.value.trim();
        if (messageText !== "") {
            const messageElement = document.createElement("div");
            messageElement.className = "message-bubble sent";
            const messageTextElement = document.createElement("div");
            messageTextElement.className = "message-text";
            messageTextElement.textContent = messageText;
            messageElement.appendChild(messageTextElement);
            chatMessages.appendChild(messageElement);
            chatMessages.scrollTop = chatMessages.scrollHeight; // Scroll to the bottom of the chat messages
            messageInput.value = ""; // Clear the input field

            // Simulate received message (for demonstration purposes)
            
        }
    });

    // Allow sending messages by pressing Enter key
    messageInput.addEventListener("keydown", function(event) {
        if (event.key === "Enter") {
            sendButton.click();
        }
    });
});
