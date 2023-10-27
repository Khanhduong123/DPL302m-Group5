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
      
          // Send user message
          fetch('http://127.0.0.1:5000/post', {
            method: 'POST', 
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({
              message: messageText  
            })
          })
          .then(response => response.json())
          .then(data => {
            
            // Bot response  
            const receivedMessageElement = document.createElement("div");
            receivedMessageElement.className = "message-bubble received";
      
            const receivedMessageTextElement = document.createElement("div");
            receivedMessageTextElement.className = "message-text";
      
            receivedMessageTextElement.textContent = data.response;
      
            receivedMessageElement.appendChild(receivedMessageTextElement);
      
            chatMessages.appendChild(receivedMessageElement);
      
            chatMessages.scrollTop = chatMessages.scrollHeight;
      
          })
          .catch(error => {
            console.error('Error getting bot response', error); 
          });
            // user input
            const messageElement = document.createElement("div");
            messageElement.className = "message-bubble sent";

            const messageTextElement = document.createElement("div");
            messageTextElement.className = "message-text";
            messageTextElement.textContent = messageText;

            messageElement.appendChild(messageTextElement);

            chatMessages.appendChild(messageElement);
      
          // Clear input
          messageInput.value = "";
      
        }
      });

    // Allow sending messages by pressing Enter key
    messageInput.addEventListener("keydown", function(event) {
        if (event.key === "Enter") {
            sendButton.click();
        }
    });
});
