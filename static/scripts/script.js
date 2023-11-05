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
      fetch('http://127.0.0.1:5000/', {
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
        // Display user input
        const userMessageElement = document.createElement("div");
        userMessageElement.className = "message-bubble sent";
    
        const userMessageTextElement = document.createElement("div");
        userMessageTextElement.className = "message-text";
        userMessageTextElement.textContent = data.user_message;  // Access the user input from the response
    
        userMessageElement.appendChild(userMessageTextElement);
    
        chatMessages.appendChild(userMessageElement);
    
        // Bot response
        const botMessageElement = document.createElement("div");
        botMessageElement.className = "message-bubble received";
    
        const botMessageTextElement = document.createElement("div");
        botMessageTextElement.className = "message-text";
        botMessageTextElement.textContent = data.bot_response;  // Access the bot response from the response
    
        botMessageElement.appendChild(botMessageTextElement);
    
        chatMessages.appendChild(botMessageElement);
    
        chatMessages.scrollTop = chatMessages.scrollHeight;
    })
    .catch(error => {
        console.error('Error getting bot response', error);
    });
  
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