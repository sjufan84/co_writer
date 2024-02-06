async function sendMessage() {
    const userInput = document.getElementById('userInput').value;
    const chatBox = document.getElementById('chatBox');
  
    // Append user's message to chatBox
    chatBox.innerHTML += `<div>User: ${userInput}</div>`;
  
    // Send the user's message to the back-end and get LLM's response
    // Replace this with actual API call logic
    const llmResponse = await fetch('http://localhost:3000/chat', {
      method: 'POST',
      body: JSON.stringify({ message: userInput }),
      headers: { 'Content-Type': 'application/json' },
      // disable cors
        mode: 'no-cors',
    }).then(response => response.json());
  
    // Append LLM's response to chatBox
    chatBox.innerHTML += `<div>LLM: ${llmResponse.message}</div>`;
  }
  