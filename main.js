const socket = new WebSocket("ws://localhost:8765");

socket.onopen = () => {
  console.log("Connected to the WebSocket server");
};

socket.onmessage = (event) => {
  console.log("Received text:", event.data);
};

socket.onerror = (error) => {
  console.error("WebSocket error:", error);
};

socket.onclose = () => {
  console.log("WebSocket connection closed");
};
