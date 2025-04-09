const socket = new WebSocket("ws://localhost:8765");

socket.onopen = () => {
  console.log("Connected to the WebSocket server");
};

socket.onmessage = (event) => {
  try {
    const data = JSON.parse(event.data);
    console.log("Received text:", data.data);
  } catch (error) {
    console.error("Error parsing message:", error);
  }
};

socket.onerror = (error) => {
  console.error("WebSocket error:", error);
};

socket.onclose = () => {
  console.log("WebSocket connection closed");
};
