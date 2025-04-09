const socket = new WebSocket(
	'ws://192.168.1.247:8765/socket.io/?EIO=4&transport=websocket'
);

socket.onopen = () => {
	console.log('Connected to the WebSocket server');
};

socket.onmessage = (event) => {
	try {
		const data = JSON.parse(event.data);
		if (data.type === 'text') {
			console.log('Received text:', data.data);
		}
	} catch (error) {
		console.error('Error parsing message:', error);
	}
};

socket.onerror = (error) => {
	console.error('WebSocket error:', error);
};

socket.onclose = () => {
	console.log('WebSocket connection closed');
};
