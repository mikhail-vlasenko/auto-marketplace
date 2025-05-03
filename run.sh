curl -X POST http://localhost:8000/chat-message \
  -H "Content-Type: multipart/form-data" \
  -F "user_id=alice123" \
  -F "text=Here’s my item for sale" \
  -F "images=@bike 1.jpg;type=image/jpeg" \
  -F "images=@bike 2.jpg;type=image/jpeg"
