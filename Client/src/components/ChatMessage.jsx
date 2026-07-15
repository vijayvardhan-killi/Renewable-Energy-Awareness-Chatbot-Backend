import React from "react";
import ReactMarkdown from "react-markdown";

const ChatMessage = ({ message }) => {
  return (
    <div className={`message ${message.type}`}>
      <div className="message-content">
        <ReactMarkdown>{message.content}</ReactMarkdown>
        <p className="message-time">
          {message.timestamp.toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit"
          })}
        </p>
      </div>
    </div>
  );
};

export default ChatMessage;