import React, { useState, useRef, useEffect } from 'react';


/* ------------------------------------------------------------------ */
/*  Constants                                                           */
/* ------------------------------------------------------------------ */

const CHAR_LIMIT    = 500;   // Maximum characters per user message (including spaces)
const MSG_LIMIT     = 15;    // Maximum user messages per session
const UNDO_DEPTH    = 7;     // Maximum number of word-boundary undo steps


/* ------------------------------------------------------------------ */
/*  Icons                                                               */
/* ------------------------------------------------------------------ */

/* The following function is responsible for rendering the close (X) icon
   used in the chatbot dismiss button. */
function CloseIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
      <path
        d="M11 3L3 11M3 3L11 11"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
      />
    </svg>
  );
}


/* The following function is responsible for rendering the speech bubble
   icon used in the floating chatbot toggle button. */
function ChatIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 22 22" fill="none" aria-hidden="true">
      <path
        d="M11 2C6.03 2 2 5.58 2 10C2 12.14 2.9 14.1 4.42 15.54L3 20L7.8 18.3C8.8 18.6 9.88 18.76 11 18.76C15.97 18.76 20 15.18 20 10.38C20 5.58 15.97 2 11 2Z"
        fill="white"
      />
    </svg>
  );
}


/* The following function is responsible for rendering the send arrow icon
   used in the chatbot send button. */
function SendIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 15 15" fill="none" aria-hidden="true">
      <path
        d="M13 2L2 7L6.5 9M13 2L8.5 13L6.5 9M13 2L6.5 9"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}


/* ------------------------------------------------------------------ */
/*  Typing indicator                                                    */
/* ------------------------------------------------------------------ */

/* The following function is responsible for rendering the animated
   three-dot typing indicator shown while the AI is generating a reply. */
function TypingIndicator() {
  return (
    <div className="chat-message assistant">
      <div className="chat-typing-bubble">
        <span className="chat-typing-dot" />
        <span className="chat-typing-dot" />
        <span className="chat-typing-dot" />
      </div>
    </div>
  );
}


/* ------------------------------------------------------------------ */
/*  Main ChatBot component                                              */
/* ------------------------------------------------------------------ */

/* The following function is responsible for rendering the full chatbot
   overlay including the floating toggle button and the slide-in panel.
   It manages message history, character limits, session message limits,
   custom word-based undo, and API communication with the Gemma backend. */
function ChatBot() {

  const [isOpen,    setIsOpen]    = useState(false);
  const [messages,  setMessages]  = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [msgCount,  setMsgCount]  = useState(0);

  const messagesEndRef    = useRef(null);
  const textareaRef       = useRef(null);
  // undoStackRef stores up to UNDO_DEPTH textarea value snapshots.
  const undoStackRef      = useRef([]);
  // lastPasteRef flags that the last action was a paste so handleChange
  // skips adding a redundant word-boundary snapshot.
  const lastPasteRef      = useRef(false);

  const isLimitReached = msgCount >= MSG_LIMIT;
  const canSend        = inputValue.trim().length > 0 && !isLoading && !isLimitReached;


  /* The following effect is responsible for scrolling the message list
     to the bottom whenever new messages are added. */
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);


  /* The following function is responsible for opening the chatbot panel. */
  function openChat() {
    setIsOpen(true);
  }


  /* The following function is responsible for dismissing the chatbot panel. */
  function closeChat() {
    setIsOpen(false);
  }


  /* The following function is responsible for intercepting keyboard events
     on the textarea to handle:
     - Enter (without Shift): send the message
     - Ctrl/Cmd + Z: custom word-boundary undo (up to UNDO_DEPTH steps) */
  function handleKeyDown(e) {

    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (canSend) sendMessage();
      return;
    }

    if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
      e.preventDefault();
      if (undoStackRef.current.length > 0) {
        const previous = undoStackRef.current.pop();
        setInputValue(previous);
        lastPasteRef.current = false;
      }
      return;
    }

  }


  /* The following function is responsible for handling textarea value
     changes. When the user completes a word (types a space), a snapshot
     of the current value is pushed onto the undo stack so that Ctrl+Z
     can restore it word by word. Paste actions bypass this so the entire
     pasted content is undone in one step. */
  function handleChange(e) {

    const newValue = e.target.value.slice(0, CHAR_LIMIT);

    if (!lastPasteRef.current && newValue.endsWith(' ')) {
      const stack = undoStackRef.current;
      if (stack.length >= UNDO_DEPTH) stack.shift();
      stack.push(inputValue);
    }

    lastPasteRef.current = false;
    setInputValue(newValue);

  }


  /* The following function is responsible for handling paste events on
     the textarea. It saves a snapshot of the pre-paste value so that a
     single Ctrl+Z undoes the entire pasted content at once. The 500
     character limit is applied after the paste is inserted. */
  function handlePaste() {
    const stack = undoStackRef.current;
    if (stack.length >= UNDO_DEPTH) stack.shift();
    stack.push(inputValue);
    lastPasteRef.current = true;
  }


  /* The following function is responsible for sending the current input
     as a user message, calling the backend chat endpoint with the full
     conversation history, and appending the AI reply to the message list.
     The first message is sent with no prior context; subsequent messages
     carry the growing history so the model retains conversation context. */
  async function sendMessage() {

    if (!canSend) return;

    const userText = inputValue.trim();

    const newMessages = [
      ...messages,
      { role: 'user', content: userText },
    ];

    setMessages(newMessages);
    setInputValue('');
    setMsgCount(c => c + 1);
    undoStackRef.current = [];
    setIsLoading(true);

    try {

      const response = await fetch('/api/llm/chat', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: newMessages }),
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: 'Request failed.' }));
        throw new Error(err.detail || 'Chat request failed.');
      }

      const data = await response.json();

      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: data.reply },
      ]);

    } catch (err) {
      setMessages(prev => [
        ...prev,
        {
          role:    'assistant',
          content: `Something went wrong: ${err.message}`,
        },
      ]);
    } finally {
      setIsLoading(false);
    }

  }

  const charsLeft      = CHAR_LIMIT - inputValue.length;
  const msgsLeft       = MSG_LIMIT  - msgCount;
  const charNearLimit  = charsLeft <= 50;


  /* ---------------------------------------------------------------- */
  /*  Render                                                           */
  /* ---------------------------------------------------------------- */

  return (
    <>

      {/* Floating toggle button */}
      <button
        className={isOpen ? 'chatbot-toggle-btn hidden' : 'chatbot-toggle-btn'}
        onClick={openChat}
        aria-label="Open DermAI chat assistant"
      >
        <ChatIcon />
      </button>

      {/* Slide-in chat panel */}
      <div
        className={isOpen ? 'chatbot-container open' : 'chatbot-container'}
        role="dialog"
        aria-label="DermAI chat assistant"
        aria-modal="false"
      >

        {/* Header */}
        <div className="chatbot-header">
          <div className="chatbot-header-info">
            <span className="chatbot-header-title">Karl — Your DermAI Assistant</span>
            <span className="chatbot-header-status">Dermatology specialist</span>
          </div>
          <button
            className="chatbot-dismiss-btn"
            onClick={closeChat}
            aria-label="Close chat assistant"
          >
            <CloseIcon />
          </button>
        </div>

        {/* Message list */}
        <div className="chatbot-body">

          {messages.length === 0 && !isLoading && (
            <p className="chatbot-placeholder">
              Ask Karl anything about skin health and dermatology.
            </p>
          )}

          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`chat-message ${msg.role}`}
            >
              <div className={`chat-bubble ${msg.role}`}>
                {msg.content}
              </div>
            </div>
          ))}

          {isLoading && <TypingIndicator />}

          <div ref={messagesEndRef} />

        </div>

        {/* Input footer */}
        <div className="chatbot-footer">

          {isLimitReached ? (
            <p className="chatbot-limit-banner">
              Session limit of {MSG_LIMIT} messages reached.
              Refresh the page to start a new conversation.
            </p>
          ) : (
            <>
              <div className="chatbot-input-row">
                <textarea
                  ref={textareaRef}
                  className="chatbot-textarea"
                  placeholder="Type your message..."
                  value={inputValue}
                  onChange={handleChange}
                  onKeyDown={handleKeyDown}
                  onPaste={handlePaste}
                  disabled={isLoading || isLimitReached}
                  maxLength={CHAR_LIMIT}
                  rows={2}
                  aria-label="Chat message input"
                />
                <button
                  className="chatbot-send-btn"
                  onClick={sendMessage}
                  disabled={!canSend}
                  aria-label="Send message"
                >
                  <SendIcon />
                </button>
              </div>

              <div className="chatbot-meta-row">
                <span className={`chatbot-char-count${charNearLimit ? ' near-limit' : ''}`}>
                  {charsLeft} chars left
                </span>
                <span className="chatbot-msg-count">
                  {msgsLeft} message{msgsLeft !== 1 ? 's' : ''} remaining
                </span>
              </div>
            </>
          )}

        </div>

      </div>

    </>
  );

}

export default ChatBot;
