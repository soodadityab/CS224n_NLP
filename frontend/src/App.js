import React, { useState } from "react";
import axios from "axios";
import "./App.css";

export default function App() {
  const [prompt, setPrompt] = useState("");
  const [story, setStory] = useState("");
  const [feedback, setFeedback] = useState(null);

  const generateStory = async () => {
    try {
      const response = await axios.post("http://localhost:5000/generate", {
        prompt,
      });
      setStory(response.data.story);
    } catch (error) {
      console.error("Error generating story:", error);
    }
  };

  const submitFeedback = async (feedbackValue) => {
    try {
      await axios.post("http://localhost:5000/feedback", {
        feedback: feedbackValue,
      });
      setFeedback(feedbackValue);
    } catch (error) {
      console.error("Error submitting feedback:", error);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>ReaL Stories: RL for Adaptive AI Storytelling</h1>
        <textarea
          id="prompt"
          placeholder="Enter your prompt here"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />
        <button onClick={generateStory}>Generate Story</button>
        <div id="story">{story}</div>
        {story && (
          <div id="feedback">
            <button onClick={() => submitFeedback(1)}>👍</button>
            <button onClick={() => submitFeedback(0)}>👎</button>
          </div>
        )}
        {feedback !== null && (
          <p>Feedback submitted: {feedback === 1 ? "👍" : "👎"}</p>
        )}
      </header>
      <footer>
        <p>Developed by: Aditya, Aniket, and Ayaan</p>
        <p>CS224N: Deep Learning for NLP Final Project</p>
      </footer>
    </div>
  );
}
