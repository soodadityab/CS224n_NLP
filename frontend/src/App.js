import React, { useState } from "react";
import axios from "axios";
import "./App.css";

export default function App() {
  const [prompt, setPrompt] = useState("");
  const [completions, setCompletions] = useState([]);
  const [feedback, setFeedback] = useState([]);

  const generateStory = async () => {
    try {
      const response = await axios.post("http://127.0.0.1:5000/generate", {
        prompt,
      });
      console.log(response.data);
      setCompletions(response.data.completions);
      setFeedback(Array(response.data.completions.length).fill(null));
    } catch (error) {
      console.error("Could not generate story:", error);
    }
  };

  const resetAll = () => {
    setPrompt("");
    setCompletions([]);
    setFeedback([]);
  };

  const submitFeedback = (index, value) => {
    const userInput = feedback.map((item, i) => {
      if (i === index) {
        return value;
      }
      return value === 1 ? 0 : value === 0 ? 1 : item;
    });
    setFeedback(userInput);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>ReaL Stories: RL for Adaptive AI Storytelling</h1>
        <textarea
          id="prompt"
          placeholder="Enter your prompt here"
          value={prompt}
          onChange={(item) => setPrompt(item.target.value)}
        />
        <button onClick={generateStory}>Generate Story</button>
        <button onClick={resetAll}>Reset</button>
        <div id="story-container">
          {completions.map((story, counter) => (
            <div key={counter} className="story-block">
              <div className="feedback-buttons">
                <button
                  className={feedback[counter] === 1 ? "selected" : ""}
                  onClick={() => submitFeedback(counter, 1)}
                >
                  👍
                </button>
                <button
                  className={feedback[counter] === 0 ? "selected" : ""}
                  onClick={() => submitFeedback(counter, 0)}
                >
                  👎
                </button>
              </div>
              <h2>Completion {counter + 1}</h2>
              <p>{story}</p>
              {feedback[counter] !== null && (
                <p>
                  Feedback submitted: {feedback[counter] === 1 ? "👍" : "👎"}
                </p>
              )}
            </div>
          ))}
        </div>
      </header>
      <footer>
        <p>Developed by: Aditya, Aniket, and Ayaan</p>
        <p>CS224N: Deep Learning for NLP Final Project</p>
      </footer>
    </div>
  );
}
